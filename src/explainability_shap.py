"""
SHAP explainability analysis.

This script explains which features drive the reconstruction error of the
LSTM and Transformer autoencoders, using SHAP (KernelExplainer).

Steps:
    1. Load the fitted preprocessing pipeline and the benign data.
    2. Pick a small set of benign sequences as the SHAP background.
    3. Rebuild the malware sequences and pick the highest error ones to explain.
    4. Run SHAP for both models and rank the features.
    5. Save a side by side top 10 chart and the full importance tables.

KernelExplainer is slow, so the number of explained sequences and the
number of SHAP samples are kept small.

Run from the repository root:
    python src/explainability_shap.py
"""

import sys
from pathlib import Path

# Make the parent folder importable before the local imports below
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from data_loading import RAW_DATA_DIR
from sample_large_scenario import sample_large_conn_log
from preprocessing import clean, transform, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sequence_generation import generate_sequences
from models.lstm_auto_encoder import LSTMAutoencoder, reconstruction_error as lstm_recon_error
from models.transformer_auto_encoder import TransformerAutoencoder, reconstruction_error as transformer_recon_error

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10
N_BACKGROUND = 20  # benign sequences used as the SHAP reference distribution
N_EXPLAIN = 8      # malware sequences to explain (kept small because KernelExplainer is slow)


def get_feature_names(fitted_transformer) -> list[str]:
    """
    Get readable names for each processed feature column.

    Args:
        fitted_transformer: The fitted preprocessing pipeline.

    Returns:
        List of feature names, in column order.
    """
    return list(fitted_transformer.get_feature_names_out())


def make_error_fn(model, model_type: str, window_size: int, n_features: int):
    """
    Wrap a trained model into a function that SHAP can call.

    The returned function takes a 2D array of flattened sequences with shape
    (n_samples, window_size * n_features) and returns a 1D array of
    reconstruction errors with shape (n_samples,).

    Args:
        model: Trained autoencoder.
        model_type: "lstm" or "transformer".
        window_size: Number of flows in each sequence.
        n_features: Number of features per flow.

    Returns:
        The error function.
    """
    def error_fn(flat_sequences: np.ndarray) -> np.ndarray:
        batch = flat_sequences.reshape(-1, window_size, n_features)
        if model_type == "lstm":
            return lstm_recon_error(model, batch)
        else:
            return transformer_recon_error(model, batch)
    return error_fn


def run_shap_for_model(
    model, model_type: str, background_seqs: np.ndarray, explain_seqs: np.ndarray,
    feature_names: list[str], window_size: int, n_features: int,
):
    """
    Run SHAP for one model and rank the features.

    Args:
        model: Trained autoencoder.
        model_type: "lstm" or "transformer".
        background_seqs: Benign sequences used as the reference distribution.
        explain_seqs: Malware sequences to explain.
        feature_names: Names of the processed features.
        window_size: Number of flows in each sequence.
        n_features: Number of features per flow.

    Returns:
        A tuple of (importance_df, shap_values). importance_df ranks the
        features by their summed absolute SHAP value. shap_values has shape
        (n_explain, window_size, n_features).
    """
    # SHAP works on flat rows, so flatten each sequence
    background_flat = background_seqs.reshape(len(background_seqs), -1)
    explain_flat = explain_seqs.reshape(len(explain_seqs), -1)

    error_fn = make_error_fn(model, model_type, window_size, n_features)

    print(f"  Building KernelExplainer for {model_type} "
          f"(background n = {len(background_flat)}, explaining n = {len(explain_flat)})...")
    explainer = shap.KernelExplainer(error_fn, background_flat)
    shap_values = explainer.shap_values(explain_flat, nsamples=100)  # nsamples kept low for speed

    # Reshape to (n_explain, window, features), then add up over samples and time steps
    shap_values = np.array(shap_values).reshape(len(explain_seqs), window_size, n_features)
    feature_importance = np.abs(shap_values).sum(axis=(0, 1))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": feature_importance,
    }).sort_values("mean_abs_shap", ascending=False)

    return importance_df, shap_values


if __name__ == "__main__":
    print("=" * 50)
    print("SHAP feature importance")
    print("=" * 50)

    # Load the fitted preprocessing pipeline
    fitted_transformer = joblib.load(PROCESSED_DIR / "fitted_transformer.joblib")
    feature_names = get_feature_names(fitted_transformer)
    n_features = len(feature_names)
    print(f"Loaded preprocessing pipeline with {n_features} features")

    # Background: a random sample of benign sequences
    benign_data = np.load(PROCESSED_DIR / "benign_processed.npy")
    benign_sequences = generate_sequences(benign_data, window_size=WINDOW_SIZE, overlap_ratio=0.5)
    rng = np.random.default_rng(42)
    background_idx = rng.choice(len(benign_sequences), size=N_BACKGROUND, replace=False)
    background_seqs = benign_sequences[background_idx]
    print(f"Selected {N_BACKGROUND} benign sequences as the SHAP background")

    # Explanation set: reload the malware data and rebuild the sequences
    print("\nReloading malware sample for the explanation set...")
    from pre_failure_score import load_malware_sample, clean_keep_label, MALWARE_SCENARIOS
    malware_df = load_malware_sample(MALWARE_SCENARIOS)
    malware_df_clean = clean_keep_label(malware_df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    malware_processed = transform(malware_df_clean[feature_cols], fitted_transformer)
    malware_sequences = generate_sequences(malware_processed, window_size=WINDOW_SIZE, overlap_ratio=0.0)

    # Load the final models
    lstm_model = LSTMAutoencoder(n_features=n_features, hidden_dim=32, num_layers=1, dropout=0.2)
    lstm_model.load_state_dict(torch.load(RESULTS_DIR / "lstm_ae_final.pt"))

    transformer_model = TransformerAutoencoder(
        n_features=n_features, d_model=32, n_heads=4,
        num_layers=1, dim_feedforward=64, dropout=0.2,
    )
    transformer_model.load_state_dict(torch.load(RESULTS_DIR / "transformer_ae_final.pt"))
    print("Loaded the final LSTM and Transformer models")

    # Pick the highest error (most anomalous) sequences to explain
    lstm_errors = lstm_recon_error(lstm_model, malware_sequences)
    top_idx = np.argsort(lstm_errors)[-N_EXPLAIN:]
    explain_seqs = malware_sequences[top_idx]
    print(f"Selected {len(explain_seqs)} highest error malware sequences for SHAP explanation")

    # Run SHAP for each model
    print("\n--- Running SHAP for the LSTM Autoencoder ---")
    lstm_importance, lstm_shap_values = run_shap_for_model(
        lstm_model, "lstm", background_seqs, explain_seqs, feature_names, WINDOW_SIZE, n_features
    )
    print("\nTop 10 features (LSTM):")
    print(lstm_importance.head(10).to_string(index=False))

    print("\n--- Running SHAP for the Transformer Autoencoder ---")
    transformer_importance, transformer_shap_values = run_shap_for_model(
        transformer_model, "transformer", background_seqs, explain_seqs, feature_names, WINDOW_SIZE, n_features
    )
    print("\nTop 10 features (Transformer):")
    print(transformer_importance.head(10).to_string(index=False))

    # Plot the top 10 features for both models side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in [
        (axes[0], lstm_importance.head(10), "LSTM Autoencoder"),
        (axes[1], transformer_importance.head(10), "Transformer Autoencoder"),
    ]:
        ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title)
    plt.suptitle("Top 10 Feature Contributions to Reconstruction Error (SHAP)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_feature_importance.png", dpi=200)
    print(f"\nSaved SHAP feature importance plot to {FIGURES_DIR / 'shap_feature_importance.png'}")

    # Save the full importance tables
    lstm_importance.to_csv(RESULTS_DIR / "shap_lstm_feature_importance.csv", index=False)
    transformer_importance.to_csv(RESULTS_DIR / "shap_transformer_feature_importance.csv", index=False)
    print("Saved feature importance tables to outputs/results/")