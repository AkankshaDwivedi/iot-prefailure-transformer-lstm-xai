"""
Pre-Failure Score pipeline.

This script scores the malware sequences with both final models and turns
each reconstruction error into one of four Pre-Failure Score levels.

Steps:
    1. Load a sample of the malware scenarios (up to 20,000 rows each).
    2. Preprocess with the already fitted transformer (transform only,
       never refit on malware data).
    3. Build non overlapping sequences and give each one a ground truth
       label: Malicious if any flow in the window is malicious, else Benign.
    4. Score the sequences with the final LSTM and Transformer models.
    5. Collect pooled out of fold benign errors from 5 fold training and use
       their 90th, 97th and 99.5th percentiles as the level thresholds.
    6. Assign the levels, compute the metrics and save the results to
       outputs/results/malware_prefailure_scores_v2.csv.

The final thresholds from the final models' benign errors are applied later
by recalibrate_final.py and finalize_metrics.py.

Other scripts import load_malware_sample(), clean_keep_label() and
MALWARE_SCENARIOS from this file.

Run from the repository root:
    python src/pre_failure_score.py
"""

import sys
from pathlib import Path

# Make the parent folder importable before the local imports below
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch

from data_loading import RAW_DATA_DIR, list_scenarios
from sample_large_scenario import sample_large_conn_log
from preprocessing import (
    clean, transform, DROP_COLUMNS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from sequence_generation import generate_sequences, kfold_split
from models.lstm_auto_encoder import LSTMAutoencoder, reconstruction_error as lstm_recon_error
from models.transformer_auto_encoder import TransformerAutoencoder, reconstruction_error as transformer_recon_error

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("outputs/results")

# Malware scenarios used for testing
MALWARE_SCENARIOS = [
    "CTU-IoT-Malware-Capture-1-1",
    "CTU-IoT-Malware-Capture-3-1",
    "CTU-IoT-Malware-Capture-20-1",
    "CTU-IoT-Malware-Capture-21-1",
    "CTU-IoT-Malware-Capture-34-1",
    "CTU-IoT-Malware-Capture-8-1",
]
ROWS_PER_LARGE_SCENARIO = 20_000
WINDOW_SIZE = 10


def load_malware_sample(scenario_names: list[str]) -> pd.DataFrame:
    """
    Load a sample of rows from each malware scenario.

    A scenario that cannot be loaded is skipped with a message.

    Args:
        scenario_names: Names of the scenario folders to load.

    Returns:
        One DataFrame with all scenarios combined, sorted by timestamp,
        with an extra "scenario" column.
    """
    frames = []
    for name in scenario_names:
        folder = RAW_DATA_DIR / name
        try:
            df = sample_large_conn_log(folder, n_rows=ROWS_PER_LARGE_SCENARIO)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue
        df["scenario"] = name
        frames.append(df)
        print(f"  Loaded {name}: {len(df)} rows")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("ts").reset_index(drop=True)


def clean_keep_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data like preprocessing.clean(), but keep the label column.

    The label is the ground truth. It must survive until sequence
    generation so that the results can be evaluated.

    Args:
        df: Raw malware DataFrame.

    Returns:
        Cleaned DataFrame without duplicates or rows missing ts or label.
    """
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns and c != "label"]
    df = df.drop(columns=drop_cols)
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["ts", "label"])
    print(f"Cleaning: {before} -> {len(df)} rows "
          f"({before - len(df)} removed as duplicates or invalid)")
    return df.reset_index(drop=True)


def sequence_ground_truth(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Give each window one ground truth label.

    A window is labelled "Malicious" if any flow inside it is malicious,
    otherwise "Benign". This is a sensitive choice: a single malicious flow
    in an otherwise benign window still marks the whole window as an event
    worth flagging, which fits an early warning use case.

    Args:
        labels: Flow level labels.
        window_size: Number of flows in each window.
        stride: Step between the start of one window and the next.

    Returns:
        Array with one label per window.
    """
    n_rows = len(labels)
    seq_labels = []
    for start in range(0, n_rows - window_size + 1, stride):
        window_labels = labels[start:start + window_size]
        is_malicious = any(str(l).lower() == "malicious" for l in window_labels)
        seq_labels.append("Malicious" if is_malicious else "Benign")
    return np.array(seq_labels)


def get_pooled_oof_benign_errors(model_class, model_kwargs: dict, benign_data: np.ndarray) -> np.ndarray:
    """
    Collect pooled out of fold reconstruction errors on benign sequences.

    The benign sequences are split into 5 folds. For each fold a new model
    is trained on the other folds, and the errors on the held out fold are
    collected. Pooling all folds gives about 195 benign errors, which is
    more reliable for percentile thresholds than a small holdout set.

    Args:
        model_class: LSTMAutoencoder or TransformerAutoencoder.
        model_kwargs: Arguments used to build the model.
        benign_data: Processed benign data, shape (n_rows, n_features).

    Returns:
        Array of pooled out of fold errors.
    """
    from models.lstm_auto_encoder import train_one_fold as train_lstm
    from models.transformer_auto_encoder import train_one_fold as train_transformer
    from sequence_generation import jitter_augment

    sequences = generate_sequences(benign_data, window_size=WINDOW_SIZE, overlap_ratio=0.5)
    folds = kfold_split(sequences, n_splits=5)

    pooled_errors = []
    train_fn = train_lstm if model_class is LSTMAutoencoder else train_transformer
    recon_fn = lstm_recon_error if model_class is LSTMAutoencoder else transformer_recon_error
    epochs = 150 if model_class is LSTMAutoencoder else 400

    for i, (train_idx, val_idx) in enumerate(folds):
        train_seqs = jitter_augment(sequences[train_idx], noise_std=0.01, n_augmented_copies=4)
        val_seqs = sequences[val_idx]
        model = model_class(**model_kwargs)
        result = train_fn(model, train_seqs, val_seqs, epochs=epochs, patience=15)
        fold_errors = recon_fn(result["model"], val_seqs)
        pooled_errors.extend(fold_errors.tolist())
        print(f"  Fold {i + 1}/5 done: {len(fold_errors)} out-of-fold errors collected")

    return np.array(pooled_errors)


def calibrate_thresholds(benign_errors: np.ndarray) -> dict:
    """
    Compute the level thresholds from the benign reconstruction errors.

    Args:
        benign_errors: Reconstruction errors of benign sequences.

    Returns:
        Dictionary with the 90th, 97th and 99.5th percentile values.
    """
    return {
        "normal_max": np.percentile(benign_errors, 90),
        "low_drift_max": np.percentile(benign_errors, 97),
        "pre_failure_max": np.percentile(benign_errors, 99.5),
    }


def assign_level(error: float, thresholds: dict) -> str:
    """
    Convert one reconstruction error into a Pre-Failure Score level.

    Args:
        error: Reconstruction error of one sequence.
        thresholds: Dictionary from calibrate_thresholds().

    Returns:
        "Normal", "Low drift", "Pre-failure" or "Critical".
    """
    if error <= thresholds["normal_max"]:
        return "Normal"
    elif error <= thresholds["low_drift_max"]:
        return "Low drift"
    elif error <= thresholds["pre_failure_max"]:
        return "Pre-failure"
    else:
        return "Critical"


def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict:
    """
    Compute the confusion counts and the main detection metrics.

    The ground truth ("Malicious" or "Benign") is compared with the
    Pre-Failure Score collapsed to a binary prediction: Normal and Low drift
    count as normal, Pre-failure and Critical count as anomalous.

    Args:
        y_true: Ground truth labels.
        y_pred_binary: Predictions, "Anomalous" or "Normal".

    Returns:
        Dictionary with TP, FP, TN, FN, Precision, Recall, F1 and FPR.
    """
    tp = np.sum((y_true == "Malicious") & (y_pred_binary == "Anomalous"))
    fp = np.sum((y_true == "Benign") & (y_pred_binary == "Anomalous"))
    tn = np.sum((y_true == "Benign") & (y_pred_binary == "Normal"))
    fn = np.sum((y_true == "Malicious") & (y_pred_binary == "Normal"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1, "FPR": fpr}


if __name__ == "__main__":
    print("=" * 50)
    print("Pre-Failure Score pipeline")
    print("=" * 50)

    # Load and clean the malware sample
    print("Loading malware scenario sample...")
    malware_df = load_malware_sample(MALWARE_SCENARIOS)
    print(f"\nTotal malware rows loaded: {len(malware_df)}")
    print(f"Label distribution:\n{malware_df['label'].value_counts()}")

    malware_df_clean = clean_keep_label(malware_df)
    fitted_transformer = joblib.load(PROCESSED_DIR / "fitted_transformer.joblib")

    # Transform with the fitted pipeline (never refit on malware data)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    malware_processed = transform(malware_df_clean[feature_cols], fitted_transformer)
    print(f"\nProcessed malware feature matrix: {malware_processed.shape}")

    # Build non overlapping sequences and their ground truth labels
    malware_sequences = generate_sequences(malware_processed, window_size=WINDOW_SIZE, overlap_ratio=0.0)
    seq_ground_truth = sequence_ground_truth(
        malware_df_clean["label"].values, window_size=WINDOW_SIZE, stride=WINDOW_SIZE
    )
    print(f"Generated {len(malware_sequences)} sequences, "
          f"{len(seq_ground_truth)} ground truth labels")
    print(f"Ground truth distribution: "
          f"{pd.Series(seq_ground_truth).value_counts().to_dict()}")

    # Load the final models
    n_features = malware_processed.shape[1]
    lstm_model = LSTMAutoencoder(n_features=n_features, hidden_dim=32, num_layers=1, dropout=0.2)
    lstm_model.load_state_dict(torch.load(RESULTS_DIR / "lstm_ae_final.pt"))

    transformer_model = TransformerAutoencoder(
        n_features=n_features, d_model=32, n_heads=4,
        num_layers=1, dim_feedforward=64, dropout=0.2,
    )
    transformer_model.load_state_dict(torch.load(RESULTS_DIR / "transformer_ae_final.pt"))
    print("Loaded the final LSTM and Transformer models")

    # Reconstruction errors of the malware sequences
    lstm_malware_errors = lstm_recon_error(lstm_model, malware_sequences)
    transformer_malware_errors = transformer_recon_error(transformer_model, malware_sequences)

    # Calibrate thresholds from pooled out of fold benign errors (about 195 samples)
    benign_data = np.load(PROCESSED_DIR / "benign_processed.npy")
    print("\nCalibrating LSTM thresholds via pooled out-of-fold errors...")
    lstm_oof_errors = get_pooled_oof_benign_errors(
        LSTMAutoencoder, {"n_features": n_features, "hidden_dim": 32, "num_layers": 1, "dropout": 0.2},
        benign_data,
    )
    print(f"Pooled LSTM out-of-fold benign errors: n = {len(lstm_oof_errors)}")

    print("\nCalibrating Transformer thresholds via pooled out-of-fold errors...")
    transformer_oof_errors = get_pooled_oof_benign_errors(
        TransformerAutoencoder,
        {"n_features": n_features, "d_model": 32, "n_heads": 4,
         "num_layers": 1, "dim_feedforward": 64, "dropout": 0.2},
        benign_data,
    )
    print(f"Pooled Transformer out-of-fold benign errors: n = {len(transformer_oof_errors)}")

    np.save(RESULTS_DIR / "lstm_oof_benign_errors.npy", lstm_oof_errors)
    np.save(RESULTS_DIR / "transformer_oof_benign_errors.npy", transformer_oof_errors)

    lstm_thresholds = calibrate_thresholds(lstm_oof_errors)
    transformer_thresholds = calibrate_thresholds(transformer_oof_errors)

    print(f"\nLSTM thresholds: {lstm_thresholds}")
    print(f"Transformer thresholds: {transformer_thresholds}")

    # Assign Pre-Failure Score levels, then collapse to a binary prediction
    lstm_levels = [assign_level(e, lstm_thresholds) for e in lstm_malware_errors]
    transformer_levels = [assign_level(e, transformer_thresholds) for e in transformer_malware_errors]

    lstm_binary = np.array(["Normal" if l in ("Normal", "Low drift") else "Anomalous" for l in lstm_levels])
    transformer_binary = np.array(["Normal" if l in ("Normal", "Low drift") else "Anomalous" for l in transformer_levels])

    # Compute the metrics against the ground truth
    lstm_metrics = compute_metrics(seq_ground_truth, lstm_binary)
    transformer_metrics = compute_metrics(seq_ground_truth, transformer_binary)

    print("\n" + "=" * 50)
    print("LSTM evaluation against ground truth:")
    for k, v in lstm_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nTransformer evaluation against ground truth:")
    for k, v in transformer_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save the scores and levels for the next steps
    results = pd.DataFrame({
        "ground_truth": seq_ground_truth,
        "lstm_error": lstm_malware_errors,
        "lstm_level": lstm_levels,
        "transformer_error": transformer_malware_errors,
        "transformer_level": transformer_levels,
    })
    results.to_csv(RESULTS_DIR / "malware_prefailure_scores_v2.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'malware_prefailure_scores_v2.csv'}")