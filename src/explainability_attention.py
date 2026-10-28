"""
Attention visualisation for the Transformer Autoencoder.

This script extracts the self attention weights from the Transformer's
encoder layer. They show which time steps inside a flagged sequence the
model relied on most. This is the time based counterpart to the feature
based explanation from SHAP.

Steps:
    1. Load the fitted preprocessing pipeline and the final Transformer.
    2. Rebuild the malware sequences and compute their reconstruction errors.
    3. For the highest error sequences, save an attention heatmap and a
       bar chart of attention received per time step.
    4. Count which time step is most attended across a larger sample and
       save the aggregate chart.

Run from the repository root:
    python src/explainability_attention.py
"""

import sys
from pathlib import Path

# Make the parent folder importable before the local imports below
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loading import RAW_DATA_DIR
from preprocessing import clean, transform, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sequence_generation import generate_sequences
from models.transformer_auto_encoder import TransformerAutoencoder, reconstruction_error as transformer_recon_error

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10
N_EXAMPLES = 4    # number of flagged sequences to save individual figures for
N_AGGREGATE = 50  # larger sample for the aggregate most attended position pattern


class AttentionCapture:
    """
    Placeholder for a forward hook.

    This class is not used. The weights are captured in get_attention_weights()
    by calling the attention module directly, which is simpler and reliable.
    """

    def __init__(self):
        self.weights = None

    def hook(self, module, input, output):
        pass


def get_attention_weights(model: TransformerAutoencoder, x: torch.Tensor) -> np.ndarray:
    """
    Run a forward pass and return the encoder's self attention weights.

    PyTorch's TransformerEncoderLayer does not expose the weights through
    its normal forward path, so the underlying MultiheadAttention module
    is called directly with need_weights=True.

    Args:
        model: Trained Transformer Autoencoder.
        x: Input tensor of shape (batch, seq_len, n_features).

    Returns:
        Array of shape (batch, seq_len, seq_len), averaged over the heads.
    """
    model.eval()
    with torch.no_grad():
        h = model.input_proj(x)
        h = model.pos_encoding(h)

        # Use the first encoder layer (the only one, since num_layers = 1)
        encoder_layer = model.encoder.layers[0]
        attn_output, attn_weights = encoder_layer.self_attn(
            h, h, h, need_weights=True, average_attn_weights=True
        )
    return attn_weights.cpu().numpy()


def plot_attention_heatmap(attn_matrix: np.ndarray, title: str, save_path: Path):
    """
    Save a heatmap of one sequence's attention matrix.

    Args:
        attn_matrix: Array of shape (seq_len, seq_len). Rows are the query
            position and columns are the key position.
        title: Figure title.
        save_path: Where to save the PNG file.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Time step (attended to)")
    plt.ylabel("Time step (attending from)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_attention_received(attn_matrix: np.ndarray, title: str, save_path: Path):
    """
    Save a bar chart of the attention each time step receives.

    The attention is summed over all query positions, which shows which
    positions the model treated as most important overall.

    Args:
        attn_matrix: Array of shape (seq_len, seq_len).
        title: Figure title.
        save_path: Where to save the PNG file.
    """
    received = attn_matrix.sum(axis=0)  # sum over the query dimension
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(received)), received)
    plt.xlabel("Time step within sequence window")
    plt.ylabel("Total attention received")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    print("=" * 50)
    print("Attention visualisation")
    print("=" * 50)

    # Load the fitted preprocessing pipeline and the final Transformer
    fitted_transformer = joblib.load(PROCESSED_DIR / "fitted_transformer.joblib")
    n_features = len(fitted_transformer.get_feature_names_out())

    transformer_model = TransformerAutoencoder(
        n_features=n_features, d_model=32, n_heads=4,
        num_layers=1, dim_feedforward=64, dropout=0.2,
    )
    transformer_model.load_state_dict(torch.load(RESULTS_DIR / "transformer_ae_final.pt"))
    print("Loaded the final Transformer model")

    # Rebuild the malware sequences the same way as the scoring script
    print("Reloading malware sample...")
    from pre_failure_score import load_malware_sample, clean_keep_label, MALWARE_SCENARIOS
    malware_df = load_malware_sample(MALWARE_SCENARIOS)
    malware_df_clean = clean_keep_label(malware_df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    malware_processed = transform(malware_df_clean[feature_cols], fitted_transformer)
    malware_sequences = generate_sequences(malware_processed, window_size=WINDOW_SIZE, overlap_ratio=0.0)
    print(f"Built {len(malware_sequences)} malware sequences")

    # Reconstruction error for every sequence, then pick the highest ones
    errors = transformer_recon_error(transformer_model, malware_sequences)
    top_idx = np.argsort(errors)[-N_EXAMPLES:]

    # Individual figures for the highest error sequences
    print(f"\nVisualising attention for the {N_EXAMPLES} highest error sequences...")
    for rank, idx in enumerate(top_idx):
        seq = malware_sequences[idx:idx + 1]  # keep the batch dimension
        x = torch.tensor(seq, dtype=torch.float32)
        attn = get_attention_weights(transformer_model, x)[0]  # drop batch dimension

        plot_attention_heatmap(
            attn, f"Attention heatmap -- sequence {idx} (error={errors[idx]:.4f})",
            FIGURES_DIR / f"attention_heatmap_seq{idx}.png",
        )
        plot_attention_received(
            attn, f"Attention received per time step -- sequence {idx} (error={errors[idx]:.4f})",
            FIGURES_DIR / f"attention_received_seq{idx}.png",
        )
        print(f"  Sequence {idx} (rank {rank + 1}, error = {errors[idx]:.4f}): "
              f"most attended time step = {attn.sum(axis=0).argmax()}")

    print(f"\nSaved {N_EXAMPLES} attention heatmaps and bar charts to {FIGURES_DIR}/")

    # Aggregate pattern across a larger sample
    print(f"\nCounting the most attended time step across the top {N_AGGREGATE} "
          f"highest error sequences...")
    top_agg_idx = np.argsort(errors)[-N_AGGREGATE:]
    most_attended_positions = []
    for idx in top_agg_idx:
        seq = malware_sequences[idx:idx + 1]
        x = torch.tensor(seq, dtype=torch.float32)
        attn = get_attention_weights(transformer_model, x)[0]
        most_attended_positions.append(int(attn.sum(axis=0).argmax()))

    positions, counts = np.unique(most_attended_positions, return_counts=True)
    print(f"\nMost attended time step across the top {N_AGGREGATE} sequences:")
    for p, c in zip(positions, counts):
        print(f"  Time step {p}: {c} sequences ({100 * c / N_AGGREGATE:.1f}%)")

    # Aggregate bar chart
    plt.figure(figsize=(7, 4))
    plt.bar(positions, counts)
    plt.xlabel("Time step within sequence window")
    plt.ylabel(f"Number of sequences (out of {N_AGGREGATE}) where this was most-attended")
    plt.title(f"Aggregate most-attended time step\n(top {N_AGGREGATE} highest-error malware sequences)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attention_aggregate_pattern.png", dpi=200)
    print(f"\nSaved aggregate pattern chart to {FIGURES_DIR / 'attention_aggregate_pattern.png'}")