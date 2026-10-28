"""
Threshold recalibration -- CORRECTED.

Fixes a model-mismatch bug: the previous version calibrated thresholds
using errors from 5 freshly-retrained fold models, then applied those
thresholds to a DIFFERENT final model's malware scores. This inflated
thresholds (each fold model has its own error scale) and caused a
sharp, spurious recall drop.

Correct approach: score ALL benign sequences with the SAME final model
used for malware scoring, then calibrate from that single, consistent
error distribution.

Trade-off, documented explicitly: this includes sequences the final
model was trained on (not a strictly held-out set), which is standard
practice for setting an autoencoder's own operating threshold, but is
worth stating plainly as a limitation given the small dataset (Section 6.2).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from sequence_generation import generate_sequences
from models.lstm_auto_encoder import LSTMAutoencoder, reconstruction_error as lstm_recon_error
from models.transformer_auto_encoder import TransformerAutoencoder, reconstruction_error as transformer_recon_error

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("outputs/results")
WINDOW_SIZE = 10


def calibrate_thresholds(benign_errors: np.ndarray) -> dict:
    return {
        "normal_max": np.percentile(benign_errors, 90),
        "low_drift_max": np.percentile(benign_errors, 97),
        "pre_failure_max": np.percentile(benign_errors, 99.5),
    }


if __name__ == "__main__":
    benign_data = np.load(PROCESSED_DIR / "benign_processed.npy")
    n_features = benign_data.shape[1]
    sequences = generate_sequences(benign_data, window_size=WINDOW_SIZE, overlap_ratio=0.5)
    print(f"Scoring all {len(sequences)} benign sequences with the FINAL models "
          f"(same models used for malware scoring)...")

    lstm_model = LSTMAutoencoder(n_features=n_features, hidden_dim=32, num_layers=1, dropout=0.2)
    lstm_model.load_state_dict(torch.load(RESULTS_DIR / "lstm_ae_final.pt"))

    transformer_model = TransformerAutoencoder(
        n_features=n_features, d_model=32, n_heads=4,
        num_layers=1, dim_feedforward=64, dropout=0.2,
    )
    transformer_model.load_state_dict(torch.load(RESULTS_DIR / "transformer_ae_final.pt"))

    lstm_errors = lstm_recon_error(lstm_model, sequences)
    transformer_errors = transformer_recon_error(transformer_model, sequences)

    lstm_thresholds = calibrate_thresholds(lstm_errors)
    transformer_thresholds = calibrate_thresholds(transformer_errors)

    print(f"\nLSTM benign error stats: mean={lstm_errors.mean():.5f}, "
          f"std={lstm_errors.std():.5f}, n={len(lstm_errors)}")
    print(f"LSTM thresholds: {lstm_thresholds}")

    print(f"\nTransformer benign error stats: mean={transformer_errors.mean():.5f}, "
          f"std={transformer_errors.std():.5f}, n={len(transformer_errors)}")
    print(f"Transformer thresholds: {transformer_thresholds}")

    np.save(RESULTS_DIR / "lstm_final_benign_errors.npy", lstm_errors)
    np.save(RESULTS_DIR / "transformer_final_benign_errors.npy", transformer_errors)
    print(f"\nSaved corrected calibration error arrays to {RESULTS_DIR}")
    print("Next: rerun pre_failure_score.py's threshold/metric section using "
          "these corrected thresholds (see updated calibrate_thresholds call).")