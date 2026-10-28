"""
Train the final models.

This script trains the final LSTM Autoencoder and Transformer Autoencoder
on the benign sequences, using a 70/30 split:
    70%  training set (augmented with noisy copies)
    30%  monitoring set (never augmented), used for early stopping and
         for the benign reconstruction errors

Steps:
    1. Load the processed benign data and build sequences (window 10, 50% overlap).
    2. Split 70/30 and augment the training part only.
    3. Train the LSTM and save its weights.
    4. Train the Transformer and save its weights.
    5. Save the benign reconstruction errors of the monitoring set.

Run from the repository root:
    python src/train_final.py
"""

import sys
from pathlib import Path

# Make the parent folder importable before the local imports below
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sequence_generation import generate_sequences, jitter_augment
from models.lstm_auto_encoder import LSTMAutoencoder, train_one_fold as train_lstm, reconstruction_error as lstm_recon_error
from models.transformer_auto_encoder import TransformerAutoencoder, train_one_fold as train_transformer, reconstruction_error as transformer_recon_error

# Make sure the output folder exists
Path("outputs/results").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("=" * 50)
    print("Final model training")
    print("=" * 50)

    # Load the benign data and build sequences
    data = np.load("data/processed/benign_processed.npy")
    n_features = data.shape[1]
    sequences = generate_sequences(data, window_size=10, overlap_ratio=0.5)
    print(f"Total benign sequences: {len(sequences)}")

    # 70/30 split, then augment the training part only
    train_seqs, monitor_seqs = train_test_split(sequences, test_size=0.3, random_state=42)
    train_seqs_aug = jitter_augment(train_seqs, noise_std=0.01, n_augmented_copies=4)
    print(f"Final training set: {len(train_seqs_aug)} (augmented), "
          f"monitoring set: {len(monitor_seqs)}")

    # Train and save the final LSTM
    print("\n--- Training final LSTM Autoencoder ---")
    lstm_model = LSTMAutoencoder(n_features=n_features, hidden_dim=32, num_layers=1, dropout=0.2)
    lstm_result = train_lstm(lstm_model, train_seqs_aug, monitor_seqs, epochs=150, patience=15)
    torch.save(lstm_result["model"].state_dict(), "outputs/results/lstm_ae_final.pt")
    print(f"Final LSTM monitoring loss: {lstm_result['best_val_loss']:.5f}")
    print("Saved: outputs/results/lstm_ae_final.pt")

    # Train and save the final Transformer
    print("\n--- Training final Transformer Autoencoder ---")
    transformer_model = TransformerAutoencoder(
        n_features=n_features, d_model=32, n_heads=4,
        num_layers=1, dim_feedforward=64, dropout=0.2,
    )
    transformer_result = train_transformer(
        transformer_model, train_seqs_aug, monitor_seqs, epochs=400, patience=15
    )
    torch.save(transformer_result["model"].state_dict(), "outputs/results/transformer_ae_final.pt")
    print(f"Final Transformer monitoring loss: {transformer_result['best_val_loss']:.5f}")
    print("Saved: outputs/results/transformer_ae_final.pt")

    # Save the monitoring set reconstruction errors. These are the basis
    # for the benign percentile thresholds.
    lstm_benign_errors = lstm_recon_error(lstm_result["model"], monitor_seqs)
    transformer_benign_errors = transformer_recon_error(transformer_result["model"], monitor_seqs)

    np.save("outputs/results/lstm_benign_errors.npy", lstm_benign_errors)
    np.save("outputs/results/transformer_benign_errors.npy", transformer_benign_errors)
    print(f"\nSaved benign reconstruction error distributions "
          f"({len(monitor_seqs)} samples each) for threshold calibration.")