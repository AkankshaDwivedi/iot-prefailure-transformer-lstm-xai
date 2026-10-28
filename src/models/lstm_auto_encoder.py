"""
LSTM Autoencoder baseline.

This script trains an LSTM Autoencoder on benign IoT-23 sequences only.
The model learns to reconstruct normal behaviour, and the reconstruction
error is later used as the anomaly signal.

Steps when run as a script:
    1. Load the processed benign data.
    2. Build overlapping sequences (window of 10 flows, 50% overlap).
    3. Split the sequences into 5 folds.
    4. For each fold, augment the training part with noisy copies and
       train the model with early stopping.
    5. Report the mean validation loss and save the last fold's model
       as a quick checkpoint.

Run from the repository root:
    python src/models/lstm_auto_encoder.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Fixed seed so that results can be repeated
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Make the parent folder importable and make sure the output folder exists
sys.path.append(str(Path(__file__).resolve().parent.parent))
Path("outputs/results").mkdir(parents=True, exist_ok=True)

from sequence_generation import generate_sequences, jitter_augment, kfold_split


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder.

    The encoder reads a sequence and keeps the final hidden state as a
    compact summary (the latent vector). The decoder receives this latent
    vector at every time step and tries to rebuild the original sequence.

    Args:
        n_features: Number of input features per time step.
        hidden_dim: Size of the LSTM hidden state (latent size).
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between encoder layers (used only if num_layers > 1).
    """

    def __init__(self, n_features: int, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of sequences.

        Args:
            x: Tensor of shape (batch, seq_len, n_features).

        Returns:
            Reconstruction with the same shape as x.
        """
        seq_len = x.size(1)

        # Encode: take the hidden state of the last layer as the latent vector
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]  # shape (batch, hidden_dim)

        # Decode: repeat the latent vector for every time step
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)

        # Map back to the original feature size
        reconstruction = self.output_layer(decoded)
        return reconstruction


def train_one_fold(
    model: nn.Module,
    train_seqs: np.ndarray,
    val_seqs: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Train the model on one fold with early stopping on validation loss.

    Args:
        model: The autoencoder to train.
        train_seqs: Training sequences, shape (n, seq_len, n_features).
        val_seqs: Validation sequences (never augmented).
        epochs: Maximum number of epochs.
        batch_size: Mini batch size.
        lr: Learning rate for the Adam optimiser.
        patience: Epochs to wait for an improvement before stopping.
        device: "cpu" or "cuda".

    Returns:
        Dictionary with the model (best weights loaded), the loss history
        and the best validation loss.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(train_seqs, dtype=torch.float32)
    val_tensor = torch.tensor(val_seqs, dtype=torch.float32)

    # Seeded generator so the batch order is repeatable
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size,
                              shuffle=True, generator=g)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training pass
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_tensor)

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_reconstruction = model(val_tensor.to(device))
            val_loss = criterion(val_reconstruction, val_tensor.to(device)).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        # Keep the best weights and apply early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(best val_loss = {best_val_loss:.5f})")
                break

        # Progress message every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d} | train_loss = {epoch_loss:.5f} "
                  f"| val_loss = {val_loss:.5f}")

    # Restore the best weights before returning
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_loss": best_val_loss}


def reconstruction_error(model: nn.Module, sequences: np.ndarray,
                         device: str = "cpu") -> np.ndarray:
    """
    Compute the mean squared reconstruction error for each sequence.

    This value is the core anomaly signal: normal sequences are rebuilt
    well (low error) and unusual sequences are rebuilt badly (high error).

    Args:
        model: Trained autoencoder.
        sequences: Array of shape (n, seq_len, n_features).
        device: "cpu" or "cuda".

    Returns:
        Array of shape (n,) with one error value per sequence.
    """
    model.eval()
    tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstruction = model(tensor)
    errors = torch.mean((reconstruction - tensor) ** 2, dim=(1, 2))
    return errors.cpu().numpy()


if __name__ == "__main__":
    print("=" * 50)
    print("LSTM Autoencoder: 5 fold cross validation")
    print("=" * 50)

    # Load the processed benign data
    data = np.load("data/processed/benign_processed.npy")
    n_features = data.shape[1]
    print(f"Loaded benign data: {data.shape[0]} rows, {n_features} features")

    # Build overlapping sequences
    sequences = generate_sequences(data, window_size=10, overlap_ratio=0.5)
    print(f"Generated {len(sequences)} sequences (window size 10)")

    # Split into folds
    folds = kfold_split(sequences, n_splits=5)

    fold_results = []
    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {i + 1} of {len(folds)} ---")

        # Augment training data only. Validation data stays untouched.
        train_seqs = jitter_augment(sequences[train_idx], noise_std=0.01,
                                    n_augmented_copies=4)
        val_seqs = sequences[val_idx]
        print(f"  Training sequences: {len(train_seqs)} | "
              f"Validation sequences: {len(val_seqs)}")

        model = LSTMAutoencoder(n_features=n_features, hidden_dim=32,
                                num_layers=1, dropout=0.2)
        result = train_one_fold(model, train_seqs, val_seqs, epochs=100, patience=10)
        fold_results.append(result["best_val_loss"])
        print(f"  Fold {i + 1} best val_loss: {result['best_val_loss']:.5f}")

    # Summary across folds
    print("\n" + "=" * 50)
    print(f"Mean val_loss over {len(folds)} folds: {np.mean(fold_results):.5f} "
          f"(std {np.std(fold_results):.5f})")
    print("=" * 50)

    # Save the last fold's model as a quick checkpoint
    torch.save(result["model"].state_dict(), "outputs/results/lstm_ae_checkpoint.pt")
    print("Saved checkpoint to outputs/results/lstm_ae_checkpoint.pt")