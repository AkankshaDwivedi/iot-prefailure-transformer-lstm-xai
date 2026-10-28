"""
Sequence generation for IoT-23 benign traffic.

The benign data is small (about 2,000 flows across the three benign
scenarios), so this module uses:
    1. Moderately overlapping sliding windows. The stride is not 1, which
       would create near duplicate sequences, and not fully independent,
       which would leave too few sequences for training.
    2. Light Gaussian jitter augmentation on the features, applied only to
       the training fold and never to validation or test data.
    3. K fold cross validation instead of a single fixed split, to make
       efficient use of the limited benign data.

Run from the repository root for a quick demo:
    python src/sequence_generation.py
"""

import numpy as np
from sklearn.model_selection import KFold


def generate_sequences(
    data: np.ndarray,
    window_size: int = 20,
    stride: int | None = None,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """
    Slide a fixed size window over the data.

    If stride is not given, it is worked out from overlap_ratio. For example,
    overlap_ratio = 0.5 gives stride = window_size // 2 (50% overlap), a
    middle ground between fully independent windows (stride = window_size)
    and heavily correlated windows (stride = 1).

    Args:
        data: Array of shape (n_rows, n_features).
        window_size: Number of flows in each sequence.
        stride: Step between the start of one window and the next.
        overlap_ratio: Share of overlap between neighbouring windows.

    Returns:
        Array of shape (n_sequences, window_size, n_features).
    """
    if stride is None:
        stride = max(1, int(window_size * (1 - overlap_ratio)))

    n_rows = data.shape[0]
    sequences = [
        data[start:start + window_size]
        for start in range(0, n_rows - window_size + 1, stride)
    ]
    return np.array(sequences) if sequences else np.empty((0, window_size, data.shape[1]))


def jitter_augment(
    sequences: np.ndarray,
    noise_std: float = 0.01,
    n_augmented_copies: int = 1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Add small Gaussian noise to the sequences to create augmented copies.

    Important: apply this to training data only, never to validation or
    test sequences. Augmenting evaluation data would distort the reported
    metrics.

    The features are already min max scaled to [0, 1] by the preprocessing
    pipeline, so noise_std = 0.01 means noise of about 1% of the feature's
    full range. Noisy values are clipped back into [0, 1].

    Args:
        sequences: Array of shape (n, window_size, n_features).
        noise_std: Standard deviation of the noise.
        n_augmented_copies: Number of noisy copies to add.
        random_state: Seed for repeatable noise.

    Returns:
        The original sequences followed by the noisy copies.
    """
    rng = np.random.default_rng(random_state)
    augmented = [sequences]  # keep the originals
    for _ in range(n_augmented_copies):
        noise = rng.normal(loc=0.0, scale=noise_std, size=sequences.shape)
        augmented.append(np.clip(sequences + noise, 0.0, 1.0))
    return np.concatenate(augmented, axis=0)


def kfold_split(sequences: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """
    Split the sequences for k fold cross validation.

    Used because the benign data is limited. The malware test data stays a
    separate, fully held out set regardless of this split.

    Args:
        sequences: Array of shape (n, window_size, n_features).
        n_splits: Number of folds.
        random_state: Seed for the shuffle.

    Returns:
        List of (train_idx, val_idx) index pairs, one per fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(sequences))


if __name__ == "__main__":
    print("=" * 50)
    print("Sequence generation demo")
    print("=" * 50)

    # Load the benign feature matrix saved by preprocessing.py
    data = np.load("data/processed/benign_processed.npy")
    print(f"Loaded processed benign data: {data.shape}")

    # A window of 10 gives more sequences from the same limited data
    window_size = 10
    sequences = generate_sequences(data, window_size=window_size, overlap_ratio=0.5)
    print(f"\nWindow = {window_size}, 50% overlap -> {len(sequences)} sequences")

    # K fold split
    folds = kfold_split(sequences, n_splits=5)
    print(f"5 fold split sizes (train, val): "
          f"{[(len(tr), len(val)) for tr, val in folds]}")

    # Show augmentation on one fold's training data only
    train_idx, val_idx = folds[0]
    train_seqs = sequences[train_idx]
    val_seqs = sequences[val_idx]  # left untouched, no augmentation

    # Four extra copies raise the effective training volume
    augmented_train = jitter_augment(train_seqs, noise_std=0.01, n_augmented_copies=4)
    print(f"\nFold 1: {len(train_seqs)} original training sequences "
          f"-> {len(augmented_train)} after augmentation (4 extra copies)")
    print(f"Fold 1: {len(val_seqs)} validation sequences (left unaugmented)")