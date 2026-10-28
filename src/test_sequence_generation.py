"""
Window and stride test for sequence generation.

This is an exploration helper that was used early in the project. It loads
the benign (Honeypot) scenarios, keeps five simple numeric features and
counts how many sequences each window size and stride combination gives.
The counts helped to choose the final setting (window 10, stride 5).

The final pipeline uses sequence_generation.py, not this file.

Run from the repository root:
    python src/test_sequence_generation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import RAW_DATA_DIR, list_scenarios, load_conn_log

# Scenario folders with this word in the name are benign
BENIGN_KEYWORD = "Honeypot"

NUMERIC_FEATURES = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]


def load_all_benign(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load and combine all benign (Honeypot Capture) scenarios.

    Args:
        raw_dir: Folder that contains the scenario folders.

    Returns:
        One DataFrame sorted by timestamp, with an extra "scenario" column.
    """
    scenarios = [s for s in list_scenarios(raw_dir) if BENIGN_KEYWORD in s]
    print(f"Loading {len(scenarios)} benign scenarios: {scenarios}")

    frames = []
    for name in scenarios:
        df = load_conn_log(raw_dir / name)
        df["scenario"] = name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("ts").reset_index(drop=True)
    print(f"Total combined benign rows: {len(combined)}")
    return combined


def prepare_numeric(df: pd.DataFrame) -> np.ndarray:
    """
    Minimal cleaning of the numeric features.

    The fields are converted to numbers, missing values are filled with 0,
    and the result is returned as an array.

    Args:
        df: Benign DataFrame.

    Returns:
        Array of shape (n_rows, 5).
    """
    numeric_df = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return numeric_df.values


def generate_sequences(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Slide a fixed size window over the data with a given stride.

    Args:
        data: Array of shape (n_rows, n_features).
        window_size: Number of flows in each sequence.
        stride: Step between the start of one window and the next.

    Returns:
        Array of shape (n_sequences, window_size, n_features).
    """
    n_rows = data.shape[0]
    sequences = []
    for start in range(0, n_rows - window_size + 1, stride):
        sequences.append(data[start:start + window_size])
    return np.array(sequences) if sequences else np.empty((0, window_size, data.shape[1]))


if __name__ == "__main__":
    print("=" * 60)
    print("Window and stride test")
    print("=" * 60)

    df = load_all_benign()
    numeric_data = prepare_numeric(df)
    print(f"\nNumeric feature matrix shape: {numeric_data.shape}")

    print("\n" + "=" * 60)
    print("Sequence count for different window and stride combinations")
    print("=" * 60)
    print(f"{'Window':>8} {'Stride':>8} {'Sequences':>12} {'Overlap %':>10}")

    # (window size, stride) pairs: no overlap, 50% overlap and stride 1
    configs = [
        (10, 10), (10, 5), (10, 1),
        (20, 20), (20, 10), (20, 1),
        (50, 50), (50, 25), (50, 1),
    ]

    for window_size, stride in configs:
        seqs = generate_sequences(numeric_data, window_size, stride)
        overlap_pct = round((1 - stride / window_size) * 100)
        print(f"{window_size:>8} {stride:>8} {len(seqs):>12} {overlap_pct:>9}%")

    print("\nGuidance:")
    print("* No overlap (stride = window size) gives the most independent")
    print("  sequences, but the fewest. Check that the count is workable.")
    print("* Stride 1 gives the most sequences, but heavy overlap means")
    print("  neighbouring sequences are highly correlated, so they are not")
    print("  fully independent training examples. This is a deliberate")
    print("  trade off and should be stated as a limitation.")