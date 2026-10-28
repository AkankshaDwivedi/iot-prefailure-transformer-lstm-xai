"""
Sampling helper for very large IoT-23 scenarios.

Some malware scenarios have tens of millions of lines, which is too much to
load into memory. This module reads the labelled conn.log file in chunks
and returns a random sample. It is used by pre_failure_score.py to build
the malware test set.

Run from the repository root for a quick test:
    python src/sample_large_scenario.py
"""

from pathlib import Path

import pandas as pd

from data_loading import RAW_DATA_DIR, find_conn_log, COLUMNS


def sample_large_conn_log(
    scenario_folder: Path,
    n_rows: int = 50_000,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Read a large conn.log.labeled file in chunks and return a sample.

    The full file is never held in memory. From each chunk, starting at the
    beginning of the file, a random slice of rows is taken (fixed seed, so
    the sample is repeatable). Reading stops as soon as enough rows have
    been collected.

    Args:
        scenario_folder: Path to one scenario folder.
        n_rows: Number of rows wanted in the sample.
        chunksize: Number of rows read from the file at a time.

    Returns:
        DataFrame with at most n_rows rows.
    """
    log_file = find_conn_log(scenario_folder)
    reader = pd.read_csv(
        log_file, sep=r"\s+", comment="#", names=COLUMNS,
        na_values="-", low_memory=False, chunksize=chunksize,
    )

    sampled_chunks = []
    rows_collected = 0
    for i, chunk in enumerate(reader):
        # Take a proportional slice from each chunk
        take = max(1, int(chunksize * (n_rows / 1_000_000)))
        sampled_chunks.append(chunk.sample(min(take, len(chunk)), random_state=42))
        rows_collected += len(sampled_chunks[-1])
        if rows_collected >= n_rows:
            break

    result = pd.concat(sampled_chunks, ignore_index=True)
    return result.head(n_rows)


if __name__ == "__main__":
    print("=" * 50)
    print("Large scenario sampling check")
    print("=" * 50)

    # Example: sample 50,000 rows from a large malware scenario (about 73.5M lines)
    scenario = RAW_DATA_DIR / "CTU-IoT-Malware-Capture-39-1"
    print(f"Sampling from {scenario.name} "
          f"(reads in chunks, so it does not load 73M rows at once)...")
    sample = sample_large_conn_log(scenario, n_rows=50_000)
    print(f"Sampled {len(sample)} rows")
    print(f"\nLabel distribution:\n{sample['label'].value_counts()}")