"""
Data loading for the IoT-23 dataset.

This module finds the capture scenario folders and loads the labelled
Zeek conn.log files into pandas DataFrames. Other scripts import
load_conn_log() and list_scenarios() from here.

The IoT-23 files are not included in the repository. Download them from
https://www.stratosphereips.org/datasets-iot23 and place the scenario
folders in data/raw/IoTScenarios/.

Run from the repository root to check that the files load:
    python src/data_loading.py
"""

from pathlib import Path

import pandas as pd

# Folder that holds one sub folder per capture scenario
RAW_DATA_DIR = Path("data/raw/IoTScenarios")

# Column names of the labelled Zeek conn.log file, in file order
COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
    "resp_ip_bytes", "tunnel_parents", "label", "detailed-label",
]


def list_scenarios(raw_dir: Path = RAW_DATA_DIR) -> list[str]:
    """
    List all available capture scenario folders.

    Args:
        raw_dir: Folder that contains the scenario folders.

    Returns:
        Sorted list of folder names, for example "CTU-Honeypot-Capture-4-1".
    """
    return sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])


def find_conn_log(scenario_folder: Path) -> Path:
    """
    Find the labelled conn.log file inside a scenario folder.

    The file may sit in a sub folder, so the search goes through all levels.

    Args:
        scenario_folder: Path to one scenario folder.

    Returns:
        Path to the first conn.log.labeled file found.

    Raises:
        FileNotFoundError: If no such file exists under the folder.
    """
    matches = list(scenario_folder.rglob("conn.log.labeled*"))
    if not matches:
        raise FileNotFoundError(f"No conn.log.labeled file found under {scenario_folder}")
    return matches[0]


def load_conn_log(scenario_folder: Path) -> pd.DataFrame:
    """
    Load one scenario's labelled conn.log file into a DataFrame.

    Lines that start with # are comments and are skipped. A dash in the
    file means a missing value and is read as NaN.

    Args:
        scenario_folder: Path to one scenario folder.

    Returns:
        DataFrame with one row per connection and the columns in COLUMNS.
    """
    log_file = find_conn_log(scenario_folder)
    df = pd.read_csv(
        log_file,
        sep=r"\s+",
        comment="#",
        names=COLUMNS,
        na_values="-",
        low_memory=False,
    )
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("IoT-23 data loading check")
    print("=" * 50)

    # List the scenarios that were found
    scenarios = list_scenarios()
    print(f"Found {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  - {s}")

    # Load the first scenario as a quick test
    if scenarios:
        first = RAW_DATA_DIR / scenarios[0]
        print(f"\nLoading first scenario: {scenarios[0]}")
        try:
            df = load_conn_log(first)
            print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            print(df.head())
            print(f"\nLabel distribution:\n{df['label'].value_counts()}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Check the folder structure with: find data/raw -name 'conn.log.labeled*'")