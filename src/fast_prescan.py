"""
Fast pre scan of the IoT-23 scenarios.

This is an exploration helper. It counts the lines in each scenario's
conn.log file without loading anything into pandas, so it finishes quickly
even for very large files. The totals show how much benign (Honeypot) data
there is compared with malware data.

Run from the repository root:
    python src/fast_prescan.py
"""

import subprocess
from pathlib import Path

from data_loading import RAW_DATA_DIR, list_scenarios, find_conn_log


def fast_line_count(filepath: Path) -> int:
    """
    Count the lines in a file without loading it into pandas.

    Gzipped files (.gz) are read with zcat. Other files are read with wc.
    This needs a Mac or Linux terminal.

    Args:
        filepath: Path to the file.

    Returns:
        Number of lines, or -1 if the count could not be read.
    """
    if filepath.suffix == ".gz":
        result = subprocess.run(
            f"zcat < '{filepath}' | wc -l", shell=True,
            capture_output=True, text=True
        )
    else:
        result = subprocess.run(
            ["wc", "-l", str(filepath)], capture_output=True, text=True
        )
    try:
        return int(result.stdout.split()[0])
    except (IndexError, ValueError):
        return -1


if __name__ == "__main__":
    print("=" * 50)
    print("Fast pre scan of IoT-23 scenarios")
    print("=" * 50)

    scenarios = list_scenarios(RAW_DATA_DIR)
    print(f"Pre scanning {len(scenarios)} scenarios (line counts only, no parsing)...\n")

    # Count the lines in each scenario's conn.log file
    results = []
    for name in scenarios:
        folder = RAW_DATA_DIR / name
        try:
            log_file = find_conn_log(folder)
            n_lines = fast_line_count(log_file)
            results.append((name, n_lines))
            print(f"{name}: ~{n_lines:,} lines")
        except FileNotFoundError as e:
            print(f"{name}: FAILED - {e}")
            results.append((name, None))

    # Summary: Honeypot scenarios are benign, Malware scenarios are malicious
    print("\n" + "=" * 50)
    benign = [r for r in results if "Honeypot" in r[0]]
    malware = [r for r in results if "Malware" in r[0]]

    benign_total = sum(r[1] for r in benign if r[1])
    malware_total = sum(r[1] for r in malware if r[1])

    print(f"Benign scenarios total (approx):  {benign_total:,} lines")
    print(f"Malware scenarios total (approx): {malware_total:,} lines")
    print(f"Ratio malware:benign = {malware_total / max(benign_total, 1):.1f}:1")