"""
Full scan of the IoT-23 scenarios.

This is an exploration helper. It loads every scenario's conn.log file and
summarises the number of rows and the label distribution (benign, malicious
and other labels). The summary is saved to data/processed/scenario_summary.csv.

Loading every scenario can be slow and needs a lot of memory for the large
ones. For a quick line count only, use fast_prescan.py instead.

Run from the repository root:
    python src/scan_scenarios.py
"""

from pathlib import Path

import pandas as pd

from data_loading import RAW_DATA_DIR, list_scenarios, load_conn_log


def scan_all_scenarios(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load every scenario's conn.log file and summarise it.

    A scenario that cannot be loaded is reported as FAIL and gets empty
    values in the summary.

    Args:
        raw_dir: Folder that contains the scenario folders.

    Returns:
        DataFrame with one row per scenario and the columns scenario,
        total_rows, benign, malicious and other_labels.
    """
    scenarios = list_scenarios(raw_dir)
    summary_rows = []

    for name in scenarios:
        folder = raw_dir / name
        try:
            df = load_conn_log(folder)
            label_counts = df["label"].value_counts().to_dict()
            summary_rows.append({
                "scenario": name,
                "total_rows": len(df),
                "benign": label_counts.get("benign", 0),
                "malicious": label_counts.get("Malicious", 0) + label_counts.get("malicious", 0),
                "other_labels": {k: v for k, v in label_counts.items()
                                 if k.lower() not in ("benign", "malicious")},
            })
            print(f"OK   {name}: {len(df)} rows")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            summary_rows.append({
                "scenario": name, "total_rows": None,
                "benign": None, "malicious": None, "other_labels": None
            })

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    print("=" * 60)
    print("IoT-23 scenario scan")
    print("=" * 60)

    summary = scan_all_scenarios()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    # Save the summary table
    summary.to_csv("data/processed/scenario_summary.csv", index=False)
    print("\nSaved to data/processed/scenario_summary.csv")

    # Overall totals
    total = summary["total_rows"].sum()
    total_benign = summary["benign"].sum()
    total_malicious = summary["malicious"].sum()
    print(f"\nTotal rows across all scenarios: {total:,.0f}")
    print(f"Total benign:    {total_benign:,.0f}")
    print(f"Total malicious: {total_malicious:,.0f}")