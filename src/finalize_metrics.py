"""
Final threshold application and metrics.

This script applies the final Pre-Failure Score thresholds and computes
Precision, Recall, F1 and False Positive Rate for both models.

It reuses the malware errors and ground truth already saved in
malware_prefailure_scores_v2.csv. Only the thresholds are different
(they come from the final models' benign errors, as in recalibrate_final.py),
so there is no need to rescore the malware data or retrain anything.

Pre-Failure Score levels, from the benign error percentiles:
    Normal       up to the 90th percentile
    Low drift    90th to 97th percentile
    Pre-failure  97th to 99.5th percentile
    Critical     above the 99.5th percentile

A sequence is counted as anomalous at Pre-failure or Critical.

Run from the repository root:
    python src/finalize_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("outputs/results")


def calibrate_thresholds(benign_errors: np.ndarray) -> dict:
    """
    Compute the level thresholds from the benign reconstruction errors.

    Args:
        benign_errors: Reconstruction errors of benign sequences.

    Returns:
        Dictionary with the 90th, 97th and 99.5th percentile values.
    """
    return {
        "normal_max": np.percentile(benign_errors, 90),
        "low_drift_max": np.percentile(benign_errors, 97),
        "pre_failure_max": np.percentile(benign_errors, 99.5),
    }


def assign_level(error: float, thresholds: dict) -> str:
    """
    Convert one reconstruction error into a Pre-Failure Score level.

    Args:
        error: Reconstruction error of one sequence.
        thresholds: Dictionary from calibrate_thresholds().

    Returns:
        "Normal", "Low drift", "Pre-failure" or "Critical".
    """
    if error <= thresholds["normal_max"]:
        return "Normal"
    elif error <= thresholds["low_drift_max"]:
        return "Low drift"
    elif error <= thresholds["pre_failure_max"]:
        return "Pre-failure"
    else:
        return "Critical"


def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> dict:
    """
    Compute the confusion counts and the main detection metrics.

    Args:
        y_true: Ground truth labels, "Malicious" or "Benign".
        y_pred_binary: Predictions, "Anomalous" or "Normal".

    Returns:
        Dictionary with TP, FP, TN, FN, Precision, Recall, F1 and FPR.
    """
    tp = np.sum((y_true == "Malicious") & (y_pred_binary == "Anomalous"))
    fp = np.sum((y_true == "Benign") & (y_pred_binary == "Anomalous"))
    tn = np.sum((y_true == "Benign") & (y_pred_binary == "Normal"))
    fn = np.sum((y_true == "Malicious") & (y_pred_binary == "Normal"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Precision": precision, "Recall": recall, "F1": f1, "FPR": fpr}


if __name__ == "__main__":
    print("=" * 50)
    print("Final thresholds and metrics")
    print("=" * 50)

    # Load the malware errors and ground truth saved earlier
    results = pd.read_csv(RESULTS_DIR / "malware_prefailure_scores_v2.csv")
    print(f"Loaded {len(results)} previously scored sequences")
    print(f"Ground truth distribution:\n{results['ground_truth'].value_counts()}")

    # Benign errors of the final models
    lstm_benign_errors = np.load(RESULTS_DIR / "lstm_final_benign_errors.npy")
    transformer_benign_errors = np.load(RESULTS_DIR / "transformer_final_benign_errors.npy")

    # Thresholds for each model
    lstm_thresholds = calibrate_thresholds(lstm_benign_errors)
    transformer_thresholds = calibrate_thresholds(transformer_benign_errors)

    print(f"\nLSTM thresholds: {lstm_thresholds}")
    print(f"Transformer thresholds: {transformer_thresholds}")

    # Assign a Pre-Failure Score level to every malware set sequence
    results["lstm_level"] = results["lstm_error"].apply(lambda e: assign_level(e, lstm_thresholds))
    results["transformer_level"] = results["transformer_error"].apply(
        lambda e: assign_level(e, transformer_thresholds)
    )

    print("\nLSTM Pre-Failure Score distribution:")
    print(results["lstm_level"].value_counts())
    print("\nTransformer Pre-Failure Score distribution:")
    print(results["transformer_level"].value_counts())

    # Collapse the four levels into a binary prediction
    lstm_binary = np.where(
        results["lstm_level"].isin(["Normal", "Low drift"]), "Normal", "Anomalous"
    )
    transformer_binary = np.where(
        results["transformer_level"].isin(["Normal", "Low drift"]), "Normal", "Anomalous"
    )

    # Compute the metrics against the ground truth
    lstm_metrics = compute_metrics(results["ground_truth"].values, lstm_binary)
    transformer_metrics = compute_metrics(results["ground_truth"].values, transformer_binary)

    print("\n" + "=" * 50)
    print("LSTM evaluation against ground truth:")
    for k, v in lstm_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nTransformer evaluation against ground truth:")
    for k, v in transformer_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save the scored results with the final levels
    results.to_csv(RESULTS_DIR / "malware_prefailure_scores_final.csv", index=False)
    print(f"\nSaved final results to {RESULTS_DIR / 'malware_prefailure_scores_final.csv'}")