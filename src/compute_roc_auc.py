"""
ROC AUC and ROC curve.

This script completes the metric set for the final comparison.

It uses the continuous reconstruction error scores directly (not the
discretised Pre-Failure Score levels) against the binary ground truth.
This evaluates detection performance across all possible thresholds,
instead of the single level based cutoff used for Precision, Recall,
F1 and False Positive Rate.

Steps:
    1. Load the final malware scores.
    2. Compute the ROC AUC for the LSTM and the Transformer.
    3. Save a ROC curve comparison figure.
    4. Recompute Precision, Recall, F1 and FPR from the same file.
    5. Save the final metrics summary table.

Run from the repository root:
    python src/compute_roc_auc.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 50)
    print("ROC AUC and ROC curve")
    print("=" * 50)

    # Load the final scores and build the binary ground truth (1 = malicious)
    results = pd.read_csv(RESULTS_DIR / "malware_prefailure_scores_final.csv")
    y_true = (results["ground_truth"] == "Malicious").astype(int).values
    print(f"Loaded {len(results)} sequences "
          f"({y_true.sum()} malicious, {len(y_true) - y_true.sum()} benign)")

    # ROC AUC from the continuous reconstruction errors
    lstm_auc = roc_auc_score(y_true, results["lstm_error"].values)
    transformer_auc = roc_auc_score(y_true, results["transformer_error"].values)

    print(f"\nLSTM ROC AUC:        {lstm_auc:.4f}")
    print(f"Transformer ROC AUC: {transformer_auc:.4f}")

    # ROC curve points for both models
    lstm_fpr, lstm_tpr, _ = roc_curve(y_true, results["lstm_error"].values)
    transformer_fpr, transformer_tpr, _ = roc_curve(y_true, results["transformer_error"].values)

    # Plot and save the comparison figure
    plt.figure(figsize=(7, 6))
    plt.plot(lstm_fpr, lstm_tpr, label=f"LSTM Autoencoder (AUC = {lstm_auc:.3f})", linewidth=2)
    plt.plot(transformer_fpr, transformer_tpr,
             label=f"Transformer Autoencoder (AUC = {transformer_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: LSTM vs Transformer Autoencoder\n(IoT-23 Pre-Failure Detection)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=200)
    print(f"\nSaved ROC curve to {FIGURES_DIR / 'roc_curve_comparison.png'}")

    # Recompute Precision, Recall, F1 and FPR fresh from this results file,
    # rather than hardcoding old values from an earlier run
    def _metrics_from_level(level_col):
        """
        Compute Precision, Recall, F1 and FPR from a Pre-Failure Score level column.

        Normal and Low drift count as normal. Pre-failure and Critical
        count as anomalous.
        """
        binary = np.where(results[level_col].isin(["Normal", "Low drift"]), "Normal", "Anomalous")
        tp = np.sum((y_true == 1) & (binary == "Anomalous"))
        fp = np.sum((y_true == 0) & (binary == "Anomalous"))
        tn = np.sum((y_true == 0) & (binary == "Normal"))
        fn = np.sum((y_true == 1) & (binary == "Normal"))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return precision, recall, f1, fpr

    lstm_p, lstm_r, lstm_f1, lstm_fpr = _metrics_from_level("lstm_level")
    tf_p, tf_r, tf_f1, tf_fpr = _metrics_from_level("transformer_level")

    # Build and save the final summary table
    summary = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score", "FPR", "ROC-AUC"],
        "LSTM Autoencoder": [round(lstm_p, 4), round(lstm_r, 4), round(lstm_f1, 4),
                             round(lstm_fpr, 4), round(lstm_auc, 4)],
        "Transformer Autoencoder": [round(tf_p, 4), round(tf_r, 4), round(tf_f1, 4),
                                    round(tf_fpr, 4), round(transformer_auc, 4)],
    })
    summary.to_csv(RESULTS_DIR / "final_metrics_summary.csv", index=False)

    print("\nFinal metrics summary")
    print("-" * 50)
    print(summary.to_string(index=False))
    print("-" * 50)
    print(f"Saved summary table to {RESULTS_DIR / 'final_metrics_summary.csv'}")