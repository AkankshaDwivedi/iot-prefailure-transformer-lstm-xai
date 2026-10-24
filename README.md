# IoT Pre-Failure Detection: Transformer vs LSTM Autoencoders with Explainable AI

Code for the Masters thesis **"Predictive Pre-Failure Detection of IoT Device Messaging with Generative and Explainable AI"**.

The project learns what normal IoT device messaging looks like, turns the reconstruction error into a graded **Pre-Failure Score**, and explains each warning with **SHAP** and **attention**. A Transformer Autoencoder is compared with an LSTM Autoencoder baseline on the public IoT-23 dataset.

## Research questions

| RQ | Question |
|----|----------|
| RQ1 | How well can a Transformer Autoencoder learn normal IoT communication behaviour? |
| RQ2 | Can reconstruction error show early behavioural drift before failure? |
| RQ3 | How does the Transformer compare with an LSTM Autoencoder? |
| RQ4 | Can SHAP and attention explain the results? |
| RQ5 | Is the framework suitable for real deployment? |

## Method in brief

1. **Data:** IoT-23 connection logs (Zeek). Benign traffic only is used for training.
2. **Sequences:** windows of 10 flows, stride 5 (50% overlap). 1,956 cleaned benign flows give 390 benign sequences. The malware test set has 3,200 sequences (2,801 malicious, 399 benign).
3. **Augmentation:** four noisy copies of each training sequence (about 1% Gaussian jitter), on training data only.
4. **Models:** an LSTM Autoencoder and a Transformer Autoencoder with a mean pooled latent bottleneck, both trained with mean squared error and early stopping (patience 15).
5. **Validation:** 5-fold cross-validation, then final models on a 70/30 train and monitor split.
6. **Pre-Failure Score:** four levels (Normal, Low drift, Pre-failure, Critical) from the 90th, 97th and 99.5th percentiles of benign reconstruction error. A sequence is flagged at Pre-failure or Critical.
7. **Explainability:** SHAP (KernelExplainer) for feature level explanations and attention weights for time step explanations.

## Results (malware test set)

| Metric | LSTM Autoencoder | Transformer Autoencoder |
|--------|------------------|-------------------------|
| Precision | 1.0000 | 1.0000 |
| Recall | 0.0346 | 0.6501 |
| F1 score | 0.0669 | 0.7880 |
| ROC AUC | 0.8731 | 0.9793 |
| False positive rate | 0 | 0 |

Confusion counts (TP, FP, TN, FN): LSTM 97, 0, 399, 2,704. Transformer 1,821, 0, 399, 980.

Both models rank the connection state `S0` (a connection attempt with no reply, a sign of scanning) as the most important feature. For the 50 highest error sequences, Transformer attention peaks at time step 8 for 25 of them.

The final metrics are saved in `outputs/results/final_metrics_summary.csv`.

## Repository structure

```
iot-prefailure-transformer-lstm-xai/
├── README.md
├── requirements.txt
├── Makefile                          # one command per pipeline step
├── .gitignore
├── src/
│   ├── data_loading.py               # load IoT-23 conn.log files
│   ├── preprocessing.py              # cleaning, encoding and scaling
│   ├── sequence_generation.py        # windows, stride, augmentation, k-fold
│   ├── models/
│   │   ├── lstm_auto_encoder.py      # LSTM Autoencoder and k-fold training
│   │   └── transformer_auto_encoder.py  # Transformer Autoencoder and k-fold training
│   ├── train_final.py                # final models on the 70/30 split
│   ├── pre_failure_score_v2.py       # score malware sequences
│   ├── recalibrate_final.py          # thresholds from the final model's benign errors
│   ├── finalize_metrics.py           # precision, recall, F1 and FPR
│   ├── compute_roc_auc.py            # ROC AUC and ROC curve
│   ├── explainability_shap.py        # SHAP feature importance
│   ├── explainability_attention.py   # attention visualisation
│   └── scan_scenarios.py, fast_prescan.py, sample_large_scenario.py,
│       test_sequence_generation.py, pre_failure_score.py   # exploration helpers and an earlier version
└── outputs/
    ├── figures/                      # ROC curve, SHAP chart, attention figures
    └── results/                      # metrics, scores, benign errors, final model weights
```

## Setup

Python 3.12 is used.

```
make install
```

This creates a `venv` folder and installs the libraries in `requirements.txt` (pandas, numpy, scikit-learn, torch, shap, matplotlib, plotly and jupyter). To do it by hand:

```
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The IoT-23 dataset is not included in this repository. Download it from the Stratosphere Laboratory:
https://www.stratosphereips.org/datasets-iot23

Place the scenario folders in `data/raw/IoTScenarios/` (this folder is ignored by git). Processed files are written to `data/processed/`.

## How to run

Run the steps in this order, from the repository root:

```
make data-loading            # check that the IoT-23 files load
make preprocess              # clean, encode and scale the benign data
make sequence-generation     # build the benign sequences
make train-lstm              # 5-fold training, LSTM Autoencoder
make train-transformer       # 5-fold training, Transformer Autoencoder
make train-final             # final models on the 70/30 split
make pre-failure-score-v2    # score the malware sequences
make recalibrate-final       # thresholds from the final model's benign errors
make finalize-metrics        # precision, recall, F1 and false positive rate
make compute-roc-auc         # ROC AUC and ROC curve
make explain-shap            # SHAP feature importance
make explain-attention       # attention figures
```

Results are written to `outputs/results/` and figures to `outputs/figures/`.

## Limitations

* The benign set is small (390 sequences) and only one dataset is used.
* The Transformer still misses about 35% of malicious sequences.
* Sequence labels are mixed: a sequence is malicious if any flow inside it is malicious.
* Attention is indicative only, and SHAP is computationally expensive.

## Future work

Test on more IoT datasets and live traffic, collect more benign data, tune the thresholds for each deployment, try other generative models, and plan retraining to follow concept drift.

## Citation

If you use this work, please cite the thesis:

```
Dwivedi, A. (2026) Predictive Pre-Failure Detection of IoT Device Messaging
with Generative and Explainable AI. MSc Data Science thesis. Liverpool John Moores University.
```