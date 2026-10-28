"""
Preprocessing pipeline for IoT-23 conn.log data.
Implements the four steps from Section 7.3 of the methodology:
  1. Data cleaning
  2. Missing value handling
  3. Feature transformation (categorical encoding)
  4. Feature scaling

"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from data_loading import RAW_DATA_DIR, list_scenarios, load_conn_log

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Column groups, per Table 7.3 ---
NUMERIC_FEATURES = [
    "duration", "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts",
]
CATEGORICAL_FEATURES = ["proto", "service", "conn_state"]

# Columns dropped entirely: identifiers and raw addresses carry no
# generalisable behavioural signal
DROP_COLUMNS = [
    "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "local_orig", "local_resp", "tunnel_parents", "history",
    "missed_bytes", "orig_ip_bytes", "resp_ip_bytes", "detailed-label",
]


# ---------- Step 1: Data cleaning ----------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-informative columns, duplicate rows, and invalid observations."""
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["ts", "label"])  # rows must have a timestamp and a label
    print(f"Cleaning: {before} -> {len(df)} rows "
          f"({before - len(df)} removed as duplicates/invalid)")
    return df.reset_index(drop=True)


# ---------- Steps 2-4: Missing values, encoding, scaling ----------

def build_transformer() -> ColumnTransformer:
    """
    Build the fit-once, reuse-everywhere preprocessing transformer.
    Numeric: median imputation -> min-max scaling.
    Categorical: mode imputation -> one-hot encoding.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformer = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return transformer


def fit_transform(df: pd.DataFrame, transformer: ColumnTransformer) -> np.ndarray:
    """Fit the transformer on this data and return the transformed array."""
    return transformer.fit_transform(df)


def transform(df: pd.DataFrame, transformer: ColumnTransformer) -> np.ndarray:
    """Apply an already-fitted transformer (e.g. to malware/test data)."""
    return transformer.transform(df)


def get_feature_names(transformer: ColumnTransformer) -> list[str]:
    """Human-readable names for the transformed feature columns (for debugging/EDA)."""
    return list(transformer.get_feature_names_out())


# ---------- Convenience: load + combine multiple scenarios ----------

def load_scenarios(names: list[str], raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and concatenate a list of named scenarios, sorted by timestamp."""
    frames = []
    for name in names:
        df = load_conn_log(raw_dir / name)
        df["scenario"] = name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    # --- Fit on benign scenarios (training distribution) ---
    benign_scenarios = [s for s in list_scenarios() if "Honeypot" in s]
    print(f"Loading benign scenarios: {benign_scenarios}")
    benign_df = load_scenarios(benign_scenarios)
    benign_df = clean(benign_df)

    transformer = build_transformer()
    benign_processed = fit_transform(benign_df, transformer)

    print(f"\nProcessed benign feature matrix shape: {benign_processed.shape}")
    print(f"Feature names: {get_feature_names(transformer)}")

    # Save the fitted transformer so malware/test scenarios can reuse
    # the EXACT SAME encoding and scaling later (critical for a fair
    # reconstruction-error comparison in Section 7.10)
    joblib.dump(transformer, PROCESSED_DIR / "fitted_transformer.joblib")
    np.save(PROCESSED_DIR / "benign_processed.npy", benign_processed)
    benign_df.to_csv(PROCESSED_DIR / "benign_cleaned.csv", index=False)

    print(f"\nSaved:")
    print(f"  - {PROCESSED_DIR / 'fitted_transformer.joblib'} (reuse this on malware data)")
    print(f"  - {PROCESSED_DIR / 'benign_processed.npy'} (numeric feature matrix)")
    print(f"  - {PROCESSED_DIR / 'benign_cleaned.csv'} (cleaned, pre-encoding, for inspection)")