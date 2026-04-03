import pandas as pd
import numpy as np

STAGE_ORDER = {
    "neonatal": 0,
    "infant": 1,
    "toddler": 2,
    "child": 3,
    "preteen": 4,
    "adolescent": 5
}


def load_longitudinal_data(path):
    """Load longitudinal stacked dataset."""
    df = pd.read_csv(path, low_memory=False)
    return df


def sort_by_patient_stage(df):
    """Ensure rows are ordered by patient and developmental stage."""
    df["stage_index"] = df["stage"].str.lower().map(STAGE_ORDER)
    df = df.sort_values(["patient_id", "stage_index"])
    return df


def normalize_numeric_columns(df):
    """
    Force all feature columns to numeric, fill missing values,
    and clip to [0, 1] to ensure valid normalized range.
    """
    feature_cols = [
        col for col in df.columns
        if col not in ["patient_id", "stage", "stage_index"]
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Clip each feature column to [0, 1]
    df[feature_cols] = df[feature_cols].clip(0.0, 1.0)
    return df


def build_patient_sequences(df):
    """
    Convert dataframe into per-patient sequences.

    Returns:
        sequences : list of np.ndarray, each shape [num_stages, num_features]
        feature_cols : list of feature column names
    """
    feature_cols = [
        col for col in df.columns
        if col not in ["patient_id", "stage", "stage_index"]
    ]
    patients = df["patient_id"].unique()
    sequences = []
    for pid in patients:
        patient_data = df[df["patient_id"] == pid].sort_values("stage_index")
        sequence = patient_data[feature_cols].values.astype(np.float32)
        # Only keep patients who have at least 2 developmental stages
        if sequence.shape[0] >= 2:
            sequences.append(sequence)
    return sequences, feature_cols


def prepare_dataset(path):
    """Full preprocessing pipeline."""
    df = load_longitudinal_data(path)
    df = sort_by_patient_stage(df)
    df = normalize_numeric_columns(df)
    sequences, features = build_patient_sequences(df)
    print(f"Total patients (>=2 stages): {len(sequences)}")
    print(f"Features used: {len(features)}")
    return sequences, features