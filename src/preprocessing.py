import pandas as pd
import numpy as np


def load_longitudinal_data(path):
    """
    Load longitudinal stacked dataset
    """
    df = pd.read_csv(path, low_memory=False)
    return df


def sort_by_patient_stage(df):
    """
    Ensure rows are ordered by patient and developmental stage
    """
    stage_order = {
        "neonatal": 0,
        "infant": 1,
        "toddler": 2,
        "child": 3,
        "preteen": 4,
        "adolescent": 5
    }

    df["stage_index"] = df["stage"].map(stage_order)
    df = df.sort_values(["patient_id", "stage_index"])
    return df


def normalize_numeric_columns(df):
    """
    Force all feature columns to numeric and fill missing values
    """

    # Convert all columns except identifiers to numeric
    feature_cols = [col for col in df.columns if col not in ["patient_id", "stage", "stage_index"]]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    return df


def build_patient_sequences(df):
    """
    Convert dataframe into sequences per patient
    Output:
        sequences: list of arrays [num_patients, time_steps, features]
    """

    patients = df["patient_id"].unique()
    sequences = []

    feature_cols = [col for col in df.columns if col not in ["patient_id", "stage", "stage_index"]]

    for pid in patients:
        patient_data = df[df["patient_id"] == pid]
        patient_data = patient_data.sort_values("stage_index")

        sequence = patient_data[feature_cols].values
        sequences.append(sequence)

    return sequences, feature_cols


def prepare_dataset(path):
    """
    Full preprocessing pipeline
    """

    df = load_longitudinal_data(path)
    df = sort_by_patient_stage(df)
    df = normalize_numeric_columns(df)

    sequences, features = build_patient_sequences(df)

    print("Total patients:", len(sequences))
    print("Features used:", len(features))

    return sequences, features