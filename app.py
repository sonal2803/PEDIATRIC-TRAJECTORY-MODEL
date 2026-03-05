import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
project_root = Path(__file__).parent.absolute()
env_path = project_root / ".env"
load_dotenv(env_path)

# ---------------------------------------------------
# Add src to path
# ---------------------------------------------------
sys.path.append(str(project_root / "src"))

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM
from evaluation.trajectory_simulator import simulate_future
from disease_domain_classifier import DiseaseDomainClassifier
from llm_explainer import generate_detailed_explanation

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
DATA_PATH = "data/Longitudinal_Master_Dataset.csv"
MODEL_PATH = "models/trained_trajectory_model.pt"

st.set_page_config(page_title="Pediatric Neuro Risk Simulator", layout="wide")
st.title("🧠 Pediatric Neurological Risk Projection Simulator")

# ---------------------------------------------------
# Stage Selection
# ---------------------------------------------------
stage_options = {
    "Neonatal (0–1 month)": 0,
    "Infant (1–12 months)": 1,
    "Toddler (1–3 years)": 2,
    "Child (4–9 years)": 3,
    "Pre-Teen (10–12 years)": 4,
    "Adolescent (13–18 years)": 5,
}

selected_stage = st.selectbox(
    "Select Current Developmental Stage",
    list(stage_options.keys()),
    key="stage_selector"
)

stage_index = stage_options[selected_stage]

# ---------------------------------------------------
# Clinical Inputs
# ---------------------------------------------------
st.subheader("Clinical Indicators")

col1, col2 = st.columns(2)

with col1:
    seizures = st.selectbox("Early Seizures",
                            ["None", "Occasional", "Recurrent"],
                            key="seizure_selector")

    structural = st.selectbox("Structural Brain Abnormality",
                              ["None", "Mild", "Moderate", "Severe"],
                              key="structural_selector")

    cognitive = st.selectbox("Cognitive Delay",
                             ["None", "Mild", "Moderate", "Severe"],
                             key="cognitive_selector")

with col2:
    genetic = st.selectbox("Genetic Risk",
                           ["Low", "Moderate", "High"],
                           key="genetic_selector")

    birth = st.selectbox("Birth Complications",
                         ["None", "Mild", "Significant"],
                         key="birth_selector")

explanation_mode = st.radio(
    "Explanation Mode",
    ["Parent-Friendly", "Clinical Detail"],
    key="explanation_mode_selector"
)

disease_input = st.text_input(
    "Diagnosed Condition (Optional)",
    key="disease_input_box"
)

# ---------------------------------------------------
# Feature Mapping
# ---------------------------------------------------
def map_inputs_to_features(feature_dim, domain):
    vec = np.zeros(feature_dim)

    seizure_map = {"None": 0.0, "Occasional": 0.4, "Recurrent": 0.8}
    structural_map = {"None": 0.0, "Mild": 0.3, "Moderate": 0.6, "Severe": 1.0}
    cognitive_map = {"None": 0.0, "Mild": 0.3, "Moderate": 0.6, "Severe": 1.0}
    genetic_map = {"Low": 0.1, "Moderate": 0.5, "High": 0.9}
    birth_map = {"None": 0.0, "Mild": 0.4, "Significant": 0.8}

    vec[0:10] += structural_map[structural] * 1.2
    vec[10:20] += cognitive_map[cognitive] * 1.0
    vec[20:30] += seizure_map[seizures] * 1.3
    vec[30:40] += genetic_map[genetic] * 0.8
    vec[40:50] += birth_map[birth] * 0.9

    if domain == "neurodegenerative":
        vec[10:30] += 0.6
    elif domain == "genetic_epileptic":
        vec[20:30] += 0.8
    elif domain == "neuroinflammatory":
        vec[0:30] += 0.6

    return vec


# ---------------------------------------------------
# Domain-Weighted Risk
# ---------------------------------------------------
def compute_weighted_risk(sequence, domain):

    structural_risk = sequence[:, 0:10].mean(axis=1)
    cognitive_risk = sequence[:, 10:20].mean(axis=1)
    seizure_risk = sequence[:, 20:30].mean(axis=1)

    if domain == "genetic_epileptic":
        return 0.2*structural_risk + 0.3*cognitive_risk + 0.5*seizure_risk
    elif domain == "neurodegenerative":
        return 0.4*structural_risk + 0.4*cognitive_risk + 0.2*seizure_risk
    elif domain == "neuroinflammatory":
        return 0.35*structural_risk + 0.35*cognitive_risk + 0.3*seizure_risk
    else:
        return 0.33*structural_risk + 0.33*cognitive_risk + 0.34*seizure_risk


# ---------------------------------------------------
# Trend Detection
# ---------------------------------------------------
def detect_trend(mean_risk):
    slope = np.polyfit(range(len(mean_risk)), mean_risk, 1)[0]
    if slope > 0.002:
        return "rising"
    elif slope < -0.002:
        return "declining"
    else:
        return "stable"


# ---------------------------------------------------
# Simulation
# ---------------------------------------------------
if st.button("Run Simulation", key="run_sim_button"):

    classifier = DiseaseDomainClassifier()
    domain = classifier.classify(disease_input)

    sequences, _ = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrajectoryLSTM(input_size=dataset.feature_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    initial_vector = map_inputs_to_features(dataset.feature_dim, domain)

    initial_sequence = np.zeros((stage_index + 1, dataset.feature_dim))
    initial_sequence[-1] = initial_vector

    initial_tensor = torch.tensor(initial_sequence,
                                  dtype=torch.float32).unsqueeze(0).to(device)

    futures = simulate_future(model, initial_tensor, domain, steps=5, simulations=50)

    risk_trajectories = []
    for f in futures:
        risk = compute_weighted_risk(f, domain)
        risk_trajectories.append(risk)

    risk_trajectories = np.array(risk_trajectories)

    mean_risk = risk_trajectories.mean(axis=0)
    std_risk = risk_trajectories.std(axis=0)

    p25 = np.percentile(risk_trajectories, 25, axis=0)
    p75 = np.percentile(risk_trajectories, 75, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mean_risk))

    ax.plot(x, mean_risk, linewidth=3)
    ax.fill_between(x, mean_risk-std_risk, mean_risk+std_risk, alpha=0.2)
    ax.fill_between(x, p25, p75, alpha=0.15)

    ax.axhspan(0, 0.2, alpha=0.05)
    ax.axhspan(0.2, 0.4, alpha=0.05)
    ax.axhspan(0.4, 1.0, alpha=0.05)

    ax.set_title("Domain-Weighted Projected Neurological Risk")
    ax.set_xlabel("Future Development Stage")
    ax.set_ylabel("Risk Index")
    ax.set_ylim(
    max(0, mean_risk.min() - 0.02),
    mean_risk.max() + 0.02
)

    st.pyplot(fig)

    final_risk = mean_risk[-1]

    if final_risk < 0.2:
        category = "Low Risk"
    elif final_risk < 0.4:
        category = "Moderate Risk"
    else:
        category = "High Risk"

    trend = detect_trend(mean_risk)

    st.subheader(f"Risk Category: {category}")

    try:
        explanation = generate_detailed_explanation(
            disease_input=disease_input,
            domain=domain,
            category=category,
            mean_risk=mean_risk,
            selected_stage=selected_stage,
            explanation_mode=explanation_mode,
            trend=trend,
            variability=float(std_risk.mean())
        )

        st.subheader("Detailed Neurodevelopmental Interpretation")
        st.write(explanation)

    except Exception as e:
        st.error(f"LLM explanation failed: {e}")

    st.write("Mean Risk Values:", mean_risk)

    st.caption("⚠️ This tool provides probabilistic projections and does not constitute medical diagnosis.")