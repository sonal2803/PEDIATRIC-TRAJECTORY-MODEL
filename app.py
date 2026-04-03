import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Path Setup
# ──────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.absolute()
load_dotenv(project_root / ".env")
sys.path.append(str(project_root / "src"))

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM
from evaluation.trajectory_simulator import simulate_future
from disease_domain_classifier import DiseaseDomainClassifier
from llm_explainer import generate_detailed_explanation

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/Longitudinal_Master_Dataset.csv"
MODEL_PATH = "models/trained_trajectory_model.pt"

STAGE_OPTIONS = {
    "Neonatal (0–1 month)":    0,
    "Infant (1–12 months)":    1,
    "Toddler (1–3 years)":     2,
    "Child (4–9 years)":       3,
    "Pre-Teen (10–12 years)":  4,
    "Adolescent (13–18 years)": 5,
}

STAGE_LABELS = ["Neonatal", "Infant", "Toddler", "Child", "Pre-Teen", "Adolescent"]

DOMAIN_DISPLAY = {
    "neurodegenerative": "Neurodegenerative Disorder",
    "genetic_epileptic": "Genetic Epileptic Encephalopathy",
    "neuroinflammatory": "Neuroinflammatory Condition",
    "metabolic":         "Metabolic Encephalopathy",
    "structural":        "Structural Brain Malformation",
    "vascular":          "Vascular Neurological Injury",
    "demyelinating":     "Demyelinating Disorder",
}

# Domain-specific weights (ws, wc, we) matching risk_stratification.py
DOMAIN_WEIGHTS = {
    "genetic_epileptic":  (0.20, 0.30, 0.50),
    "neurodegenerative":  (0.40, 0.40, 0.20),
    "neuroinflammatory":  (0.35, 0.35, 0.30),
    "structural":         (0.50, 0.30, 0.20),
    "metabolic":          (0.30, 0.40, 0.30),
    "vascular":           (0.45, 0.35, 0.20),
    "demyelinating":      (0.30, 0.45, 0.25),
}


# ──────────────────────────────────────────────────────────────────────────────
# Cached resource loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_dataset():
    sequences, _ = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryLSTM(input_size=dataset.feature_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    return model, dataset, device


@st.cache_resource
def load_classifier():
    return DiseaseDomainClassifier()


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────
def map_inputs_to_features(feature_dim: int, domain: str, inputs: dict) -> np.ndarray:
    """
    Convert UI clinical inputs into a normalised feature vector in [0, 1].

    Feature layout (matches paper):
        Cols  0– 9 : structural
        Cols 10–19 : cognitive
        Cols 20–29 : epileptic / seizure
        Cols 30–39 : genetic
        Cols 40–49 : perinatal
        Cols 50–51 : auxiliary (set to 0)

    Ordinal mappings are kept within [0, 1]; domain priors add calibrated
    small offsets, clamped at the end.
    """
    vec = np.zeros(feature_dim, dtype=np.float32)

    # ── Ordinal input mappings ──────────────────────────────────────
    severity_map   = {"None": 0.0, "Mild": 0.30, "Moderate": 0.60, "Severe": 0.90}
    frequency_map  = {"None": 0.0, "Occasional (< monthly)": 0.25,
                      "Monthly": 0.50, "Weekly": 0.75, "Daily": 0.95}
    genetic_map    = {"Low": 0.10, "Moderate": 0.50, "High": 0.90}
    birth_map      = {"None": 0.0, "Mild": 0.35, "Significant": 0.75}
    regression_map = {"None": 0.0, "Mild plateauing": 0.35,
                      "Clear regression": 0.70, "Severe regression": 0.95}
    mri_map        = {
        "Normal": 0.0,
        "Non-specific white matter changes": 0.25,
        "Cortical atrophy / volume loss": 0.55,
        "Structural malformation": 0.75,
        "Severe / progressive changes": 0.95,
    }
    motor_map      = {"None": 0.0, "Mild (walks with support)": 0.30,
                      "Moderate (wheelchair part-time)": 0.60,
                      "Severe (non-ambulant)": 0.90}

    # ── Assign to feature groups ─────────────────────────────────
    # Structural (0-9)
    s_val = (mri_map[inputs["mri"]] + severity_map[inputs["structural"]]) / 2
    vec[0:10] = s_val

    # Cognitive (10-19)
    c_val = (severity_map[inputs["cognitive"]] + regression_map[inputs["regression"]]) / 2
    vec[10:20] = c_val

    # Epileptic (20-29)
    e_val = frequency_map[inputs["seizures"]]
    vec[20:30] = e_val

    # Genetic (30-39)
    g_val = genetic_map[inputs["genetic"]]
    vec[30:40] = g_val

    # Perinatal (40-49)
    p_val = (birth_map[inputs["birth"]] + motor_map[inputs["motor"]]) / 2
    vec[40:50] = p_val

    # ── Domain-specific baseline offsets (small, clinically motivated) ──
    if domain == "neurodegenerative":
        vec[10:20] += 0.10   # cognitive burden is primary
        vec[0:10]  += 0.05
    elif domain == "genetic_epileptic":
        vec[20:30] += 0.15   # seizure burden is primary
        vec[30:40] += 0.10
    elif domain == "neuroinflammatory":
        vec[0:20]  += 0.08
    elif domain == "metabolic":
        vec[10:20] += 0.08
        vec[0:10]  += 0.05
    elif domain == "vascular":
        vec[0:10]  += 0.12
        vec[10:20] += 0.06
    elif domain == "demyelinating":
        vec[10:20] += 0.08
        vec[0:10]  += 0.04

    # ── Final clip to valid normalised range ──────────────────────
    vec = np.clip(vec, 0.0, 1.0)
    return vec


def build_initial_sequence(feature_vec: np.ndarray, stage_index: int,
                            feature_dim: int) -> np.ndarray:
    """
    Build a realistic initial sequence of shape [stage_index+1, feature_dim].

    Instead of filling all prior stages with zeros, we populate each prior
    stage with a slightly attenuated version of the current feature vector.
    This gives the LSTM a meaningful developmental history to attend to,
    which is critical for non-flat outputs.

    Prior stages are attenuated by a decay factor: earlier = less severe
    (reflecting that the disease progresses over development).
    """
    n_stages = stage_index + 1
    sequence = np.zeros((n_stages, feature_dim), dtype=np.float32)

    for i in range(n_stages):
        # Linear attenuation: stage 0 = 40% of current severity,
        # current stage = 100%
        if n_stages == 1:
            attenuation = 1.0
        else:
            attenuation = 0.40 + 0.60 * (i / (n_stages - 1))
        sequence[i] = feature_vec * attenuation

    return sequence


# ──────────────────────────────────────────────────────────────────────────────
# Risk Computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_weighted_risk(sequence: np.ndarray, domain: str) -> np.ndarray:
    """
    Compute domain-weighted composite neurological risk index per stage.
    Paper Eq. 28: R_t = ws·S_t + wc·C_t + we·E_t

    Args:
        sequence : np.ndarray [T, features]
        domain   : disease domain string

    Returns:
        risk : np.ndarray [T]
    """
    ws, wc, we = DOMAIN_WEIGHTS.get(domain, (0.33, 0.34, 0.33))

    S = sequence[:, 0:10].mean(axis=1)
    C = sequence[:, 10:20].mean(axis=1)
    E = sequence[:, 20:30].mean(axis=1)

    return ws * S + wc * C + we * E


def detect_trend(mean_risk: np.ndarray) -> str:
    """
    Classify trajectory as rising / stable / declining using linear regression.
    Paper Eq. 29.

    Calibrated thresholds: use 0.005 per stage (empirically validated range for
    normalised risk scores in [0, 1]).
    """
    slope = float(np.polyfit(range(len(mean_risk)), mean_risk, 1)[0])
    if slope > 0.005:
        return "rising"
    elif slope < -0.005:
        return "declining"
    else:
        return "stable"


def uncertainty_width(risk_trajectories: np.ndarray) -> np.ndarray:
    """
    Predictive uncertainty width per stage (paper Eq. 33):
    U_t = P75(R_t) - P25(R_t)
    """
    return np.percentile(risk_trajectories, 75, axis=0) - \
           np.percentile(risk_trajectories, 25, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pediatric Neuro Risk Simulator",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Pediatric Neurological Risk Projection Simulator")
st.caption(
    "A probabilistic trajectory modeling tool for rare pediatric neurological diseases. "
    "Outputs are for research and educational purposes only — not for clinical diagnosis."
)

# ── Sidebar: Configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_simulations = st.slider("Monte Carlo Simulations", 20, 200, 50, step=10,
                               help="More simulations = smoother uncertainty bands but slower.")
    n_steps = st.slider("Future Stages to Project", 3, 8, 5,
                         help="Number of developmental stages ahead to simulate.")
    explanation_mode = st.radio(
        "Explanation Mode",
        ["Parent-Friendly", "Clinical Detail"],
        help="Controls the vocabulary and depth of the AI-generated narrative."
    )

# ── Main area: Stage & Clinical Inputs ───────────────────────────────────
st.subheader("📅 Current Developmental Stage")
selected_stage = st.selectbox(
    "Select the child's current developmental stage",
    list(STAGE_OPTIONS.keys()),
    key="stage_selector"
)
stage_index = STAGE_OPTIONS[selected_stage]

st.subheader("🩺 Clinical Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Seizure Profile**")
    seizures = st.selectbox(
        "Seizure Frequency",
        ["None", "Occasional (< monthly)", "Monthly", "Weekly", "Daily"],
        key="seizure_selector",
        help="Frequency of clinically confirmed seizure events."
    )

    st.markdown("**Structural / Neuroimaging**")
    structural = st.selectbox(
        "Structural Brain Abnormality (clinical assessment)",
        ["None", "Mild", "Moderate", "Severe"],
        key="structural_selector"
    )
    mri = st.selectbox(
        "MRI Finding",
        [
            "Normal",
            "Non-specific white matter changes",
            "Cortical atrophy / volume loss",
            "Structural malformation",
            "Severe / progressive changes",
        ],
        key="mri_selector",
        help="Most recent neuroimaging result."
    )

with col2:
    st.markdown("**Cognitive & Developmental**")
    cognitive = st.selectbox(
        "Cognitive Delay Severity",
        ["None", "Mild", "Moderate", "Severe"],
        key="cognitive_selector"
    )
    regression = st.selectbox(
        "Developmental Regression",
        ["None", "Mild plateauing", "Clear regression", "Severe regression"],
        key="regression_selector",
        help="Loss of previously acquired developmental milestones."
    )

    st.markdown("**Motor Function**")
    motor = st.selectbox(
        "Motor Impairment",
        [
            "None",
            "Mild (walks with support)",
            "Moderate (wheelchair part-time)",
            "Severe (non-ambulant)",
        ],
        key="motor_selector"
    )

with col3:
    st.markdown("**Genetic & Perinatal**")
    genetic = st.selectbox(
        "Genetic / Genomic Risk",
        ["Low", "Moderate", "High"],
        key="genetic_selector",
        help="Based on genetic panel results or clinical probability."
    )
    birth = st.selectbox(
        "Perinatal / Birth Complications",
        ["None", "Mild", "Significant"],
        key="birth_selector"
    )

    st.markdown("**Condition (optional)**")
    disease_input = st.text_input(
        "Diagnosed or Suspected Condition",
        placeholder="e.g. Dravet syndrome, Rett syndrome, Batten disease",
        key="disease_input_box",
        help="Type the condition name. The system will classify it into a neurological domain automatically."
    )

# ── Run Simulation ─────────────────────────────────────────────────────────
run_button = st.button("▶ Run Simulation", type="primary", key="run_sim_button")

if run_button:
    with st.spinner("Loading model and data..."):
        try:
            model, dataset, device = load_model_and_dataset()
            classifier = load_classifier()
        except FileNotFoundError as e:
            st.error(
                f"Model or data file not found: {e}\n\n"
                "Please run `python src/training/train_trajectory_model.py` first."
            )
            st.stop()

    # ── Domain classification ─────────────────────────────────────
    domain, confidence = classifier.classify_with_confidence(disease_input)
    domain_label = DOMAIN_DISPLAY.get(domain, "Unknown")

    if disease_input and disease_input.strip():
        st.info(
            f"🔍 Classified domain: **{domain_label}** "
            f"(semantic confidence: {confidence:.2f})"
        )
    else:
        st.info(f"🔍 No condition entered — using default domain: **{domain_label}**")

    # ── Build clinical feature vector ─────────────────────────────
    inputs = {
        "seizures": seizures, "structural": structural, "mri": mri,
        "cognitive": cognitive, "regression": regression,
        "motor": motor, "genetic": genetic, "birth": birth,
    }
    feature_vec = map_inputs_to_features(dataset.feature_dim, domain, inputs)

    # ── Build realistic initial sequence ──────────────────────────
    initial_sequence = build_initial_sequence(feature_vec, stage_index, dataset.feature_dim)
    initial_tensor = torch.tensor(
        initial_sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)  # [1, stage_index+1, features]

    # ── Monte Carlo simulation ────────────────────────────────────
    with st.spinner(f"Running {n_simulations} Monte Carlo trajectories..."):
        futures = simulate_future(
            model, initial_tensor,
            domain=domain, steps=n_steps, simulations=n_simulations
        )

    # ── Compute risk trajectories ─────────────────────────────────
    risk_trajectories = np.array([
        compute_weighted_risk(f, domain) for f in futures
    ])  # [n_simulations, T]

    mean_risk = risk_trajectories.mean(axis=0)
    std_risk  = risk_trajectories.std(axis=0)
    p25       = np.percentile(risk_trajectories, 25, axis=0)
    p75       = np.percentile(risk_trajectories, 75, axis=0)
    iqr_width = p75 - p25

    trend = detect_trend(mean_risk)
    final_risk = float(mean_risk[-1])

    # ── Risk category ─────────────────────────────────────────────
    if final_risk < 0.20:
        category = "Low Risk"
        cat_color = "🟢"
    elif final_risk < 0.40:
        category = "Moderate Risk"
        cat_color = "🟡"
    else:
        category = "High Risk"
        cat_color = "🔴"

    # ── Build x-axis labels ───────────────────────────────────────
    n_total = mean_risk.shape[0]
    n_observed = stage_index + 1
    x_labels = []
    for i in range(n_total):
        if i < n_observed:
            x_labels.append(STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"Stage {i}")
        else:
            proj_stage = i - n_observed + 1
            x_labels.append(f"Proj. +{proj_stage}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_total)

    # Shade observed vs projected
    ax.axvspan(-0.5, n_observed - 0.5, alpha=0.05, color="steelblue", label="Observed stages")
    ax.axvspan(n_observed - 0.5, n_total - 0.5, alpha=0.05, color="orange", label="Projected stages")

    # Risk zone bands
    ax.axhspan(0.0, 0.20, alpha=0.08, color="green")
    ax.axhspan(0.20, 0.40, alpha=0.08, color="gold")
    ax.axhspan(0.40, 1.00, alpha=0.08, color="red")

    # Uncertainty envelopes
    ax.fill_between(x, mean_risk - std_risk, mean_risk + std_risk,
                    alpha=0.18, color="steelblue", label="±1 SD")
    ax.fill_between(x, p25, p75, alpha=0.25, color="steelblue", label="IQR (P25–P75)")

    # Mean trajectory line
    ax.plot(x, mean_risk, color="steelblue", linewidth=2.8,
            marker="o", markersize=5, label="Mean risk")

    # Vertical separator: observed vs projected
    ax.axvline(x=n_observed - 0.5, color="grey", linestyle="--", linewidth=1.2,
               label="Projection start")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
    ax.set_xlabel("Developmental Stage", fontsize=11)
    ax.set_ylabel("Domain-Weighted Risk Index", fontsize=11)
    ax.set_title(
        f"Neurological Risk Trajectory — {domain_label}\n"
        f"Trend: {trend.capitalize()}  |  Final Stage Risk: {final_risk:.3f}  |  {cat_color} {category}",
        fontsize=12
    )
    ax.set_ylim(max(0.0, mean_risk.min() - 0.05), min(1.0, mean_risk.max() + 0.08))
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

    # ── Metrics row ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Risk Score", f"{final_risk:.3f}")
    m2.metric("Risk Category", category)
    m3.metric("Trajectory Trend", trend.capitalize())
    m4.metric("Avg. Uncertainty (IQR)", f"{iqr_width.mean():.3f}")

    # ── Stage-by-stage risk table ─────────────────────────────────
    with st.expander("📊 Stage-by-Stage Risk Values"):
        import pandas as pd
        df_risk = pd.DataFrame({
            "Stage": x_labels,
            "Mean Risk": mean_risk.round(4),
            "P25": p25.round(4),
            "P75": p75.round(4),
            "IQR Width": iqr_width.round(4),
            "Type": ["Observed"] * n_observed + ["Projected"] * (n_total - n_observed),
        })
        st.dataframe(df_risk, use_container_width=True)

    # ── Generative AI interpretation ──────────────────────────────
    st.subheader(f"{cat_color} Risk Category: {category}")
    st.subheader("📋 Detailed Neurodevelopmental Interpretation")

    with st.spinner("Generating clinical interpretation..."):
        try:
            explanation = generate_detailed_explanation(
                disease_input=disease_input,
                domain=domain,
                category=category,
                mean_risk=mean_risk,
                selected_stage=selected_stage,
                explanation_mode=explanation_mode,
                trend=trend,
                variability=float(std_risk.mean()),
            )
            st.write(explanation)
        except ValueError as e:
            st.warning(f"⚠️ LLM explanation unavailable: {e}")
        except Exception as e:
            st.warning(f"⚠️ LLM explanation failed: {e}")
            st.write("**Risk index values by stage:**", mean_risk.tolist())

    st.caption(
        "⚠️ This tool provides probabilistic projections and does **not** constitute "
        "medical diagnosis. All clinical decisions must be made in consultation with "
        "a qualified pediatric neurologist."
    )