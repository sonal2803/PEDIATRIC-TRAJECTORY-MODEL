"""
Pediatric Neurological Risk Projection Simulator
Streamlit application — fixed version

Key fixes vs original:
  1. Feature mapping is properly clamped to [0, 1] before model input
  2. Initial sequence history is populated realistically at ALL observed stages
     (not just the last row) — avoids all-zero-history causing flat outputs
  3. Domain-weighted risk uses correct feature index ranges (paper §III.E)
  4. Trend detection uses a calibrated slope threshold
  5. Uncertainty width uses IQR (P75-P25) per paper Eq. 33
  6. UI improvements: domain display, confidence, richer clinical inputs
  7. Unnecessary compute blocks removed; proper error messages added
"""

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
    Build a realistic developmental history of shape [stage_index+1, feature_dim].

    Uses a sigmoid-shaped attenuation so that early stages start at ~15% of the
    current severity and ramp up smoothly to 100% at the current stage.

    A sigmoid curve (rather than linear) is more biologically realistic because:
      - Disease burden typically starts low at birth/neonatal
      - Accelerates through infancy/toddler as the condition manifests
      - Reaches the clinically apparent severity at the current stage

    This smooth ramp prevents the artificial spike-then-drop artifact that a
    linear 40→100% ramp creates at the observed/projected boundary.
    """
    n_stages = stage_index + 1
    sequence = np.zeros((n_stages, feature_dim), dtype=np.float32)

    for i in range(n_stages):
        if n_stages == 1:
            attenuation = 1.0
        else:
            # Sigmoid centred at the midpoint of the observed stages
            # Maps [0, n_stages-1] → roughly [0.15, 1.0]
            t = (i - (n_stages - 1) / 2.0) / max(1.0, (n_stages - 1) / 6.0)
            sig = 1.0 / (1.0 + np.exp(-t))
            # Re-scale so first stage ≈ 0.15 and last stage = 1.0
            sig_min = 1.0 / (1.0 + np.exp(-(-(n_stages - 1) / 2.0) /
                                            max(1.0, (n_stages - 1) / 6.0)))
            sig_max = 1.0 / (1.0 + np.exp(0.0))  # centre = 0.5 at midpoint, 1.0 at last
            # Normalise to [0.15, 1.0]
            attenuation = 0.15 + 0.85 * (sig - sig_min) / max(1e-6, 1.0 - sig_min)
            attenuation = float(np.clip(attenuation, 0.15, 1.0))

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


def compute_clinical_severity(inputs: dict) -> float:
    """
    Compute a normalised clinical severity index [0, 1] directly from user inputs.

    This is used as a scaling anchor for the risk display. Because the LSTM is
    trained on population data (majority low-severity), its sigmoid output is
    pulled toward the dataset mean (~0.15-0.30). The severity multiplier lifts
    the displayed risk scores to properly reflect the user's clinical inputs
    without altering the LSTM's numerically valid predictions.

    Returns a value in [0, 1] where:
        0.0  = all inputs at minimum (healthy)
        1.0  = all inputs at maximum (most severe presentation)
    """
    severity_map   = {"None": 0.0, "Mild": 0.30, "Moderate": 0.60, "Severe": 0.90}
    frequency_map  = {"None": 0.0, "Occasional (< monthly)": 0.20,
                      "Monthly": 0.45, "Weekly": 0.70, "Daily": 0.95}
    genetic_map    = {"Low": 0.05, "Moderate": 0.45, "High": 0.90}
    birth_map      = {"None": 0.0, "Mild": 0.30, "Significant": 0.70}
    regression_map = {"None": 0.0, "Mild plateauing": 0.30,
                      "Clear regression": 0.65, "Severe regression": 0.95}
    mri_map        = {
        "Normal": 0.0,
        "Non-specific white matter changes": 0.20,
        "Cortical atrophy / volume loss": 0.50,
        "Structural malformation": 0.70,
        "Severe / progressive changes": 0.95,
    }
    motor_map = {"None": 0.0, "Mild (walks with support)": 0.25,
                 "Moderate (wheelchair part-time)": 0.55,
                 "Severe (non-ambulant)": 0.90}

    components = [
        severity_map[inputs["structural"]],
        severity_map[inputs["cognitive"]],
        frequency_map[inputs["seizures"]],
        genetic_map[inputs["genetic"]],
        birth_map[inputs["birth"]],
        regression_map[inputs["regression"]],
        mri_map[inputs["mri"]],
        motor_map[inputs["motor"]],
    ]
    return float(np.mean(components))


def apply_severity_scaling(risk_array: np.ndarray, severity: float,
                            n_observed: int) -> np.ndarray:
    """
    Scale the raw LSTM-derived risk trajectory to reflect the user-entered severity.

    The LSTM output is pulled toward the training distribution mean (low risk).
    We compute a target floor and ceiling based on the user's clinical inputs and
    linearly rescale the PROJECTED portion of the trajectory into that range.

    The OBSERVED stages are left unchanged (they reflect the LSTM's own assessment
    of the input history). Only the projected stages are re-anchored.

    severity=0.0 → no scaling (healthy child, risk stays low)
    severity=0.5 → moderate rescaling
    severity=1.0 → strong rescaling (high-severity, risk should be visibly elevated)

    The minimum output floor for projected stages:
        floor = 0.05 + 0.30 * severity          (0.05 for healthy → 0.35 for severe)
    The ceiling:
        ceiling = floor + (1 - floor) * 0.6 * severity  (expands range proportionally)
    """
    if severity < 0.05:
        return risk_array  # healthy — no adjustment needed

    scaled = risk_array.copy()
    proj = scaled[n_observed:]
    if len(proj) == 0:
        return scaled

    floor   = 0.05 + 0.30 * severity
    ceiling = floor + (1.0 - floor) * 0.65 * severity

    r_min = float(proj.min())
    r_max = float(proj.max())
    r_range = max(r_max - r_min, 1e-6)

    # Rescale projected risk into [floor, ceiling]
    proj_scaled = floor + (proj - r_min) / r_range * (ceiling - floor)
    scaled[n_observed:] = proj_scaled
    return scaled


def detect_trend(mean_risk: np.ndarray, n_observed: int) -> str:
    """
    Classify trajectory as rising / stable / declining using linear regression
    on the PROJECTED portion only (paper Eq. 29).

    Using projected-only avoids the observed-stage ramp influencing the trend
    classification — clinicians care about where the disease is GOING, not
    what has already been observed.

    Calibrated threshold: 0.005 per stage for normalised [0,1] risk.
    """
    proj = mean_risk[n_observed:]
    if len(proj) < 2:
        proj = mean_risk  # fallback if only 1 projected stage
    slope = float(np.polyfit(range(len(proj)), proj, 1)[0])
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
    severity    = compute_clinical_severity(inputs)

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

    # ── Compute raw risk trajectories ─────────────────────────────
    n_observed = stage_index + 1
    raw_risk = np.array([
        compute_weighted_risk(f, domain) for f in futures
    ])  # [n_simulations, T]

    # ── Apply clinical severity scaling to projected stages ────────
    scaled_risk = np.array([
        apply_severity_scaling(r, severity, n_observed) for r in raw_risk
    ])

    mean_risk = scaled_risk.mean(axis=0)
    std_risk  = scaled_risk.std(axis=0)
    p25       = np.percentile(scaled_risk, 25, axis=0)
    p75       = np.percentile(scaled_risk, 75, axis=0)
    iqr_width = p75 - p25

    # Trend and uncertainty computed on PROJECTED stages only
    trend      = detect_trend(mean_risk, n_observed)
    final_risk = float(mean_risk[-1])
    proj_iqr   = iqr_width[n_observed:].mean() if n_steps > 0 else 0.0

    # ── Risk category ─────────────────────────────────────────────
    if final_risk < 0.20:
        category  = "Low Risk"
        cat_color = "🟢"
    elif final_risk < 0.40:
        category  = "Moderate Risk"
        cat_color = "🟡"
    else:
        category  = "High Risk"
        cat_color = "🔴"

    # ── Build x-axis labels ───────────────────────────────────────
    n_total  = mean_risk.shape[0]
    x_labels = []
    for i in range(n_total):
        if i < n_observed:
            x_labels.append(STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"Stage {i}")
        else:
            proj_stage = i - n_observed + 1
            x_labels.append(f"Proj. +{proj_stage}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x      = np.arange(n_total)
    x_obs  = x[:n_observed]
    x_proj = x[n_observed - 1:]  # overlap by 1 for visual continuity

    # Background zone shading
    ax.axhspan(0.0, 0.20, alpha=0.07, color="green",  zorder=0)
    ax.axhspan(0.20, 0.40, alpha=0.07, color="gold",  zorder=0)
    ax.axhspan(0.40, 1.00, alpha=0.07, color="tomato", zorder=0)

    # Observed vs projected region shading
    ax.axvspan(-0.5, n_observed - 0.5, alpha=0.04, color="steelblue", zorder=0)
    ax.axvspan(n_observed - 0.5, n_total - 0.5, alpha=0.04, color="darkorange", zorder=0)

    # Uncertainty envelopes — PROJECTED only (meaningful Monte Carlo spread)
    if n_steps > 0:
        xp = x[n_observed - 1:]   # include last observed for smooth join
        ax.fill_between(xp,
                        (mean_risk - std_risk)[n_observed - 1:],
                        (mean_risk + std_risk)[n_observed - 1:],
                        alpha=0.15, color="steelblue", label="±1 SD (projected)")
        ax.fill_between(xp,
                        p25[n_observed - 1:],
                        p75[n_observed - 1:],
                        alpha=0.22, color="steelblue", label="IQR P25–P75 (projected)")

    # Observed trajectory — solid dark markers (deterministic history)
    ax.plot(x_obs, mean_risk[:n_observed],
            color="steelblue", linewidth=2.5,
            marker="o", markersize=7,
            markerfacecolor="white", markeredgecolor="steelblue",
            markeredgewidth=2.0,
            label="Observed stages", zorder=5)

    # Projected trajectory — filled markers + dashed connector from last observed
    ax.plot(x[n_observed - 1:], mean_risk[n_observed - 1:],
            color="steelblue", linewidth=2.5, linestyle="--",
            marker="o", markersize=6,
            markerfacecolor="steelblue", markeredgecolor="steelblue",
            label="Projected mean risk", zorder=5)

    # Vertical separator
    ax.axvline(x=n_observed - 0.5, color="grey", linestyle=":", linewidth=1.4,
               label="Projection start", zorder=4)

    # Zone labels on right margin
    ax.text(n_total - 0.45, 0.10, "Low", fontsize=7.5, color="green",
            alpha=0.7, va="center")
    ax.text(n_total - 0.45, 0.30, "Mod", fontsize=7.5, color="goldenrod",
            alpha=0.7, va="center")
    ax.text(n_total - 0.45, 0.60, "High", fontsize=7.5, color="tomato",
            alpha=0.7, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
    ax.set_xlabel("Developmental Stage", fontsize=11)
    ax.set_ylabel("Domain-Weighted Risk Index", fontsize=11)

    y_lo = max(0.0, mean_risk.min() - 0.06)
    y_hi = min(1.0, mean_risk.max() + 0.10)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.5, n_total - 0.5)

    ax.set_title(
        f"Neurological Risk Trajectory — {domain_label}\n"
        f"Projected Trend: {trend.capitalize()}  |  "
        f"Final Risk: {final_risk:.3f}  |  {cat_color} {category}",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Metrics row ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Risk Score", f"{final_risk:.3f}")
    m2.metric("Risk Category", category)
    m3.metric("Projected Trend", trend.capitalize())
    m4.metric("Avg. Uncertainty (IQR)", f"{proj_iqr:.3f}",
              help="IQR computed over projected stages only")

    # ── Severity info box ──────────────────────────────────────────
    if severity > 0.5:
        sev_label = "High" if severity > 0.70 else "Moderate-High"
        st.info(
            f"🔬 Clinical severity index: **{severity:.2f}** ({sev_label}). "
            "Risk scores have been scaled to reflect your clinical inputs."
        )
    elif severity > 0.20:
        st.info(
            f"🔬 Clinical severity index: **{severity:.2f}** (Moderate). "
            "Risk scores reflect mild-to-moderate clinical burden."
        )

    # ── Stage-by-stage risk table ─────────────────────────────────
    with st.expander("📊 Stage-by-Stage Risk Values"):
        import pandas as pd
        iqr_display = []
        for i in range(n_total):
            if i < n_observed:
                iqr_display.append("—")   # deterministic; no MC spread
            else:
                iqr_display.append(f"{iqr_width[i]:.4f}")

        df_risk = pd.DataFrame({
            "Stage":     x_labels,
            "Mean Risk": mean_risk.round(4),
            "P25":       [f"{v:.4f}" if i >= n_observed else "—"
                          for i, v in enumerate(p25)],
            "P75":       [f"{v:.4f}" if i >= n_observed else "—"
                          for i, v in enumerate(p75)],
            "IQR Width": iqr_display,
            "Type":      ["Observed"] * n_observed + ["Projected"] * n_steps,
        })
        st.dataframe(df_risk, use_container_width=True)
        st.caption(
            "P25, P75, and IQR are not applicable (—) for observed stages "
            "because they are deterministic inputs, not probabilistic projections."
        )

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