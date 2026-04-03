import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM
from evaluation.trajectory_simulator import simulate_future

DATA_PATH  = "data/Longitudinal_Master_Dataset.csv"
MODEL_PATH = "models/trained_trajectory_model.pt"
SAVE_DIR   = "risk_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def composite_risk_score(feature_vector: np.ndarray, domain: str = None) -> float:
    """
    Domain-weighted neurological risk index (paper Eq. 25-28).

    Feature partitions (0-indexed):
        Structural  : cols  0 –  9  → weight ws
        Cognitive   : cols 10 – 19  → weight wc
        Epileptic   : cols 20 – 29  → weight we

    Domain-specific weight tuples (ws, wc, we):
        genetic_epileptic    : (0.2, 0.3, 0.5)
        neurodegenerative    : (0.4, 0.4, 0.2)
        neuroinflammatory    : (0.35, 0.35, 0.3)
        structural           : (0.5, 0.3, 0.2)
        metabolic            : (0.3, 0.4, 0.3)
        vascular             : (0.45, 0.35, 0.2)
        demyelinating        : (0.3, 0.45, 0.25)
        default              : (0.33, 0.34, 0.33)
    """
    DOMAIN_WEIGHTS = {
        "genetic_epileptic":  (0.20, 0.30, 0.50),
        "neurodegenerative":  (0.40, 0.40, 0.20),
        "neuroinflammatory":  (0.35, 0.35, 0.30),
        "structural":         (0.50, 0.30, 0.20),
        "metabolic":          (0.30, 0.40, 0.30),
        "vascular":           (0.45, 0.35, 0.20),
        "demyelinating":      (0.30, 0.45, 0.25),
    }
    ws, wc, we = DOMAIN_WEIGHTS.get(domain, (0.33, 0.34, 0.33))

    S = float(np.mean(feature_vector[0:10]))
    C = float(np.mean(feature_vector[10:20]))
    E = float(np.mean(feature_vector[20:30]))

    return ws * S + wc * C + we * E


def run_risk_analysis(domain: str = None, patient_index: int = 0):
    sequences, _ = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryLSTM(input_size=dataset.feature_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    # Use the first full trajectory from the dataset
    first_sequence = sequences[patient_index]  # [num_stages, features]
    initial_tensor = torch.tensor(
        first_sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)   # [1, num_stages, features]

    futures = simulate_future(model, initial_tensor, domain=domain, steps=5, simulations=50)

    # Compute risk trajectory per simulation
    risk_trajectories = []
    for future in futures:  # future shape: [num_stages + steps, features]
        risk_path = [composite_risk_score(stage, domain) for stage in future]
        risk_trajectories.append(risk_path)

    risk_trajectories = np.array(risk_trajectories)  # [50, T]
    mean_risk = np.mean(risk_trajectories, axis=0)
    std_risk  = np.std(risk_trajectories, axis=0)

    # Plot
    x = np.arange(len(mean_risk))
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_risk, label="Mean Risk", linewidth=3, color="steelblue")
    plt.fill_between(
        x,
        mean_risk - std_risk,
        mean_risk + std_risk,
        alpha=0.30,
        label="±1 SD Uncertainty",
        color="steelblue"
    )
    plt.axhspan(0.0, 0.2, alpha=0.05, color="green")
    plt.axhspan(0.2, 0.4, alpha=0.05, color="orange")
    plt.axhspan(0.4, 1.0, alpha=0.05, color="red")
    plt.title(f"Composite Neurological Risk Trajectory — {domain or 'Unclassified'}")
    plt.xlabel("Developmental Stage")
    plt.ylabel("Risk Index")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "risk_trajectory.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

    # Severe outcome probability (top 25th percentile of final stage risk)
    threshold   = np.percentile(risk_trajectories[:, -1], 75)
    severe_prob = float(np.mean(risk_trajectories[:, -1] > threshold))
    print(f"Severe Outcome Probability: {severe_prob:.2f}")


if __name__ == "__main__":
    run_risk_analysis(domain="neurodegenerative", patient_index=0)