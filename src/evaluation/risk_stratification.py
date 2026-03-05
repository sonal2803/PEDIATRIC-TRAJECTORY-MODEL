import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM


DATA_PATH = "data/Longitudinal_Master_Dataset.csv"
MODEL_PATH = "src/models/trained_trajectory_model.pt"
SAVE_DIR = "risk_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def composite_risk_score(feature_vector):
    """
    Weighted neurological risk index.
    Adjust weights as schema matures.
    """
    total_features = len(feature_vector)

    # Divide features into 3 rough groups
    group1 = feature_vector[:total_features//3]          # structural
    group2 = feature_vector[total_features//3:2*total_features//3]  # cognitive
    group3 = feature_vector[2*total_features//3:]        # executive/other

    score = (
        0.3 * np.mean(group1) +
        0.5 * np.mean(group2) +
        0.2 * np.mean(group3)
    )

    return score


def simulate_future(model, initial_sequence, steps=5, simulations=50):

    model.train()  # enable dropout

    futures = []

    for s in range(simulations):
        seq = initial_sequence.clone()

        for _ in range(steps):
            with torch.no_grad():
                pred = model(seq.unsqueeze(0)).squeeze(0)

            # Progressive uncertainty
            time_index = seq.shape[0]
            noise_scale = 0.01 * time_index
            noise = torch.randn_like(pred) * noise_scale
            pred = pred + noise

            seq = torch.cat([seq, pred.unsqueeze(0)], dim=0)

        futures.append(seq.cpu().numpy())

    return futures


def run_risk_analysis():

    sequences, features = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = dataset.feature_dim
    model = TrajectoryLSTM(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Use one neonatal example
    first_sequence = sequences[0]
    initial_stage = torch.tensor(first_sequence[:1], dtype=torch.float32).to(device)

    futures = simulate_future(model, initial_stage, steps=5, simulations=50)

    risk_trajectories = []

    for future in futures:
        risk_path = []
        for stage in future:
            risk_path.append(composite_risk_score(stage))
        risk_trajectories.append(risk_path)

    risk_trajectories = np.array(risk_trajectories)

    mean_risk = np.mean(risk_trajectories, axis=0)
    std_risk = np.std(risk_trajectories, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_risk, label="Mean Risk", linewidth=3)
    plt.fill_between(
        range(len(mean_risk)),
        mean_risk - std_risk,
        mean_risk + std_risk,
        alpha=0.3,
        label="Uncertainty Envelope"
    )

    plt.title("Composite Neurological Risk Trajectory")
    plt.xlabel("Development Stage")
    plt.ylabel("Risk Index")
    plt.legend()

    save_path = os.path.join(SAVE_DIR, "risk_trajectory.png")
    plt.savefig(save_path)
    plt.show()

    # Severe outcome probability
    threshold = np.percentile(risk_trajectories[:, -1], 75)
    severe_prob = np.mean(risk_trajectories[:, -1] > threshold)

    print(f"Severe Outcome Probability: {severe_prob:.2f}")


if __name__ == "__main__":
    run_risk_analysis()
    