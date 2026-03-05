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
SAVE_DIR = "evaluation_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)


def evaluate_model():

    # Load dataset
    sequences, features = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)

    # Load model
    input_size = dataset.feature_dim
    model = TrajectoryLSTM(input_size=input_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Take one batch
    X, Y = dataset[10]
    X = X.unsqueeze(0).to(device)
    Y = Y.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        pred = model(X)

    pred = pred.cpu().numpy().flatten()
    real = Y.cpu().numpy().flatten()

    # --- Numerical metrics ---
    rmse = np.sqrt(np.mean((pred - real) ** 2))
    mae = np.mean(np.abs(pred - real))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # --- Clinical trajectory plot ---
    plt.figure(figsize=(12, 6))

    plt.plot(real[:20], label="Real", linewidth=2)
    plt.plot(pred[:20], label="Predicted", linestyle="--")

    plt.title("Trajectory Feature Comparison (sample)")
    plt.legend()

    # Save plot
    save_path = os.path.join(SAVE_DIR, "trajectory_sample.png")
    plt.savefig(save_path)

    # Show plot
    plt.show()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    evaluate_model()