"""
Evaluation script: computes MAE and RMSE on the held-out test set.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM

DATA_PATH  = "data/Longitudinal_Master_Dataset.csv"
MODEL_PATH = "models/trained_trajectory_model.pt"


def evaluate():
    sequences, _ = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)

    # Reproduce the same 80/20 split used during training
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    _, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrajectoryLSTM(input_size=dataset.feature_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    mae_total = 0.0
    mse_total = 0.0
    count = 0

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)
            mae_total += torch.abs(preds - Y).sum().item()
            mse_total += torch.pow(preds - Y, 2).sum().item()
            count += Y.numel()

    MAE  = mae_total / count
    RMSE = float(np.sqrt(mse_total / count))
    print(f"Test MAE:  {MAE:.6f}")
    print(f"Test RMSE: {RMSE:.6f}")
    return MAE, RMSE


if __name__ == "__main__":
    evaluate()