import torch
import numpy as np
from torch.utils.data import DataLoader

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM

DATA_PATH = "data/Longitudinal_Master_Dataset.csv"
MODEL_PATH = "models/trained_trajectory_model.pt"

# Load dataset
sequences, _ = prepare_dataset(DATA_PATH)
dataset = TrajectoryDataset(sequences)

test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrajectoryLSTM(input_size=dataset.feature_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

mae_total = 0
mse_total = 0
count = 0

with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        Y = Y.to(device)

        preds = model(X)

        mae_total += torch.abs(preds - Y).sum().item()
        mse_total += torch.pow(preds - Y, 2).sum().item()
        count += Y.numel()

MAE = mae_total / count
RMSE = np.sqrt(mse_total / count)

print(f"MAE: {MAE:.6f}")
print(f"RMSE: {RMSE:.6f}")