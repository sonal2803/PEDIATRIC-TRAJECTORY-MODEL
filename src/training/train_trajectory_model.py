import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM


DATA_PATH = "data/Longitudinal_Master_Dataset.csv"


# Load dataset
sequences, features = prepare_dataset(DATA_PATH)
dataset = TrajectoryDataset(sequences)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model
input_size = dataset.feature_dim
model = TrajectoryLSTM(input_size=input_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        predictions = model(X)

        # --- Base loss ---
        mse_loss = criterion(predictions, Y)

        # --- Realism constraints ---
        smoothness_penalty = torch.mean((predictions[:, 1:] - predictions[:, :-1])**2)
        persistence_penalty = torch.mean(torch.relu(Y - predictions))
        stability_penalty = torch.mean(torch.abs(predictions))

        smoothness_penalty = torch.mean((predictions[:, 1:] - predictions[:, :-1])**2)
        loss = (
            mse_loss
            + 0.1 * smoothness_penalty
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# Ensure directory exists
os.makedirs("src/models", exist_ok=True)

save_path = "src/models/trained_trajectory_model.pt"
torch.save(model.state_dict(), "models/trained_trajectory_model.pt")
print(f"Model saved to {save_path}")