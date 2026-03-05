import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM

DATA_PATH = "data/longitudinal_master_dataset.csv"

# Load sequences
sequences, features = prepare_dataset(DATA_PATH)

dataset = TrajectoryDataset(sequences)

# Take small batch
batch = torch.stack([dataset[i] for i in range(8)])  # batch of 8

print("Batch shape:", batch.shape)

# Build model
input_size = batch.shape[2]
model = TrajectoryLSTM(input_size=input_size)

# Forward pass
output = model(batch)

print("Output shape:", output.shape)