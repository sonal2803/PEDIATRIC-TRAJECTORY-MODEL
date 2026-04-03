"""
Training script for the pediatric neurological trajectory LSTM.

Loss function follows the paper (Eq. 12):
    L = L_MSE + λ1·L_smooth + λ2·L_persistence + λ3·L_stability

where:
    L_MSE         = ||x_{t+1} - x̂_{t+1}||²   (prediction accuracy)
    L_smooth      = ||x̂_{t+1} - x̂_t||²        (temporal smoothness)
    L_persistence = max(0, x_{t+1} - x̂_{t+1}) (no spontaneous regression)
    L_stability   = ||x̂_{t+1}||₁              (stability regularization)

Hyperparameters per paper: λ1=0.3, λ2=0.3, λ3=0.1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset
from models.baseline_lstm import TrajectoryLSTM

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATA_PATH = "data/Longitudinal_Master_Dataset.csv"
SAVE_PATH = "models/trained_trajectory_model.pt"

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10

# Biological regularization hyperparameters (from paper)
LAMBDA_SMOOTH = 0.3
LAMBDA_PERSISTENCE = 0.3
LAMBDA_STABILITY = 0.1


def biologically_constrained_loss(predictions, targets, prev_predictions=None):
    """
    Compute the composite biologically constrained loss (Eq. 12).

    Args:
        predictions  : model output x̂_{t+1}, shape [batch, features]
        targets      : ground truth x_{t+1}, shape [batch, features]
        prev_predictions: x̂_t from previous step (for smoothness), or None

    Returns:
        total_loss, dict of individual loss components
    """
    # L_MSE – prediction accuracy
    mse_loss = torch.mean((targets - predictions) ** 2)

    # L_smooth – penalise abrupt changes between consecutive predictions
    # When prev_predictions is unavailable (first batch step), use targets as proxy
    if prev_predictions is not None:
        smooth_loss = torch.mean((predictions - prev_predictions) ** 2)
    else:
        smooth_loss = torch.mean((predictions - targets) ** 2)

    # L_persistence – discourage spontaneous regression in progressive disorders
    # max(0, x_{t+1} - x̂_{t+1}): penalise when true value is higher than predicted
    # (i.e., model predicts improvement when disease actually worsens)
    persistence_loss = torch.mean(torch.relu(targets - predictions))

    # L_stability – L1 regularisation on outputs
    stability_loss = torch.mean(torch.abs(predictions))

    total = (
        mse_loss
        + LAMBDA_SMOOTH * smooth_loss
        + LAMBDA_PERSISTENCE * persistence_loss
        + LAMBDA_STABILITY * stability_loss
    )

    return total, {
        "mse": mse_loss.item(),
        "smooth": smooth_loss.item(),
        "persistence": persistence_loss.item(),
        "stability": stability_loss.item(),
    }


def train():
    # ── Load data ──────────────────────────────────────────────────
    sequences, features = prepare_dataset(DATA_PATH)
    dataset = TrajectoryDataset(sequences)

    # 80 / 20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ── Model ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TrajectoryLSTM(
        input_size=dataset.feature_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            prev_preds = None
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss, components = biologically_constrained_loss(preds, Y, prev_preds)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            prev_preds = preds.detach()

        avg_train = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                preds = model(X)
                loss, _ = biologically_constrained_loss(preds, Y)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f}"
        )

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Model saved to {SAVE_PATH}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()