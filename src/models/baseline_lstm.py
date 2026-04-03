import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    """
    Multi-layer LSTM for pediatric neurodevelopmental trajectory prediction.

    Architecture:
        - 2-layer LSTM with dropout
        - Fully connected projection head with ReLU + Sigmoid output
        - Sigmoid constrains all outputs to [0, 1] (normalized clinical range)

    Input:  [batch_size, time_steps, input_size]
    Output: [batch_size, input_size]  — next-stage feature prediction
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(TrajectoryLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_size),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, time_steps, input_size]
        Returns:
            predictions: Tensor of shape [batch_size, input_size]
        """
        lstm_out, _ = self.lstm(x)
        # Use only the last timestep's hidden state for prediction
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions