import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(TrajectoryLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x shape: [batch_size, time_steps, features]
        """

        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        predictions = self.fc(last_output)

        return predictions