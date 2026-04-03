import torch
from torch.utils.data import Dataset
import numpy as np


class TrajectoryDataset(Dataset):
    """
    Builds supervised (X, Y) pairs from longitudinal patient sequences.

    For a sequence of T stages [s0, s1, ..., sT-1]:
        X = [s0, ..., s_{t-1}]  (past history, padded to max_len)
        Y = s_t                  (next stage to predict)

    This yields T-1 pairs per patient.
    """

    def __init__(self, sequences):
        self.X = []
        self.Y = []

        for seq in sequences:
            time_steps = seq.shape[0]
            for t in range(1, time_steps):
                past = seq[:t]          # shape [t, features]
                target = seq[t]         # shape [features]
                self.X.append(past)
                self.Y.append(target)

        self.max_len = max(x.shape[0] for x in self.X)
        self.feature_dim = self.X[0].shape[1]

        self.X = self._pad_sequences(self.X)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def _pad_sequences(self, sequences):
        """Left-pad sequences with zeros so all have length max_len."""
        padded = []
        for seq in sequences:
            pad_len = self.max_len - seq.shape[0]
            if pad_len > 0:
                padding = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
                seq = np.vstack([padding, seq])
            padded.append(seq.astype(np.float32))
        return torch.tensor(np.array(padded), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]