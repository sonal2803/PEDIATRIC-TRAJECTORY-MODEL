import torch
from torch.utils.data import Dataset
import numpy as np


class TrajectoryDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: list of numpy arrays [time_steps, features]
        Build temporal training pairs:
            X = past sequence
            Y = next stage
        """

        self.X = []
        self.Y = []

        for seq in sequences:
            time_steps = seq.shape[0]

            for t in range(1, time_steps):
                past = seq[:t]
                target = seq[t]

                self.X.append(past)
                self.Y.append(target)

        self.max_len = max([x.shape[0] for x in self.X])
        self.feature_dim = self.X[0].shape[1]

        self.X = self._pad_sequences(self.X)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def _pad_sequences(self, sequences):
        padded = []

        for seq in sequences:
            pad_len = self.max_len - seq.shape[0]

            if pad_len > 0:
                padding = np.zeros((pad_len, self.feature_dim))
                seq = np.vstack((padding, seq))

            padded.append(seq)

        return torch.tensor(np.array(padded), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]