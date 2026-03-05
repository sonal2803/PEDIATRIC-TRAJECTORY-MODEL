from preprocessing import prepare_dataset
from dataset_builder import TrajectoryDataset

DATA_PATH = "data/Longitudinal_Master_Dataset.csv"

sequences, features = prepare_dataset(DATA_PATH)

dataset = TrajectoryDataset(sequences)

print("Total training pairs:", len(dataset))

X, Y = dataset[0]

print("Input sequence shape:", X.shape)
print("Target shape:", Y.shape)