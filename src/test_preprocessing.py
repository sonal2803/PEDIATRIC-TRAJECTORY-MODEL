from preprocessing import prepare_dataset

DATA_PATH = "data/longitudinal_master_dataset.csv"

sequences, features = prepare_dataset(DATA_PATH)

print("Example sequence shape:", len(sequences[0]), "timesteps")