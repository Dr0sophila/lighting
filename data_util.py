import numpy as np
import pandas as pd
import pickle

def get_tower_map():
    file_path = "./Tower_Info_MetaData_10272024_Hanbo_Final.xlsx"
    xls = pd.ExcelFile(file_path)
    df = xls.parse('hanbo_cleared')

    # Load numpy arrays
    names = np.load("./data/names.npy", allow_pickle=True)
    X = np.load("./data/X.npy", allow_pickle=True)

    # Ensure the lengths match
    assert len(names) == len(X), "Mismatch between names and X data length."

    # Convert names into a DataFrame
    names_df = pd.DataFrame({'Name': names})

    # Merge with tower information
    merged_df = names_df.merge(df[['Name', 'Tower ID']], on='Name', how='left')

    # Group by 'Tower ID' and collect corresponding X values
    towerid_waves_map = merged_df.groupby('Tower ID').apply(lambda g: X[g.index].tolist()).to_dict()
    converted_dict = {
        key: np.array([np.array(sublist, dtype=np.float32) for sublist in value], dtype=np.float32)
        for key, value in towerid_waves_map.items()
    }

    # Print to verify

        # Save the mapping as a pickle file
    output_path = "./data/towerid_waves_map.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(converted_dict, f)

def augment(save=False):
    with open("./data/towerid_waves_map.pkl", "rb") as f:
        towerid_waves_map = pickle.load(f)

    # Augment the data by repeating each wave 100 times
    augmented_data = {
        tower_id: np.tile(waves, (100, 1))  # Repeat each wave 100 times
        for tower_id, waves in towerid_waves_map.items()
    }

    if save:
        with open("./data/augment.pkl", "wb") as f:
            pickle.dump(augmented_data, f)
    return augmented_data
# augment()

def augment(save=False):
    with open("./data/towerid_waves_map.pkl", "rb") as f:
        towerid_waves_map = pickle.load(f)

    # Augment the data by repeating each wave 100 times
    augmented_data = {
        tower_id: np.tile(waves, (100, 1))  # Repeat each wave 100 times
        for tower_id, waves in towerid_waves_map.items()
    }

    if save:
        with open("./data/augment.pkl", "wb") as f:
            pickle.dump(augmented_data, f)
    return augmented_data