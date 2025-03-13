import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE


def plot_wave(wave):
    plt.figure(figsize=(10, 4))
    plt.plot(wave)
    plt.grid(True)
    plt.show(block=True)


def get_all_data(data):
    """
    Combines all waves from a map<tower_id, waves> into a single numpy array of numpy arrays,
    returns an array of their lengths.

    :param data: Dictionary mapping tower IDs to numpy arrays of waves.
    :return: (combined_data, lengths, tower_lengths)
             - combined_data: A single numpy array containing all wave arrays.
             - tower_lengths: Dictionary with {tower_id: number of waves}.
    """
    # Collect all individual numpy arrays
    combined_data = np.array([wave for waves in data.values() for wave in waves], dtype=np.float32)

    # Count the number of waves per tower
    tower_lengths = [len(waves) for tower_id, waves in data.items()]

    return combined_data, tower_lengths


def plot_cluster_tsne(data, group_index):
    """
    Compress high-dimensional wave data to 2D using t-SNE and plot it by group.

    :param data: A NumPy array of shape (n_samples, n_features), obtained from get_all_data.
    :param group_index: A NumPy array where each entry represents the count of data points for a group.
    """
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=32)
    tsne_result = tsne.fit_transform(data)

    # Create scatter plot
    plt.figure(figsize=(8, 6))

    # Compute cumulative indices for slicing
    cumulative_indices = np.cumsum(group_index)
    start_idx = 0

    # Plot each group separately
    for i, end_idx in enumerate(cumulative_indices):
        plt.scatter(tsne_result[start_idx:end_idx, 0], tsne_result[start_idx:end_idx, 1], label=f"Tower {i}", alpha=0.7)
        start_idx = end_idx

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization of Wave Data")
    plt.legend()
    plt.show()

def plot_2d_data(data, group_index):
    """
    Compress high-dimensional wave data to 2D using t-SNE and plot it by group.

    :param data: A NumPy array of shape (n_samples, n_features), obtained from get_all_data.
    :param group_index: A NumPy array where each entry represents the count of data points for a group.
    """


    # Create scatter plot
    plt.figure(figsize=(8, 6))

    # Compute cumulative indices for slicing
    cumulative_indices = np.cumsum(group_index)
    start_idx = 0

    # Plot each group separately
    for i, end_idx in enumerate(cumulative_indices):
        plt.scatter(data[start_idx:end_idx, 0], data[start_idx:end_idx, 1], label=f"Tower {i}", alpha=0.7)
        start_idx = end_idx
    #
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.title("t-SNE Visualization of Wave Data")
    plt.legend()
    plt.show()


# with open("./data/towerid_waves_map.pkl", "rb") as f:
#     towerid_waves_map = pickle.load(f)
#
# data, index = get_all_data(towerid_waves_map)
# print(data.shape)
# plot_cluster_tsne(data, index)

