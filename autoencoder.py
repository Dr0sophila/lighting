import pickle
import numpy as np
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import util


class WaveAutoencoder(nn.Module):
    def __init__(self, input_dim=251, encoded_dim=64, lr=0.001):
        """
        Autoencoder for wave data.
        """
        super(WaveAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoded_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Loss & Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def _initialize_weights(self):
        """Initializes model weights using Xavier initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    import numpy as np

    def evaluate_pearson_batch(self, np_data):
        """
        Compute Pearson Correlation Coefficient (PCC) for a batch of wave data efficiently.
        """
        self.eval()
        tensor_data = torch.tensor(np.array(np_data), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            reconstructed, _ = self(tensor_data)

        original_np = tensor_data.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        # Compute Pearson correlation for all rows at once using NumPy
        mean_orig = np.mean(original_np, axis=1, keepdims=True)
        mean_recon = np.mean(reconstructed_np, axis=1, keepdims=True)

        numerator = np.sum((original_np - mean_orig) * (reconstructed_np - mean_recon), axis=1)
        denominator = np.sqrt(
            np.sum((original_np - mean_orig) ** 2, axis=1) * np.sum((reconstructed_np - mean_recon) ** 2, axis=1)
        )

        # Handle division by zero by setting small denominators to a safe value
        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
        pcc_values = numerator / denominator

        # Ensure no NaNs (replace with 0.0 if needed)
        return np.nan_to_num(pcc_values, nan=0.0)

    def train_model(self, np_data, zero_data, batch_size=10, epochs=20):
        """
        Train the autoencoder and calculate AUC per epoch efficiently.
        """
        tensor_data = torch.tensor(np.array(np_data), dtype=torch.float32).to(self.device)

        # Create DataLoader
        dataset = TensorDataset(tensor_data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # AUC tracking
        auc_per_epoch = []

        # Convert zero_data once to avoid repeated conversions
        zero_data = np.array(zero_data)

        # Training loop
        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch in data_loader:
                batch = batch[0]
                self.optimizer.zero_grad()
                reconstructed, _ = self(batch)
                loss = self.criterion(reconstructed, batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

            # Evaluate Pearson correlation in batches
            real_pccs = self.evaluate_pearson_batch(np_data)
            zero_pccs = self.evaluate_pearson_batch(zero_data)

            # Labels: 1 for real data, 0 for zero data
            y_true = np.concatenate([np.ones_like(real_pccs), np.zeros_like(zero_pccs)])
            y_scores = np.concatenate([real_pccs, zero_pccs])

            # Compute AUC
            auc = roc_auc_score(y_true, y_scores)
            auc_per_epoch.append(auc)
            print(f"Epoch [{epoch + 1}/{epochs}] - AUC: {auc:.6f}")

        return auc_per_epoch

def single_data_test(data_path, zero_path, enc_dim=2, batch_size=10, epochs=20):
    with open(data_path, "rb") as f:
        tower_id_waves_map = pickle.load(f)

    # Load all waves (positive samples)
    waves, _ = util.get_all_data(tower_id_waves_map)

    # Load zero dataset (negative samples)
    zeros = np.load(zero_path)  # Assume zero.npy is already loaded

    num_waves = len(waves)

    # AUC matrix: (num_waves x epochs)
    auc_matrix = np.zeros((num_waves, epochs))

    for i, single_wave in enumerate(waves):
        copy_waves = np.tile(single_wave, (100, 1))

        # Initialize and train autoencoder
        autoencoder = WaveAutoencoder(encoded_dim=enc_dim)
        auc_per_epoch = autoencoder.train_model(copy_waves, zeros, batch_size=batch_size, epochs=epochs)

        # Store AUC values for the current wave
        auc_matrix[i, :] = auc_per_epoch

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(auc_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label("AUC Score")

    # Set axis labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Wave Index")
    ax.set_xticks(range(0, epochs, max(1, epochs // 10)))  # Reduce tick density if too many epochs
    ax.set_xticklabels(range(1, epochs + 1, max(1, epochs // 10)))
    ax.set_yticks(range(0, num_waves, max(1, num_waves // 10)))  # Reduce tick density if too many waves
    ax.set_yticklabels(range(1, num_waves + 1, max(1, num_waves // 10)))

    ax.set_title("AUC Heatmap Over Epochs and Waves")
    plt.show()

    return auc_matrix


def vis_ae(data, enc_dim=2):
    with open(data, "rb") as f:
        tower_id_waves_map = pickle.load(f)

    autoencoder = WaveAutoencoder(encoded_dim=enc_dim)
    waves, index = util.get_all_data(tower_id_waves_map)
    autoencoder.train_model(waves, batch_size=10, epochs=20)
    _, data = autoencoder(torch.tensor(waves).to("cuda"))
    util.plot_cluster_tsne(data.cpu().detach().numpy(), index)


def vis_2d_ae(data, enc_dim=2):
    with open(data, "rb") as f:
        tower_id_waves_map = pickle.load(f)

    autoencoder = WaveAutoencoder(encoded_dim=enc_dim)
    waves, index = util.get_all_data(tower_id_waves_map)
    autoencoder.train_model(waves, batch_size=10, epochs=20)
    _, data = autoencoder(torch.tensor(waves).to("cuda"))
    util.plot_2d_data(data.cpu().detach().numpy(), index)
