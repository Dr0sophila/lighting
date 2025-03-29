import pickle

import numpy as np

import autoencoder
import util

if __name__ == "__main__":

    # with open("./data/towerid_waves_map.pkl", "rb") as f:
    #     tower_id_waves_map = pickle.load(f)
    #
    # autoencoder.vis_ae(data=np.load("./data/zero.npy"), enc_dim=16)
    # autoencoder.vis_2d_ae(data="./data/augment.pkl")
    autoencoder.single_data_test(data_path="./data/towerid_waves_map.pkl", zero_path="./data/zero.npy", enc_dim=16)
    # with open("./data/towerid_waves_map.pkl", "rb") as f:
    #     tower_id_waves_map = pickle.load(f)

    # Load all waves (positive samples)
    # waves, _ = util.get_all_data(tower_id_waves_map)
    # util.plot_wave(waves[36])
    # util.plot_wave(waves[50])
    # autoencoder = autoencoder.WaveAutoencoder(encoded_dim=16)
    # autoencoder.train_model(np.tile(waves[0], (100, 1)),np.load("./data/zero.npy"), batch_size=10, epochs=20)
    # rwave, _ = autoencoder(torch.tensor(waves).to("cuda"))
    # util.plot_wave(rwave[50].cpu().detach().numpy())


    # with open("./data/towerid_waves_map.pkl", "rb") as f:
    #     tower_id_waves_map = pickle.load(f)
    #
    # waves, _ = util.get_all_data(tower_id_waves_map)
    # util.plot_wave(waves[28])