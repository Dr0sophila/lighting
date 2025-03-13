import autoencoder

if __name__ == "__main__":
    # autoencoder.vis_ae(data="./data/towerid_waves_map.pkl", enc_dim=16)
    # autoencoder.vis_2d_ae(data="./data/augment.pkl")
    autoencoder.single_data_test(data_path="./data/towerid_waves_map.pkl", zero_path="./data/zero.npy", enc_dim=16)
