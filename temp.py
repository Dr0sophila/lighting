import pickle
import numpy as np
import util

with open("./data/towerid_waves_map.pkl", "rb") as f:
    towerid_waves_map = pickle.load(f)


# print(towerid_waves_map["#1041193"])

# util.plot_wave(np.load("./data/X.npy")[4])
util.plot_wave(towerid_waves_map["#1041193"][5])