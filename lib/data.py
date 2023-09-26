import os
import torch
import numpy as np
import pandas as pd
import nibabel as nb
from torch.nn import functional as F
from torch.utils.data import Dataset
from lib.data_utils import split_dataset, save_data
from lib.data_utils import create_if_not_exist_dataset

# Based on: https://proceedings.mlr.press/v151/tonekaboni22a/tonekaboni22a.pdf
class Simulation(Dataset):
    def __init__(self, seed=42):
        self.seed = seed
        super().__init__()
        self.class_dict = {
            1: (0.5, 0.05, 1.8, 40, -1.5),
            2: (0.5, -0.05, 1.8, 40, 1.5),
            3: (0.8, 0.05, 0.8, 20, -1.5),
            4: (0.8, -0.05, 0.8, 20, 1.5)
        }
        self.X, self.y = self.generate_data(self.class_dict, seed)
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y - 1).long()

    @staticmethod
    def generate_data(class_dict, seed):
        t = np.linspace(0, 2*np.pi, 100)
        rng = np.random.default_rng(seed)
        classes = rng.random.randint(1, 5, size=(500, ))
        X_ls = []
        for i in range(500):
            alpha, gamma, a, b, c = class_dict[classes[i]]
            x = alpha * (gamma * t + a * np.sin((b * t)/(2*np.pi)) + c)
            X_ls.append(x)
        X = np.stack(X_ls, axis=0)
        X = X + rng.random.normal(0, 0.1, size=(500, 100))
        return X, classes

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]
