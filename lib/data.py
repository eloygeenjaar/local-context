import torch
import numpy as np
import pandas as pd
import nibabel as nb
from tqdm import tqdm
from nilearn import signal
from torch.nn import functional as F
from torch.utils.data import Dataset
from lib.utils import get_airquality, get_physionet, get_fbirn, get_icaukbb, get_icafbirn
from sklearn.model_selection import train_test_split

comp_ix = [68, 52, 97, 98, 44,

            20, 55,

            2, 8, 1, 10, 26, 53, 65, 79, 71,

            15, 4, 61, 14, 11, 92, 19, 7, 76,

            67, 32, 42, 69, 60, 54, 62, 78, 83, 95, 87, 47, 80, 36, 66, 37, 82,

            31, 39, 22, 70, 16, 50, 93,

            12, 17, 3, 6]


class ICAfBIRN(Dataset):
    def __init__(self, data_type, seed):
        super().__init__()
        self.data_type = data_type
        self.seed = seed

        train_df, valid_df, test_df = get_icafbirn(42)

        if data_type == 'train':
            self.df = train_df.copy()
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()

        self.indices = self.df.index.values
        self.window_size = 20
        self.num_timesteps = 150
        self.step = 10
        self.num_windows = (((self.num_timesteps - self.window_size) // self.step) + 1)
        self.num_subjects = self.df.shape[0]
        self.data = []
        for (i, row) in self.df.iterrows():
            x = nb.load(row['path']).get_fdata()[:150, comp_ix]
            x = signal.clean(x, detrend=True,
            standardize='zscore_sample', t_r=2.0,
            low_pass=0.15, high_pass=0.008)
            self.data.append(x)
        self.data = np.stack(self.data, axis=0)
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return self.num_subjects * self.num_windows

    def __getitem__(self, ix):
        subj_ix = ix // self.num_windows
        x = self.data[subj_ix]
        x = x.float()
        y = self.df.loc[self.indices[subj_ix], 'sz'] == 2
        temp_ix = ix % self.num_windows
        min_pos_ix = max(0, temp_ix - 1, temp_ix - 2)
        max_pos_ix = min(self.num_windows-1, temp_ix+1, temp_ix+2)
        pos_ix = np.random.choice(np.array([min_pos_ix, max_pos_ix]))
        return (x[(temp_ix * self.step):((temp_ix * self.step) + self.window_size)],
                x[(pos_ix * self.step):((pos_ix * self.step) + self.window_size)], 
                (subj_ix, temp_ix),
                y)

    @property
    def data_size(self):
        return 53

    @property
    def learning_rate(self):
        return 0.001

class Simulation(Dataset):
    def __init__(self, data_type, seed):
        super().__init__()
        self.data = torch.from_numpy(np.load(f'data/simulated/x_{data_type}.npy')).float()
        self.latents = torch.from_numpy(np.load(f'data/simulated/l_{data_type}.npy')).float()
        self.states = torch.from_numpy(np.load(f'data/simulated/s_{data_type}.npy')).float()
        self.targets = torch.from_numpy(np.load(f'data/simulated/y_{data_type}.npy')).long()

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] // 20)

    def __getitem__(self, ix):
        subj_ix = ix // (self.data.shape[1] // 20)
        temp_ix = ix % (self.data.shape[1] // 20)
        return self.data[subj_ix, (temp_ix * 20):((temp_ix + 1) * 20)], (subj_ix, temp_ix), self.targets[subj_ix]

    @property
    def data_size(self):
        return 53

    @property
    def learning_rate(self):
        return 0.001
