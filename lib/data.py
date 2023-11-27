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


class fBIRN(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix):
        super().__init__()
        self.data_type = data_type
        self.seed = seed

        train_df, valid_df, test_df = get_fbirn(seed, fold_ix, num_folds)

        if data_type == 'train':
            self.df = train_df.copy()
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()
        self.indices = self.df.index.values

        data = []
        print(f'Loading {data_type} data')
        for (i, row) in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            x = np.transpose(nb.load(row['path']).get_fdata(), (3, 0, 1, 2))
            sd_mask = x.std(0) != 0
            x[:, sd_mask] -= x[:, sd_mask].mean(0)
            x[:, sd_mask] /= x[:, sd_mask].std(0)
            x_pad = np.pad(x, ((2, 1), (6, 5), (1, 0), (6, 6)), mode="constant", constant_values=0)
            data.append(x_pad.astype(np.float16))
        self.X = np.stack(data, axis=0)
        self.mask = F.pad(torch.ones(x.shape), (6, 6, 1, 0, 6, 5, 2, 1), "constant", 0).bool()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        x = torch.from_numpy(self.X[ix]).float()
        return x.view(x.size(0), -1).float(), self.mask, (torch.zeros((1, )), self.df.loc[self.indices[ix], 'diagnosis'])

    @property
    def data_size(self):
        return 128

    @property
    def num_classes(self):
        return 2

    @property
    def window_size(self):
        return 10

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

class ICAUKBiobank(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix):
        super().__init__()
        self.data_type = data_type
        self.seed = seed

        train_df, valid_df, test_df = get_icaukbb(seed)

        if data_type == 'train':
            self.df = train_df.copy()
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()

        self.indices = self.df.index.values
        self.mask = torch.ones((100, 53)).bool()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        x = nb.load(self.df.loc[self.indices[ix], 'path']).get_fdata()[:100, comp_ix]
        # TR from: https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf
        x = signal.clean(x, detrend=True,
            standardize='zscore_sample', t_r=0.735,
            low_pass=0.15, high_pass=0.01)
        x = torch.from_numpy(x).float()
        return x.view(x.size(0), -1).float(), self.mask, (0, self.df.loc[self.indices[ix], 'sex'])

    @property
    def data_size(self):
        return 53

    @property
    def num_classes(self):
        return 2

    @property
    def window_size(self):
        return 10

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

class ICAfBIRN(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix):
        super().__init__()
        self.data_type = data_type
        self.seed = seed

        train_df, valid_df, test_df = get_icafbirn(seed)

        if data_type == 'train':
            self.df = train_df.copy()
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()

        self.indices = self.df.index.values
        self.mask = torch.ones((150, 53)).bool()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        x = nb.load(self.df.loc[self.indices[ix], 'path']).get_fdata()[:150, comp_ix]
        # TR from: https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf
        #x = signal.clean(x, detrend=True,
        #    standardize='zscore_sample', t_r=2.0,
        #    low_pass=0.15, high_pass=0.01)
        x = signal.clean(x, detrend=True,
            standardize='zscore_sample', t_r=2.0,
            low_pass=None, high_pass=None)
        x = torch.from_numpy(x).float()
        return x.view(x.size(0), -1).float(), torch.Tensor([ix]).long(), (0, self.df.loc[self.indices[ix], 'sz'])

    @property
    def data_size(self):
        return 53

    @property
    def num_classes(self):
        return 2

    @property
    def window_size(self):
        return 20

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.0005

    @property
    def num_timesteps(self):
        return 150

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
        return self.data[subj_ix, (temp_ix * 20):((temp_ix + 1) * 20)], self.targets[subj_ix]

    @property
    def data_size(self):
        return 53

    @property
    def learning_rate(self):
        return 0.001
