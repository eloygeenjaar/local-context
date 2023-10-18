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

# Based on: https://proceedings.mlr.press/v151/tonekaboni22a/tonekaboni22a.pdf
class Simulation(Dataset):
    def __init__(self, data_type, seed=42):
        self.seed = seed
        super().__init__()
        self.class_dict = {
            0: (0.5, 0.05, 1.8, 40, -1.5),
            1: (0.5, -0.05, 1.8, 40, 1.5),
            2: (0.8, 0.05, 0.8, 20, -1.5),
            3: (0.8, -0.05, 0.8, 20, 1.5)
        }
        self.X, self.y = self.generate_data(self.class_dict, seed)
        indices = np.arange(self.X.shape[0])
        train_val_ix, test_ix = train_test_split(indices, test_size=0.2, random_state=seed)
        train_ix, valid_ix = train_test_split(train_val_ix, train_size=0.9, random_state=seed)
        if data_type == 'train':
            self.X = self.X[train_ix]
            self.y = self.y[train_ix]
        elif data_type == 'valid':
            self.X = self.X[valid_ix]
            self.y = self.y[valid_ix]
        elif data_type == 'test':
            self.X = self.X[test_ix]
            self.y = self.y[test_ix]
        self.X = torch.from_numpy(self.X).float().unsqueeze(-1)
        self.y = torch.from_numpy(self.y).long()

    @staticmethod
    def generate_data(class_dict, seed):
        t = np.linspace(0, 10, 100)
        rng = np.random.default_rng(seed)
        classes = rng.integers(0, 4, size=(500, ))
        X_ls = []
        for i in range(500):
            alpha, gamma, a, b, c = class_dict[classes[i]]
            x = alpha * (gamma * t + a * np.sin((b * t)/(2*np.pi)) + c)
            X_ls.append(x)
        X = np.stack(X_ls, axis=0)
        X = X + rng.normal(0, 0.1, size=(500, 100))
        return X, classes

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]

class AirQuality(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        self.normalization = normalization

        ((train_x, valid_x, test_x),
         (train_zl, valid_zl, test_zl),
         (train_zg, valid_zg, test_zg),
         (train_mask, valid_mask, test_mask)) = get_airquality(normalization, seed)

        if data_type == 'train':
            self.X = train_x
            self.local_y = train_zl
            self.global_y = train_zg
            self.mask = train_mask
        elif data_type == 'valid':
            self.X = valid_x
            self.local_y = valid_zl
            self.global_y = valid_zg
            self.mask = valid_mask
        elif data_type == 'test':
            self.X = test_x
            self.local_y = test_zl
            self.global_y = test_zg
            self.mask = test_mask

        self.X = np.pad(self.X, ((0, 0), (4, 4), (0, 0)), mode='constant', constant_values=0)
        self.mask = np.pad(self.mask, ((0, 0), (4, 4), (0, 0)), mode='constant', constant_values=0)
        self.X = torch.from_numpy(self.X).float()
        self.mask = torch.from_numpy(self.mask).bool()
        # Second column is month of the year
        self.global_y = self.global_y[:, 1].astype(np.int32)
        self.unique_months = np.unique(self.global_y)
        for (ix, month) in enumerate(self.unique_months):
            self.global_y[self.global_y == month] = ix
        self.local_y = torch.from_numpy(self.local_y).long()
        self.global_y = torch.from_numpy(self.global_y).long()
        
        
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, ix):
        if self.normalization == 'individual':
            x = self.X[ix]
            mask = x.std(0) != 0
            x[:, mask] -= x[:, mask].mean(0)
            x[:, mask] /= x[:, mask].std(0)
            x[:, ~mask] = 0.
        elif self.normalization == 'dataset':
            x = self.X[ix]
        return x, self.mask[ix], (self.local_y[ix], self.global_y[ix])

    @property
    def data_size(self):
        return self.X.size(-1)

    @property
    def num_classes(self):
        return self.unique_months.size(0)

    @property
    def window_size(self):
        return 10

    @property
    def mask_windows(self):
        return 10

    @property
    def learning_rate(self):
        return 0.001

class PhysioNet(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        self.normalization = normalization

        ((train_x, valid_x, test_x),
         (train_zl, valid_zl, test_zl),
         (train_zg, valid_zg, test_zg),
         (train_mask, valid_mask, test_mask)) = get_physionet(normalization, seed)

        if data_type == 'train':
            self.X = train_x.float()
            self.local_y = train_zl.long()
            self.global_y = train_zg.long()
            self.mask = train_mask.bool()
        elif data_type == 'valid':
            self.X = valid_x.float()
            self.local_y = valid_zl.long()
            self.global_y = valid_zg.long()
            self.mask = valid_mask.bool()
        else:
            self.X = test_x.float()
            self.local_y = test_zl.long()
            self.global_y = test_zg.long()
            self.mask = test_mask.bool()
        
        #TODO: mask
        self.X = F.pad(self.X, (0, 0, 1, 0), "constant", 0)
        self.mask = F.pad(self.mask, (0, 0, 1, 0), "constant", 0)
        # Fourth column is ICU type
        self.global_y = self.global_y[:, 3]
        self.icu_types = torch.unique(self.global_y)
        for (ix, icu_type) in enumerate(self.icu_types):
            self.global_y[self.global_y == icu_type] = ix

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, ix):
        if self.normalization == 'individual':
            x = self.X[ix]
            mask = x.std(0) != 0
            #mask = mask.unsqueeze(0).repeat(self.X.size(1), 1)
            #mask = mask & self.mask[ix]
            x[:, mask] -= x[:, mask].mean(0)
            x[:, mask] /= x[:, mask].std(0)
            x[:, ~mask] = 0.
        elif self.normalization == 'dataset':
            x = self.X[ix]
        return x, self.mask[ix], (self.local_y[ix], self.global_y[ix])

    @property
    def data_size(self):
        return self.X.size(-1)

    @property
    def num_classes(self):
        return self.icu_types.size(0)

    @property
    def window_size(self):
        return 4

    @property
    def mask_windows(self):
        return 3

    @property
    def learning_rate(self):
        return 0.001

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
        x = signal.clean(x, detrend=True,
            standardize='zscore_sample', t_r=2.0,
            low_pass=0.15, high_pass=0.01)
        x = torch.from_numpy(x).float()
        return x.view(x.size(0), -1).float(), self.mask, (0, self.df.loc[self.indices[ix], 'sz'])

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
        return 0.001