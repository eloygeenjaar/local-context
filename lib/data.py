import torch
import numpy as np
import nibabel as nb
from nilearn import signal
from lib.definitions import comp_ix
from torch.utils.data import Dataset, DataLoader
from lib.utils import get_icafbirn, normal_sampling



class ICAfBIRN(Dataset):
    def __init__(self, data_type: str, seed: int,
                 window_size: int, window_step: int):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        self.window_size = window_size
        self.window_step = window_step
        # This value is inherent to the dataset
        self.num_timesteps = 150

        # Load the ICA-fBIRN data using a helper function
        train_df, valid_df, test_df = get_icafbirn(42)

        # Use only the dataset corresponding to the data type
        if data_type == 'train':
            self.df = train_df.copy()
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()

        self.indices = self.df.index.values
        self.num_subjects = self.df.shape[0]

        # Formula to calculate the number of windows fit in a timeseries
        self.num_windows = (((self.num_timesteps - self.window_size)
                             // self.window_step) + 1)

        # Preprocessing the data
        self.data = []
        for (i, row) in self.df.iterrows():
            x = nb.load(row['path']).get_fdata()[:150, comp_ix]
            x = signal.clean(
                x, detrend=True,
                standardize='zscore_sample', t_r=2.0,
                low_pass=0.20, high_pass=0.008)
            self.data.append(x)
        self.data = np.stack(self.data, axis=0)
        self.data = torch.from_numpy(self.data)

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_subjects * self.num_windows

    def __getitem__(self, ix):
        # Obtain the subject index
        subj_ix = ix // self.num_windows
        # Select the data for the subject
        x = self.data[subj_ix]
        x = x.float()
        # Obtain the subject's diagnosis
        # True is schizophrenia
        y = self.df.loc[self.indices[subj_ix], 'sz'] == 1
        # Obtain the temporal index
        temp_ix = ix % self.num_windows
        # Generate positive self-supervised samples
        pos_ix = normal_sampling(
            self.rng, current_ix=temp_ix, length=self.num_windows,
            sd=2.0)
        pos_ix = temp_ix
        start_window = temp_ix * self.window_step
        end_window = start_window + self.window_size
        # The positive self-supervised sample window
        # (positive samples are defined as windows close to this window)
        start_window_pos = pos_ix * self.window_step
        end_window_pos = start_window_pos + self.window_size
        return (
            x[start_window:end_window],
            x[start_window_pos:end_window_pos],
            (subj_ix, temp_ix),
            y)


class Simulation(Dataset):
    def __init__(self, data_type: str, seed: int,
                 window_size: int, window_step: int = 20):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        self.window_size = window_size
        self.window_step = window_step
        self.data = torch.from_numpy(
            np.load(f'data/simulated/x_{data_type}.npy')).float()
        self.latents = torch.from_numpy(
            np.load(f'data/simulated/l_{data_type}.npy')).float()
        self.states = torch.from_numpy(
            np.load(f'data/simulated/s_{data_type}.npy')).float()
        self.targets = torch.from_numpy(
            np.load(f'data/simulated/y_{data_type}.npy')).long()

    def __len__(self):
        return self.data.shape[0] * (
            self.data.shape[1] // self.window_size)

    def __getitem__(self, ix):
        # Obtain the subject index
        subj_ix = ix // (self.data.shape[1] // self.window_size)
        # Obtain the temporal index
        temp_ix = ix % (self.data.shape[1] // self.window_size)
        start_window = temp_ix * self.window_step
        end_window = start_window + self.window_size
        return (
            self.data[subj_ix, start_window:end_window],
            self.data[subj_ix, start_window:end_window],
            (subj_ix, temp_ix),
            self.targets[subj_ix])


class Test(Dataset):
    def __init__(self, data_type: str, seed: int,
                 window_size: int, window_step: int = 20):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        self.window_size = window_size
        self.window_step = window_step
        self.data = torch.randn((50, 100, len(comp_ix)))
        self.targets = torch.randint(low=0, high=2, size=(50, ))

    def __len__(self):
        return self.data.shape[0] * (
            self.data.shape[1] // self.window_size)

    def __getitem__(self, ix):
        # Obtain the subject index
        subj_ix = ix // (self.data.shape[1] // self.window_size)
        # Obtain the temporal index
        temp_ix = ix % (self.data.shape[1] // self.window_size)
        start_window = temp_ix * self.window_step
        end_window = start_window + self.window_size
        return (
            self.data[subj_ix, start_window:end_window],
            self.data[subj_ix, start_window:end_window],
            (subj_ix, temp_ix),
            self.targets[subj_ix])
