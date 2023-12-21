import torch
import numpy as np
import nibabel as nb
import lightning.pytorch as pl
from nilearn import signal
from lib.utils import get_icafbirn
from torch.utils.data import Dataset, DataLoader


comp_ix = [
    68, 52, 97, 98, 44,
    20, 55,
    2, 8, 1, 10, 26, 53, 65, 79, 71,
    15, 4, 61, 14, 11, 92, 19, 7, 76,
    67, 32, 42, 69, 60, 54, 62, 78, 83, 95, 87, 47, 80, 36, 66, 37, 82,
    31, 39, 22, 70, 16, 50, 93,
    12, 17, 3, 6]


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
                low_pass=0.15, high_pass=0.008)
            self.data.append(x)
        self.data = np.stack(self.data, axis=0)
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return self.num_subjects * self.num_windows

    def __getitem__(self, ix):
        # Obtain the subject index
        subj_ix = ix // self.num_windows
        # Select the data for the subject
        x = self.data[subj_ix]
        x = x.float()
        # Obtain the subject's diagnosis
        y = self.df.loc[self.indices[subj_ix], 'sz'] == 2
        # Obtain the temporal index
        temp_ix = ix % self.num_windows
        # Generate positive self-supervised samples
        # TODO: Add a way to also sample a window 2 steps away
        min_pos_ix = max(0, temp_ix - 1)
        max_pos_ix = min(self.num_windows-1, temp_ix+1)
        pos_ix = np.random.choice(np.array([min_pos_ix, max_pos_ix]))

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


class DataModule(pl.LightningDataModule):
    def __init__(self, config, dataset_type, batch_size=128):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = self.dataset_type(
            'train', self.config['seed'], self.config['window_size'], self.config['window_step'])
        self.valid_dataset = self.dataset_type(
            'valid', self.config['seed'], self.config['window_size'], self.config['window_step'])
        self.test_dataset = self.dataset_type(
            'test', self.config['seed'], self.config['window_size'], self.config['window_step'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=5, pin_memory=True,
                          batch_size=self.config["batch_size"], shuffle=True,
                          persistent_workers=True, prefetch_factor=5, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, num_workers=5, pin_memory=True,
                          batch_size=self.config["batch_size"], shuffle=True,
                          persistent_workers=True, prefetch_factor=5, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=5, pin_memory=True,
                          batch_size=self.config["batch_size"], shuffle=True,
                          persistent_workers=True, prefetch_factor=5, drop_last=True)