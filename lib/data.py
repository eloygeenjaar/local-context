import torch
import numpy as np
import pandas as pd
import nibabel as nb
from tqdm import tqdm
from nilearn import signal
from scipy.interpolate import CubicSpline
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch import distributions as D
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

    @property
    def num_timesteps(self):
        return 680

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
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix, window_step=10):
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
        self.TR = 2.0
        self.new_TR = 0.72
        x = np.arange(150) * self.TR
        new_x = np.arange(300 // self.new_TR) * self.new_TR
        data, masks = [], []
        #j = 1
        print(f'Loading {data_type} data')
        for (i, row) in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            data_subj = nb.load(row['path']).get_fdata()[:, :, 26, :150]
            data_subj = np.reshape(data_subj, (-1, 150))
            data_subj = np.transpose(data_subj, (1, 0))
            data_subj = signal.clean(data_subj, detrend=True,
            standardize='zscore_sample', t_r=self.TR,
            low_pass=0.10, high_pass=0.01)
            new_data = []
            for i in range(data_subj.shape[-1]):
                cs = CubicSpline(x, data_subj[:, i])
                new_data.append(cs(new_x))
            new_data = np.stack(new_data, axis=1)
            data.append(new_data)
        #    j += 1
        #    if j == 12:
        #        break
        self.data = np.stack(data, axis=0).astype(np.float16)
        self.window_size = 32 #int(self.window_size_sec // self.new_TR)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        data = self.data[ix]
        data = torch.from_numpy(data)
        return data, 0, 0, (0, 1)

    @property
    def data_size(self):
        return 3339

    @property
    def num_classes(self):
        return 2

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

    @property
    def num_timesteps(self):
        return self.data.shape[1]

class ICAUKBiobank(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix, window_step=30):
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
        data = []
        i = 0 
        for (ix, row) in self.df.iterrows():
            if i % 100 == 0:
                print(i / self.df.shape[0])
            data_subj = nb.load(row['path']).get_fdata(dtype=np.float32)[:100, comp_ix]
            data_subj = signal.clean(data_subj, detrend=True,
            standardize='zscore_sample', t_r=0.735,
            low_pass=0.10, high_pass=0.01)
            data.append(data_subj)
            i += 1
        self.data = torch.from_numpy(np.stack(data, axis=0)).float()
        # (subjects, num_windows, input_size, window_size)
        self.data = self.data.unfold(1, 30, window_step)
        num_subjects, num_windows, input_size, window_size = self.data.size()
        self.data = torch.reshape(self.data, (num_subjects * num_windows, input_size, window_size))
        self.mask = torch.ones_like(self.data).bool()
        print(self.data.size(0))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, ix):
        #x = nb.load(self.df.loc[self.indices[ix], 'path']).get_fdata()[:100, comp_ix]
        # TR from: https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf
        #x = signal.clean(x, detrend=True,
        #    standardize='zscore_sample', t_r=0.735,
        #    low_pass=0.1, high_pass=0.01)
        x = self.data[ix]
        return x, 0, self.mask[ix], (0, 1)#(0, self.df.loc[self.indices[ix], 'sz'])

    @property
    def data_size(self):
        return 53

    @property
    def num_classes(self):
        return 2

    @property
    def window_size(self):
        return 30
    
    @property
    def num_timesteps(self):
        return 100

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

class ICAfBIRN(Dataset):
    def __init__(self, data_type, normalization, seed, num_folds, fold_ix, window_step=20):
        super().__init__()
        self.data_type = data_type
        self.seed = seed

        train_df, valid_df, test_df = get_icafbirn(seed)

        if data_type == 'train':
            self.df = train_df.copy()
            self.dropout_rate = 0.05
            self.bern = D.Bernoulli(1 - self.dropout_rate)
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()
        self.indices = self.df.index.values
        self.TR = 2.0
        self.new_TR = 0.72
        x = np.arange(150) * self.TR
        new_x = np.arange(300 // self.new_TR) * self.new_TR
        data, masks = [], []
        for (ix, row) in self.df.iterrows():
            data_subj = nb.load(row['path']).get_fdata(dtype=np.float32)[:150, comp_ix]
            data_subj = signal.clean(data_subj, detrend=True,
            standardize='zscore_sample', t_r=self.TR,
            low_pass=0.10, high_pass=0.01)
            new_data = []
            for i in range(len(comp_ix)):
                cs = CubicSpline(x, data_subj[:, i])
                new_data.append(cs(new_x))
            new_data = np.stack(new_data, axis=1)
            data.append(data_subj)
        self.data = np.stack(data, axis=0).astype(np.float16)
        self.window_size = 32 #int(self.window_size_sec // self.new_TR)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        data = self.data[ix]
        data = torch.from_numpy(data)
        return data, 0, 0, (0, self.df.loc[self.indices[ix], 'sz'] == 2)

    @property
    def data_size(self):
        return 53

    @property
    def num_classes(self):
        return 2

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

    @property
    def num_timesteps(self):
        return self.data.shape[1]

class HCPLeft(Dataset):
    def __init__(self, data_type, *args, **kwargs):
        self.data_type = data_type
        self.df = pd.read_csv(f'/data/users1/egeenjaar/transformed-spaces/left.csv', index_col=0)
        if self.data_type == 'train':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.7)].copy()
        elif self.data_type == 'valid':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.7):int(self.df.shape[0] * 0.8)].copy()
        elif self.data_type == 'test':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.8):].copy()
        elif self.data_type == 'train_valid':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.8)].copy()
        self.indices = self.df.index.values
        self.behavioral = pd.read_csv('behavioral.csv', index_col=0)
        self.df = self.df.join(self.behavioral, how='left')
        to_replace = list(self.df['Age'].unique())
        value = np.arange(len(to_replace))
        d = {tr: v for tr, v in zip(to_replace, value)}
        self.df['Age'] = self.df['Age'].replace(d)
        data, masks = [], []
        for (i, row) in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            data_subj = np.load(row['fmri']).astype(np.float16)
            data.append(data_subj)
            masks.append(np.load(row['targets']).astype(int))
        
        self.window_size_sec = 20.
        self.window_size = 32 #int(self.window_size_sec // self.new_TR)
        self.window_step = 10 #int(window_step // self.new_TR)
        self.data = np.stack(data, axis=0)
        self.mask = np.stack(masks, axis=0)
        #self.data = torch.from_numpy(np.stack(data, axis=0)).float()
        #self.data = self.data.unfold(1, self.window_size, self.window_step)
        #num_subjects, num_windows, input_size, window_size = self.data.size()
        #self.data = torch.reshape(self.data, (num_subjects * num_windows, input_size, window_size))
        #self.masks = torch.ones_like(self.data).bool()

    @property
    def data_size(self):
        return 91282 

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

    @property
    def num_timesteps(self):
        return 284

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        data = self.data[ix]
        task_mask = self.mask[ix]
        data = torch.from_numpy(data)
        task_mask = torch.from_numpy(task_mask)
       # data = data.view(-1, 91282, self.window_size).permute(0, 2, 1)
        return data, task_mask, 0, (0, self.df.loc[self.indices[ix], 'Age'])#(0, self.df.loc[self.indices[ix], 'sz'])

class MotorGlasser(Dataset):
    def __init__(self, data_type, *args, **kwargs):
        self.data_type = data_type
        self.df = pd.read_csv(f'/data/users1/egeenjaar/local-global/motor_glasser.csv', index_col=0)
        if self.data_type == 'train':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.7)].copy()
        elif self.data_type == 'valid':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.7):int(self.df.shape[0] * 0.8)].copy()
        elif self.data_type == 'test':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.8):].copy()
        elif self.data_type == 'train_valid':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.8)].copy()
        self.indices = self.df.index.values
        self.behavioral = pd.read_csv('behavioral.csv', index_col=0)
        self.df = self.df.join(self.behavioral, how='left')
        to_replace = list(self.df['Age'].unique())
        value = np.arange(len(to_replace))
        d = {tr: v for tr, v in zip(to_replace, value)}
        self.df['Age'] = self.df['Age'].replace(d)
        data, masks = [], []
        for (i, row) in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            data_subj = np.load(row['fmri']).astype(np.float16)
            data.append(data_subj)
        
        self.window_size_sec = 20.
        self.window_size = 16 #int(self.window_size_sec // self.new_TR)
        self.window_step = 8 #int(window_step // self.new_TR)
        self.data = np.stack(data, axis=0)

    @property
    def data_size(self):
        return 379

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

    @property
    def num_timesteps(self):
        return 284

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        data = self.data[ix]
        data = torch.from_numpy(data)
       # data = data.view(-1, 91282, self.window_size).permute(0, 2, 1)
        return data, 0, 0, (0, self.df.loc[self.indices[ix], 'Age'])#(0, self.df.loc[self.indices[ix], 'sz'])

class WMGlasser(Dataset):
    def __init__(self, data_type, *args, **kwargs):
        self.data_type = data_type
        self.df = pd.read_csv(f'/data/users1/egeenjaar/local-global/WM_LR_glasser.csv', index_col=0)
        if self.data_type == 'train':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.7)].copy()
        elif self.data_type == 'valid':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.7):int(self.df.shape[0] * 0.8)].copy()
        elif self.data_type == 'test':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.8):].copy()
        elif self.data_type == 'train_valid':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.8)].copy()
        self.indices = self.df.index.values
        self.behavioral = pd.read_csv('behavioral.csv', index_col=0)
        self.df = self.df.join(self.behavioral, how='left')
        to_replace = list(self.df['Age'].unique())
        value = np.arange(len(to_replace))
        d = {tr: v for tr, v in zip(to_replace, value)}
        self.df['Age'] = self.df['Age'].replace(d)
        self.df['Gender'] = self.df['Gender'] == 'F'
        data, masks = [], []
        for (i, row) in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            data_subj = np.load(row['fmri']).astype(np.float16)
            data.append(data_subj)
        
        self.window_size_sec = 20.
        self.window_size = 10 #int(self.window_size_sec // self.new_TR)
        self.window_step = 8 #int(window_step // self.new_TR)
        self.data = np.stack(data, axis=0)

    @property
    def data_size(self):
        return 379

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.001

    @property
    def num_timesteps(self):
        return 284

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        data = self.data[ix]
        data = torch.from_numpy(data)
       # data = data.view(-1, 91282, self.window_size).permute(0, 2, 1)
        return data, 0, 0, (0, self.df.loc[self.indices[ix], 'WM_Task_Acc'])#(0, self.df.loc[self.indices[ix], 'sz'])


class RestGlasser(Dataset):
    def __init__(self, data_type, *args, **kwargs):
        self.data_type = data_type
        self.df = pd.read_csv(f'/data/users1/egeenjaar/local-global/REST1_LR_glasser.csv', index_col=0)
        if self.data_type == 'train':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.7)].copy()
        elif self.data_type == 'valid':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.7):int(self.df.shape[0] * 0.8)].copy()
        elif self.data_type == 'test':
            self.df = self.df.iloc[int(self.df.shape[0] * 0.8):].copy()
        elif self.data_type == 'train_valid':
            self.df = self.df.iloc[:int(self.df.shape[0] * 0.8)].copy()
        self.indices = self.df.index.values
        self.behavioral = pd.read_csv('behavioral.csv', index_col=0)
        self.df = self.df.join(self.behavioral, how='left')
        to_replace = list(self.df['Age'].unique())
        value = np.arange(len(to_replace))
        d = {tr: v for tr, v in zip(to_replace, value)}
        self.df['Age'] = self.df['Age'].replace(d)
        self.df['Gender'] = self.df['Gender'] == 'F'
        data, masks = [], []
        for (i, row) in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            data_subj = np.load(row['fmri']).astype(np.float16)
            if data_subj.shape[0] != 1200:
                print(i, row, data_subj.shape)
            data.append(data_subj)
        
        self.window_size_sec = 20.
        self.window_size = 10 #int(self.window_size_sec // self.new_TR)
        self.window_step = 8 #int(window_step // self.new_TR)
        self.data = np.stack(data, axis=0)

    @property
    def data_size(self):
        return 379

    @property
    def mask_windows(self):
        return 4

    @property
    def learning_rate(self):
        return 0.0001

    @property
    def num_timesteps(self):
        return 284

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        data = self.data[ix]
        data = torch.from_numpy(data)[:1024]
       # data = data.view(-1, 91282, self.window_size).permute(0, 2, 1)
        return data, 0, 0, (0, self.df.loc[self.indices[ix], 'WM_Task_Acc'])#(0, self.df.loc[self.indices[ix], 'sz'])
