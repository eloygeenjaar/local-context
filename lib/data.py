import os
import glob
import torch
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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

# Based on: https://github.com/googleinterns/local_global_ts_representation/blob/main/gl_rep/data_loaders.py#L552
class AirQuality(Dataset):
    def __init__(self, data_type, seed=42):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        all_files = glob.glob("./data/air_quality/*.csv")
        column_list = ["year",	"month", "day",	"hour",	"PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "station"]
        feature_list = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
        sample_len = 24 *28 *1  # 2 months worth of data
        all_stations = []
        for file_names in all_files:
            station_data = pd.read_csv(file_names)[column_list]
            all_stations.append(station_data)
        all_stations = pd.concat(all_stations, axis=0, ignore_index=True)
        df_sampled = all_stations[column_list].groupby(['year', 'month', 'station'])
        
        signals, signal_maps = [], []
        inds, valid_inds, test_inds = [], [], []
        z_ls, z_gs = [], []
        for i, sample in enumerate(df_sampled):
            if len(sample[1]) < sample_len:
                continue
            # Determine training indices for different years
            if sample[0][0] in [2013, 2014, 2015, 2017]:
                inds.extend([i]  )
            elif sample[0][0] in [2016]: # data from 2016 is used for testing, because we have fewer recordings for the final year
                test_inds.extend([i])
            x = sample[1][feature_list][:sample_len].astype('float32')
            sample_map = x.isna().astype('float32')
            z_l = sample[1][['day', 'RAIN']][:sample_len]
            x = x.fillna(0)
            z_g = np.array(sample[0])
            signals.append(np.array(x))
            signal_maps.append(np.array(sample_map))
            z_ls.append(np.array(z_l))
            z_gs.append(np.array(z_g))
        signals_len = np.zeros((len(signals),)) + sample_len
        signals = np.stack(signals)
        signal_maps = np.stack(signal_maps)
        z_ls = np.stack(z_ls)
        z_gs = np.stack(z_gs)

        rng = np.random.default_rng(seed)
        rng.shuffle(inds)
        train_inds = inds[:int(len(inds)*0.85)]
        valid_inds = inds[int(len(inds)*0.85):]

        # plot a random sample
        ind = np.random.randint(0, len(train_inds))
        f, axs = plt.subplots(nrows=signals[train_inds].shape[-1], ncols=1, figsize=(18 ,14))
        for i, ax in enumerate(axs):
            ax.plot(signals[train_inds][ind, :, i])
            ax.set_title(feature_list[i])
        plt.tight_layout()
        plt.savefig('./data/air_quality/sample.png')
        plt.clf()
        plt.close(f)

        print(z_gs)
        self.X = np.pad(signals[train_inds], ((0, 0), (4, 4), (0, 0)), mode='constant', constant_values=0)
        # Second column is month of the year
        self.local_y, self.global_y = z_ls[train_inds], z_gs[train_inds][:, 1].astype(np.int32)
        unique_months = np.unique(self.global_y)
        for (ix, month) in enumerate(unique_months):
            self.global_y[self.global_y == month] = ix
        print(np.unique(self.global_y))
        self.X = torch.from_numpy(self.X).float()
        self.local_y = torch.from_numpy(self.local_y).long()
        self.global_y = torch.from_numpy(self.global_y).long()
        print(self.X.size(), self.local_y.size())
        
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, ix):
        x = self.X[ix]
        mask = (np.abs(x.std(0)) >= 1E-4) 
        x[:, mask] -= x[:, mask].mean(0)
        x[:, mask] /= x[:, mask].std(0)
        x[:, ~mask] = 0.
        return x, (self.local_y[ix], self.global_y[ix])
