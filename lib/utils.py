import os
import yaml
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle

def get_default_config(args):
    if len(args) > 1:
        with Path(f'configs/config_{int(args[1])}.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    else:
        with Path('configs/default.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    return dict(default_conf)

# Adapted from: https://github.com/googleinterns/local_global_ts_representation/blob/main/gl_rep/data_loaders.py
def get_airquality(normalization, seed):
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
    signal_maps = np.stack(signal_maps).astype(bool)
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
    
    if normalization == 'dataset':
        train_signals, valid_signals, test_signals, _ = normalize_signals(
            signals, signal_maps,
            (train_inds, valid_inds, test_inds))
        train_signals = torch.from_numpy(train_signals)
        valid_signals = torch.from_numpy(valid_signals)
        test_signals = torch.from_numpy(test_signals)
        signal_maps = torch.from_numpy(signal_maps)
    else:
        train_signals = torch.from_numpy(signals[train_inds])
        valid_signals = torch.from_numpy(signals[valid_inds])
        test_signals = torch.from_numpy(signals[test_inds])
        signal_maps = torch.from_numpy(signal_maps)
    
    return ((train_signals, valid_signals, test_signals),
            (z_ls[train_inds], z_ls[valid_inds], z_ls[test_inds]),
            (z_gs[train_inds], z_gs[valid_inds], z_gs[test_inds]),
            (~signal_maps[train_inds], ~signal_maps[valid_inds], ~signal_maps[test_inds]))

# Adapted from: https://github.com/googleinterns/local_global_ts_representation/blob/main/gl_rep/data_loaders.py
def get_physionet(normalization, seed, dataset = 'set-a'):
    """Function to load the The PhysioNet Computing in Cardiology Challenge 2012 dataset into TF dataset objects

    The data is loaded, normalized, padded, and a mask channel is generated to indicate missing observations
    The raw csv files can be downloaded from:
    https://physionet.org/content/challenge-2012/1.0.0/
    A number of steps are borrowed from this repo: https://github.com/alistairewj/challenge2012

    Args:
        normalize: The type of data normalizatino to perform ["none", "mean_zero", "min_max"]
    """
    feature_map = {'Albumin': 'Serum Albumin (g/dL)',
                'ALP': 'Alkaline phosphatase (IU/L)',
                'ALT': 'Alanine transaminase (IU/L)',
                'AST': 'Aspartate transaminase (IU/L)',
                'Bilirubin': 'Bilirubin (mg/dL)',
                'BUN': 'Blood urea nitrogen (mg/dL)',
                'Cholesterol': 'Cholesterol (mg/dL)',
                'Creatinine': 'Serum creatinine (mg/dL)',
                'DiasABP': 'Invasive diastolic arterial blood pressure (mmHg)',
                'FiO2': 'Fractional inspired O2 (0-1)',
                'GCS': 'Glasgow Coma Score (3-15)',
                'Glucose': 'Serum glucose (mg/dL)',
                'HCO3': 'Serum bicarbonate (mmol/L)',
                'HCT': 'Hematocrit (%)',
                'HR': 'Heart rate (bpm)',
                'K': 'Serum potassium (mEq/L)',
                'Lactate': 'Lactate (mmol/L)',
                'Mg': 'Serum magnesium (mmol/L)',
                'MAP': 'Invasive mean arterial blood pressure (mmHg)',
                'Na': 'Serum sodium (mEq/L)',
                'NIDiasABP': 'Non-invasive diastolic arterial blood pressure (mmHg)',
                'NIMAP': 'Non-invasive mean arterial blood pressure (mmHg)',
                'NISysABP': 'Non-invasive systolic arterial blood pressure (mmHg)',
                'PaCO2': 'partial pressure of arterial CO2 (mmHg)',
                'PaO2': 'Partial pressure of arterial O2 (mmHg)',
                'pH': 'Arterial pH (0-14)',
                'Platelets': 'Platelets (cells/nL)',
                'RespRate': 'Respiration rate (bpm)',
                'SaO2': 'O2 saturation in hemoglobin (%)',
                'SysABP': 'Invasive systolic arterial blood pressure (mmHg)',
                'Temp': 'Temperature (°C)',
                'TroponinI': 'Troponin-I (μg/L)',
                'TroponinT': 'Troponin-T (μg/L)',
                'Urine': 'Urine output (mL)',
                'WBC': 'White blood cell count (cells/nL)'
                   }
    feature_list = list(feature_map.keys())
    local_list = ['MechVent', 'Weight']
    data_dir = './data/physionet'
    static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    
    if os.path.exists(('./data/physionet/processed_df.csv')):
        df_full = pd.read_csv('./data/physionet/processed_df.csv')
        df_static = pd.read_csv('./data/physionet/processed_static_df.csv')
    else:
        txt_all = list()
        for f in os.listdir(os.path.join(data_dir, dataset)):
            with open(os.path.join(data_dir, dataset, f), 'r') as fp:
                txt = fp.readlines()
            # get recordid to add as a column
            recordid = txt[1].rstrip('\n').split(',')[-1]
            try:
                txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
                txt_all.extend(txt[1:])
            except:
                continue

        # convert to pandas dataframe
        df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])

        # extract static variables into a separate dataframe
        df_static = df.loc[df['time'] == '00:00', :].copy()

        df_static = df_static.loc[df['parameter'].isin(static_vars)]

        # remove these from original df
        idxDrop = df_static.index
        df = df.loc[~df.index.isin(idxDrop), :]

        # pivot on parameter so there is one column per parameter
        df_static = df_static.pivot(index='recordid', columns='parameter', values='value')

        # some conversions on columns for convenience
        df['value'] = pd.to_numeric(df['value'], errors='raise')
        df['time'] = df['time'].map(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

        df.head()
        # convert static into numeric
        for c in df_static.columns:
            df_static[c] = pd.to_numeric(df_static[c])

        # preprocess
        for c in df_static.columns:
            x = df_static[c]
            if c == 'Age':
                # replace anon ages with 91.4
                idx = x > 130
                df_static.loc[idx, c] = 91.4
            elif c == 'Gender':
                idx = x < 0
                df_static.loc[idx, c] = np.nan
            elif c == 'Height':
                idx = x < 0
                df_static.loc[idx, c] = np.nan

                # fix incorrectly recorded heights

                # 1.8 -> 180
                idx = x < 10
                df_static.loc[idx, c] = df_static.loc[idx, c] * 100

                # 18 -> 180
                idx = x < 25
                df_static.loc[idx, c] = df_static.loc[idx, c] * 10

                # 81.8 -> 180 (inch -> cm)
                idx = x < 100
                df_static.loc[idx, c] = df_static.loc[idx, c] * 2.2

                # 1800 -> 180
                idx = x > 1000
                df_static.loc[idx, c] = df_static.loc[idx, c] * 0.1

                # 400 -> 157
                idx = x > 250
                df_static.loc[idx, c] = df_static.loc[idx, c] * 0.3937

            elif c == 'Weight':
                idx = x < 35
                df_static.loc[idx, c] = np.nan

                idx = x > 299
                df_static.loc[idx, c] = np.nan


        df = delete_value(df, 'DiasABP', -1)
        df = replace_value(df, 'DiasABP', value=np.nan, below=1)
        df = replace_value(df, 'DiasABP', value=np.nan, above=200)
        df = replace_value(df, 'SysABP', value=np.nan, below=1)
        df = replace_value(df, 'MAP', value=np.nan, below=1)

        df = replace_value(df, 'NIDiasABP', value=np.nan, below=1)
        df = replace_value(df, 'NISysABP', value=np.nan, below=1)
        df = replace_value(df, 'NIMAP', value=np.nan, below=1)

        df = replace_value(df, 'HR', value=np.nan, below=1)
        df = replace_value(df, 'HR', value=np.nan, above=299)

        df = replace_value(df, 'PaCO2', value=np.nan, below=1)
        df = replace_value(df, 'PaCO2', value=lambda x: x * 10, below=10)

        df = replace_value(df, 'PaO2', value=np.nan, below=1)
        df = replace_value(df, 'PaO2', value=lambda x: x * 10, below=20)

        # the order of these steps matters
        df = replace_value(df, 'pH', value=lambda x: x * 10, below=0.8, above=0.65)
        df = replace_value(df, 'pH', value=lambda x: x * 0.1, below=80, above=65)
        df = replace_value(df, 'pH', value=lambda x: x * 0.01, below=800, above=650)
        df = replace_value(df, 'pH', value=np.nan, below=6.5)
        df = replace_value(df, 'pH', value=np.nan, above=8.0)

        # convert to farenheit
        df = replace_value(df, 'Temp', value=lambda x: x * 9 / 5 + 32, below=10, above=1)
        df = replace_value(df, 'Temp', value=lambda x: (x - 32) * 5 / 9, below=113, above=95)

        df = replace_value(df, 'Temp', value=np.nan, below=25)
        df = replace_value(df, 'Temp', value=np.nan, above=45)

        df = replace_value(df, 'RespRate', value=np.nan, below=1)
        df = replace_value(df, 'WBC', value=np.nan, below=1)

        df = replace_value(df, 'Weight', value=np.nan, below=35)
        df = replace_value(df, 'Weight', value=np.nan, above=299)


        df_full = pd.DataFrame(columns=['time', 'recordid']+feature_list+local_list)
        df_sampled = df.groupby(['recordid'])#, 'parameter'])
        for i, sample in enumerate(df_sampled):
            id = sample[0]
            df_signal = sample[1].groupby(['parameter'])
            signal_df = pd.DataFrame(columns=['time', 'recordid'])
            for j, signal_sample in enumerate(df_signal):
                param = signal_sample[0].utils.rnn.pad_sequence
                sub_df['time'] = signal_sample[1]['time']
                signal_df = signal_df.merge(sub_df, how='outer', on=['recordid', 'time'], sort=True, suffixes=[None, None])
            # Bin the values
            bins = pd.cut(signal_df.time, np.arange(signal_df['time'].iloc[0], signal_df['time'].iloc[-1], 60))
            col_list = list(signal_df.columns[2:])# - ['recordid', 'time']
            signal_df_binned =  pd.DataFrame(columns=signal_df.columns)
            signal_df_binned['time'] = np.arange(signal_df['time'].iloc[0], signal_df['time'].iloc[-1], 60)[:-1]
            signal_df_binned['recordid'] = id
            signal_df_binned[col_list] = signal_df.groupby(bins).agg(dict(zip(col_list, ["mean"]*len(col_list)))).to_numpy()#{"Temperature": "mean"})
            df_full = pd.concat([signal_df_binned, df_full])
        df_full.to_csv('./data/physionet/processed_df.csv')
        df_static.to_csv('./data/physionet/processed_static_df.csv')


    selected_features = ['DiasABP', 'GCS', 'HCT', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp']

    # load in outcomes
    if dataset == 'set-a':
        y = pd.read_csv(os.path.join(data_dir, 'Outcomes-a.txt'))
    elif dataset == 'set-b':
        y = pd.read_csv(os.path.join(data_dir,  'Outcomes-.txt'))
    label_list = ['SAPS-I', 'SOFA', 'In-hospital_death']


    df_sampled = df_full.groupby(['recordid'])
    max_len = 80
    signals, signal_maps, signal_lens = [], [], []
    z_ls, z_gs = [], []
    for i, sample in enumerate(df_sampled):
        id = sample[0]
        x = sample[1][selected_features]
        if np.array(x.isna()).mean()>0.6 or len(x)<0.5*max_len:
            continue
        sample_map = x.isna().astype('float32')
        labels = y[y['RecordID']==id][label_list]
        z_l = sample[1][['MechVent']]
        x = x.fillna(0.0)
        z_g = df_static[df_static['RecordID']==id][['Age', 'Gender', 'Height', 'ICUType', 'Weight']]
        signals.append(torch.from_numpy(np.array(x)))
        signal_maps.append(torch.from_numpy(np.array(sample_map)))
        z_ls.append(torch.from_numpy(np.array(z_l)))
        z_gs.append(np.concatenate([np.array(z_g), np.array(labels)], axis=-1).reshape(-1,))
        signal_lens.append(min(max_len, len(x)))

    signals = nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0.0)
    z_ls =  nn.utils.rnn.pad_sequence(z_ls, batch_first=True, padding_value=0.0)
    maps = nn.utils.rnn.pad_sequence(signal_maps, batch_first=True, padding_value=0.0).bool()
    z_gs = torch.from_numpy(np.array(z_gs))
    signal_lens = np.array(signal_lens)

    test_inds = list(range(int(0.2*len(signals))))
    inds = list(range(int(0.2*len(signals)), len(signals)))
    rng = np.random.default_rng(seed)
    rng.shuffle(inds)
    train_inds = inds[:int(0.8*len(inds))]
    valid_inds = inds[int(0.8*len(inds)):]

    # plot a random sample
    ind = np.random.randint(0, len(train_inds))
    f, axs = plt.subplots(nrows=signals.shape[-1], ncols=1, figsize=(18, 14))
    for i, ax in enumerate(axs):
        ax.plot(signals[ind, :, i])
        ax.set_title(feature_list[i])
    plt.tight_layout()
    plt.savefig('./data/physionet/sample.png')
    plt.close(f)

    if normalization == 'dataset':
        train_signals, valid_signals, test_signals, _ = normalize_signals(
            signals, maps,
            (train_inds, valid_inds, test_inds))
        train_signals = torch.from_numpy(train_signals)
        valid_signals = torch.from_numpy(valid_signals)
        test_signals = torch.from_numpy(test_signals)
    else:
        train_signals = signals[train_inds]
        valid_signals = signals[valid_inds]
        test_signals = signals[test_inds]

    return ((train_signals, valid_signals, test_signals),
            (z_ls[train_inds], z_ls[valid_inds], z_ls[test_inds]),
            (z_gs[train_inds], z_gs[valid_inds], z_gs[test_inds]),
            (~maps[train_inds], ~maps[valid_inds], ~maps[test_inds]))

def delete_value(df, c, value=0):
    """Helper function for processing the Physionet dataset"""
    idx = df['parameter'] == c
    idx = idx & (df['value'] == value)
    df.loc[idx, 'value'] = np.nan
    return df

def replace_value(df, c, value=np.nan, below=None, above=None):
    """Helper function for processing the Physionet dataset"""
    idx = df['parameter'] == c
    if below is not None:
        idx = idx & (df['value'] < below)
    if above is not None:
        idx = idx & (df['value'] > above)
    if 'function' in str(type(value)):
        # value replacement is a function of the input
        df.loc[idx, 'value'] = df.loc[idx, 'value'].apply(value)
    else:
        df.loc[idx, 'value'] = value
    return df

def embed_data(device, model, train_dataset, valid_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    y_train, embedding_train, time_train = [], [], []
    for (i, batch) in enumerate(train_dataloader):
        x, mask, (y_local, y_global) = batch
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.no_grad():
            p_x_hat, local_dist, global_dist, z_t, z_g = model(x, mask, window_step=model.window_step)
        y_train.append(y_global.cpu())
        embedding_train.append(global_dist.mean[:x.size(0)].cpu())
        time_train.append(local_dist.mean.cpu())

    y_train = torch.cat(y_train, dim=0).numpy()
    x_train = torch.cat(embedding_train, dim=0).numpy()
    x_tr_time = torch.cat(time_train, dim=0).numpy()

    y_valid, embedding_valid, time_valid = [], [], []
    for (i, batch) in enumerate(valid_dataloader):
        x, mask, (y_local, y_global) = batch
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.no_grad():
            p_x_hat, local_dist, global_dist, z_t, z_g = model(x, mask, window_step=model.window_step)
        y_valid.append(y_global.cpu())
        embedding_valid.append(global_dist.mean[:x.size(0)].cpu())
        time_valid.append(local_dist.mean.cpu())

    y_valid = torch.cat(y_valid, dim=0).numpy()
    x_valid = torch.cat(embedding_valid, dim=0).numpy()
    x_va_time = torch.cat(time_valid, dim=0).numpy()

    y_test, embedding_test, time_test = [], [], []
    for (i, batch) in enumerate(test_dataloader):
        x, mask, (y_local, y_global) = batch
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.no_grad():
            p_x_hat, local_dist, global_dist, z_t, z_g = model(x, mask, window_step=model.window_step)
        y_test.append(y_global.cpu())
        embedding_test.append(global_dist.mean[:x.size(0)].cpu())
        time_test.append(local_dist.mean.cpu())

    y_test = torch.cat(y_test, dim=0).numpy()
    x_test = torch.cat(embedding_test, dim=0).numpy()
    x_te_time = torch.cat(time_test, dim=0).numpy()

    return ((x_train, x_tr_time, y_train),
            (x_valid, x_va_time, y_valid),
            (x_test, x_te_time, y_test))
    
def embed_global_data(device, model, train_dataset, valid_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    y_train, embedding_train = [], []
    for (i, batch) in enumerate(train_dataloader):
        x, mask, (y_local, y_global) = batch
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _, global_dist = model.global_encoder(x)
        y_train.append(y_global.cpu())
        embedding_train.append(global_dist.mean[:x.size(0)].cpu())

    y_train = torch.cat(y_train, dim=0).numpy()
    x_train = torch.cat(embedding_train, dim=0).numpy()

    y_valid, embedding_valid = [], []
    for (i, batch) in enumerate(valid_dataloader):
        x, mask, (y_local, y_global) = batch
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _, global_dist = model.global_encoder(x)
        y_valid.append(y_global.cpu())
        embedding_valid.append(global_dist.mean[:x.size(0)].cpu())

    y_valid = torch.cat(y_valid, dim=0).numpy()
    x_valid = torch.cat(embedding_valid, dim=0).numpy()

    y_test, embedding_test = [], []
    for (i, batch) in enumerate(test_dataloader):
        x, mask, (y_local, y_global) = batch
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _, global_dist = model.global_encoder(x)
        y_test.append(y_global.cpu())
        embedding_test.append(global_dist.mean[:x.size(0)].cpu())

    y_test = torch.cat(y_test, dim=0).numpy()
    x_test = torch.cat(embedding_test, dim=0).numpy()

    return ((x_train, y_train),
            (x_valid, y_valid),
            (x_test, y_test))

# Adapted from: https://github.com/googleinterns/local_global_ts_representation/blob/main/gl_rep/data_loaders.py
def normalize_signals(signals, masks, inds):
    # Both datasets I use are normalized with mean_zero, see: 
    # https://github.com/googleinterns/local_global_ts_representation/blob/main/main.py
    train_inds, valid_inds, test_inds = inds
    mean_vals, std_vals = [], []
    for feat in range(signals.shape[-1]):
        mean_vals.append(
            signals[train_inds, :, feat][masks[train_inds, :, feat] == 0].mean())
        std_vals.append(
            signals[train_inds, :, feat][masks[train_inds, :, feat] == 0].std())
        #mean_vals.append(
        #    signals[train_inds, :, feat][masks[train_inds, :, feat] == 0].mean(0)
        #)
        #std_vals.append(
        #    signals[train_inds, :, feat][masks[train_inds, :, feat] == 0].std(0)
        #)
    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)
    std_vals = np.where(std_vals==0, 1., std_vals)
    train_signals = np.where(masks[train_inds] == 0,
                                (signals[train_inds] - mean_vals) / std_vals,
                                signals[train_inds])
    valid_signals = np.where(masks[valid_inds] == 0,
                                (signals[valid_inds] - mean_vals) / std_vals,
                                signals[valid_inds])
    test_signals = np.where(masks[test_inds] == 0, (signals[test_inds] - mean_vals) / std_vals,
                            signals[test_inds])
    normalization_specs = {'mean': mean_vals, 'std': std_vals}
    return train_signals, valid_signals, test_signals, normalization_specs

def get_fbirn(seed, split_ix, n_splits):
    if Path('./data/fbirn/data.csv').is_file():
        df = pd.read_csv('./data/fbirn/data.csv', index_col=0)
    else:
        clin_data = pd.read_excel('/data/qneuromark/Data/FBIRN/Data_info/clinData03-20-2012-limited.xlsx', index_col=0)
        data_path = Path('/data/qneuromark/Data/FBIRN/ZN_Neuromark/ZN_Prep_fMRI')
        fmri_subjects, fmri_paths, diagnoses = [], [], []
        
        for folder in data_path.iterdir():
            fmri_path = folder / 'SM.nii'
            if fmri_path.is_file():
                subject_id = int(folder.name)
                if subject_id in clin_data.index:
                    fmri_subjects.append(subject_id)
                    fmri_paths.append(str(fmri_path))
                    if isinstance(clin_data.loc[subject_id, 'Demographics_nDEMOG_DIAGNOSIS'], pd.Series):
                        diagnosis = clin_data.loc[subject_id, 'Demographics_nDEMOG_DIAGNOSIS'].to_list()[0]
                    else:
                        diagnosis = clin_data.loc[subject_id, 'Demographics_nDEMOG_DIAGNOSIS']
                    diagnoses.append(diagnosis)
        fmri_subjects = np.asarray(fmri_subjects)
        fmri_paths = np.asarray(fmri_paths)
        diagnoses = np.asarray(diagnoses)
        arr = np.stack((fmri_subjects, fmri_paths, diagnoses), axis=1)
        df = pd.DataFrame(arr, columns=['subjectID', 'path', 'diagnosis'])
        df['diagnosis'] = df['diagnosis'].astype(int)
        df['diagnosis'] = (df['diagnosis'] == 2)
        df.to_csv('./data/fbirn/data.csv')
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    for (i, (trainval_index, test_index)) in enumerate(skf.split(df['subjectID'], df['diagnosis'])):
        if i == split_ix:
            break
    
    train_index, valid_index = train_test_split(trainval_index, train_size=0.9, stratify=df['diagnosis'].iloc[trainval_index])
    train_df = df.iloc[train_index].copy()
    valid_df = df.iloc[valid_index].copy()
    test_df = df.iloc[test_index].copy()
    return train_df, valid_df, test_df

def get_icaukbb(seed):
    df = pd.read_csv('./data/ukbb/info_df.csv', index_col=0)
    trainval_index, test_index = train_test_split(df['ID'].index.values, train_size=0.8, random_state=seed)
    train_index, valid_index = train_test_split(trainval_index, train_size=0.9, random_state=seed)
    train_df = df.loc[train_index].copy()
    valid_df = df.loc[valid_index].copy()
    test_df = df.loc[test_index].copy()
    return train_df, valid_df, test_df

def get_icafbirn(seed):
    df = pd.read_csv('./data/ica_fbirn/info_df.csv', index_col=0)
    trainval_index, test_index = train_test_split(df.index.values, train_size=0.8, random_state=seed, stratify=df['sex'])
    train_index, valid_index = train_test_split(trainval_index, train_size=0.9, random_state=seed, stratify=df.loc[trainval_index, 'sex'])
    train_df = df.loc[train_index].copy()
    valid_df = df.loc[valid_index].copy()
    test_df = df.loc[test_index].copy()
    return train_df, valid_df, test_df

def get_sprite(seed):
    data = pickle.load(open('./data/sprite/data.pkl', 'rb'))
    X_train, X_test, A_train, A_test = data['X_train'], data['X_test'], data['A_train'], data['A_test']
    D_train, D_test = data['D_train'], data['D_test']
    c_augs_train, c_augs_test = data['c_augs_train'], data['c_augs_test']
    m_augs_train, m_augs_test = data['m_augs_train'], data['m_augs_test']
    return X_train, X_test
