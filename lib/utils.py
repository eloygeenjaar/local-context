import yaml
import json
import torch
import importlib
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from optuna import distributions as od
from typing import Dict
from ray import tune
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold


class DataModule(pl.LightningDataModule):
    def __init__(self, config, dataset_type, shuffle_train=True):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.batch_size = config['batch_size']
        self.shuffle_train = shuffle_train

    def setup(self, stage=None):
        self.train_dataset = self.dataset_type(
            'train', self.config['seed'], self.config['window_size'], self.config['window_step'])
        self.valid_dataset = self.dataset_type(
            'valid', self.config['seed'], self.config['window_size'], self.config['window_step'])
        self.test_dataset = self.dataset_type(
            'test', self.config['seed'], self.config['window_size'], self.config['window_step'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=5, pin_memory=True,
                          batch_size=self.config["batch_size"], shuffle=self.shuffle_train,
                          persistent_workers=True, prefetch_factor=5, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, num_workers=5, pin_memory=True,
                          batch_size=2048, shuffle=False,
                          persistent_workers=True, prefetch_factor=5, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=5, pin_memory=True,
                          batch_size=self.config["batch_size"], shuffle=False,
                          persistent_workers=True, prefetch_factor=5, drop_last=False)


def get_default_config(args):
    if len(args) > 1:
        with Path(f'configs/config_{int(args[1])}.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    else:
        with Path('configs/default.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    return dict(default_conf)


def get_icafbirn(seed, fold=0):
    df = pd.read_csv('/data/users1/egeenjaar/local-global/data/ica_fbirn/info_df.csv', index_col=0)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    y = df['sz'].values
    splits = skf.split(df.index.values, y)
    splits = [split for split in splits]
    trainval_index, test_index = splits[fold]
    train_index, valid_index = train_test_split(
        trainval_index, train_size=0.9, random_state=seed,
        stratify=y[trainval_index])
    train_df = df.iloc[train_index].copy()
    valid_df = df.iloc[valid_index].copy()
    test_df = df.iloc[test_index].copy()
    return train_df, valid_df, test_df


def generate_version_name(config):
    version = f'm{config["model"]}_' \
        f'd{config["dataset"]}_' \
        f'se{config["seed"]}_' \
        f'ls{config["local_size"]}_' \
        f'cs{config["context_size"]}'
    return version


#def get_hyperparameters(config):
#    return {"train_loop_config": {
#        # Unused parameter for Context-only model
#        "num_layers": tune.choice([3, 4, 5, 6]) if not config['model'] == 'CO' else tune.choice([1]),
#        "spatial_hidden_size": tune.choice([64, 128, 256]),
#        # Unused parameter for Context-only model
#        "temporal_hidden_size": tune.choice([128, 256, 512]) if not config['model'] == 'CO' else tune.choice([128]),
#       "lr": tune.choice([1e-4, 5e-4, 1e-3]),
#        "batch_size": tune.choice([64, 128]),
#        # Unused parameter for Context-only model
#        "beta": tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
#        # Essentially 'beta' for the context-only model
#        "gamma": tune.choice([1e-5, 1e-4, 1e-3]),
#        "theta": tune.choice([1e-5, 1e-4, 1e-3]) if config['model'] == 'CDSVAE' else tune.choice([0]),
#        "lambda": tune.choice([1e-2, 1e-1]) if "CF" in config['model'] else tune.choice([0]),
#        "dropout": tune.choice([0, 0.1, 0.2])}
#    }


def get_hyperparameters(config):
    return {"train_loop_config": {
        # Unused parameter for Context-only model
        "num_layers": od.IntDistribution(low=3, high=5) if not config['model'] == 'CO' else od.CategoricalDistribution([1]),
        "spatial_hidden_size": od.CategoricalDistribution([64, 128, 256]),
        # Unused parameter for Context-only model
        "temporal_hidden_size": od.CategoricalDistribution([128, 256, 512]) if not config['model'] == 'CO' else od.CategoricalDistribution([128]),
        "lr": od.CategoricalDistribution([1e-4, 5e-4, 1e-3]),
        "batch_size": od.CategoricalDistribution([64, 128]),
        # Unused parameter for Context-only model
        "beta": od.IntDistribution(low=2, high=5), # These integers are converted to 1E-[int]
        # Essentially 'beta' for the context-only model
        "gamma": od.IntDistribution(low=3, high=5), # These integers are converted to 1E-[int]
        "theta": od.IntDistribution(low=3, high=3) if config['model'] == 'CDSVAE' else od.CategoricalDistribution([0]), # converted to 1E-[int]
        "lambda": od.IntDistribution(low=1, high=3) if "CF" in config['model'] else od.CategoricalDistribution([0]), # converted to 1E-[int]
        "dropout": od.FloatDistribution(low=0.0, step=0.1, high=0.2)}
    }

def load_hyperparameters(p: Path):
    with p.open('r') as f:
        hyperparameters = json.load(f)
    return hyperparameters['train_loop_config']


def init_data_module(config, shuffle_train=False) -> DataModule:
    data_module = importlib.import_module('lib.data')
    dataset_type = getattr(data_module, config['dataset'])
    dm = DataModule(config, dataset_type, shuffle_train=shuffle_train)
    dm.setup()
    return dm


def init_model(config, hyperparameters, viz, ckpt_path=None) -> pl.LightningModule:   
    model_module = importlib.import_module('lib.model')
    model_type = getattr(model_module, config['model'])
    model = model_type(config, hyperparameters, viz)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    return model


def embed_dataloader(config, model, dataloader) -> Dict[str, torch.Tensor]:
    num_subjects = dataloader.dataset.num_subjects
    num_windows = dataloader.dataset.num_windows
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dict = {
        'input': torch.empty((num_subjects, num_windows, config['window_size'], config['data_size']), device=device),
        'reconstruction': torch.empty((num_subjects, num_windows, config['window_size'], config['data_size']), device=device),
        'local_mean': torch.empty((num_subjects, num_windows, config['window_size'], config['local_size']), device=device),
        'local_sd': torch.empty((num_subjects, num_windows, config['window_size'], config['local_size']), device=device),
        'context_mean': torch.empty((num_subjects, num_windows, model.context_size), device=device),
        'context_sd': torch.empty((num_subjects, num_windows, model.context_size), device=device),
        'target': torch.empty((num_subjects, ), dtype=bool)
    }
    model.eval()
    model = model.to(device)
    for (i, batch) in enumerate(dataloader):
        x, x_p, (subj_ix, temp_ix), y = batch
        x = x.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        x = x.to(device, non_blocking=True)
        x_p = x_p.to(device, non_blocking=True)
        with torch.no_grad():
            model_output = model(x, x_p)
        upsampling = x.size(0) // model.window_size
        if upsampling > 1:
            x = x[(upsampling // 2)::upsampling]
        output_dict['input'][subj_ix, temp_ix] = x.permute(1, 0, 2)
        output_dict['reconstruction'][subj_ix, temp_ix] = model_output['x_hat'].permute(1, 0, 2)
        if config['model'] != 'CO':
            output_dict['local_mean'][subj_ix, temp_ix] = model_output['local_dist'].mean.permute(1, 0, 2)
            output_dict['local_sd'][subj_ix, temp_ix] = model_output['local_dist'].stddev.permute(1, 0, 2)
        if config['model'] != 'LVAE':
            output_dict['context_mean'][subj_ix, temp_ix] = model_output['context_dist'].mean
            output_dict['context_sd'][subj_ix, temp_ix] = model_output['context_dist'].stddev
        output_dict['target'][subj_ix] = y
    for key in output_dict.keys():
        output_dict[key] = output_dict[key].cpu()
    return output_dict


def embed_all(config, model, data_module) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        'train': embed_dataloader(config, model, data_module.train_dataloader()),
        'valid': embed_dataloader(config, model, data_module.val_dataloader()),
        'test': embed_dataloader(config, model, data_module.test_dataloader())
    }


def normal_sampling(rng, current_ix: int, length: int, sd: float):
    sample = int(np.round(rng.normal(0, sd), 0))
    # We want to sample other timesteps not the same
    if sample == 0 and (current_ix != length - 1) and (current_ix != 0):
        sampled_ix = current_ix + rng.integers(low=0, high=2) * 2 - 1
    # If the current index is zero then we can only sample up
    elif current_ix == 0:
        sampled_ix = abs(sample)
        if sampled_ix == 0:
            sampled_ix = 1
    # If we have the highest ix then we can only sample down
    elif current_ix == length - 1:
        sampled_ix = length - abs(sample)
        # This means sample = 0
        if sampled_ix == length - 1:
            sampled_ix = length - 2
    else:
        sampled_ix = current_ix + sample

    # Make sure the sampled ix is not 'out of bounds'    
    if sampled_ix < 0:
        sampled_ix = 0
    elif sampled_ix > length - 1:
        sampled_ix = length -1

    return sampled_ix
