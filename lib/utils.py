import yaml
import torch
import pandas as pd
from ray import tune
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_default_config(args):
    if len(args) > 1:
        with Path(f'configs/config_{int(args[1])}.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    else:
        with Path('configs/default.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    return dict(default_conf)


def embed_context(device, model, dataset):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=5)
    context_embeddings = torch.empty((
        dataset.num_subjects, dataset.num_windows, model.context_size))
    ys = torch.empty((dataset.num_subjects, ), dtype=bool)
    for (i, batch) in enumerate(dataloader):
        x, _, (subj_ix, temp_ix), y = batch
        x = x.to(device, non_blocking=True)
        x = x.permute(1, 0, 2)
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            context_dist = model.embed_context(x)
        context_embeddings[subj_ix, temp_ix] = context_dist.mean.cpu()
        ys[subj_ix] = y.cpu()
    return context_embeddings, ys


def get_icafbirn(seed):
    df = pd.read_csv('/data/users1/egeenjaar/local-global/data/ica_fbirn/info_df.csv', index_col=0)
    trainval_index, test_index = train_test_split(
        df.index.values, train_size=0.8, random_state=seed, stratify=df['sex'])
    train_index, valid_index = train_test_split(
        trainval_index, train_size=0.9, random_state=seed,
        stratify=df.loc[trainval_index, 'sex'])
    train_df = df.loc[train_index].copy()
    valid_df = df.loc[valid_index].copy()
    test_df = df.loc[test_index].copy()
    return train_df, valid_df, test_df


def generate_version_name(config):
    version = f'm{config["model"]}_' \
        f'd{config["dataset"]}_' \
        f's{config["seed"]}_' \
        f's{config["local_size"]}_' \
        f'g{config["context_size"]}'
    return version

def get_search_space(config):
    return {"train_loop_config": {
        # Unused parameter for Context-only model
        "num_layers": [1, 2, 3, 4] if not config['model'] == 'CO' else [1],
        "spatial_hidden_size": [64, 128, 256],
        # Unused parameter for Context-only model
        "temporal_hidden_size": [128, 256, 512] if not config['model'] == 'CO' else [128],
        "lr": tune.loguniform(1e-6, 1e-3),
        "batch_size": [64, 128],
        # Unused parameter for Context-only model
       # "beta": tune.loguniform(1e-5, 1e-3) if not config['model'] == 'CO' else [0],
       "beta": tune.loguniform(1e-7, 1e-3),
        # Essentially 'beta' for the context-only model
        #"gamma": tune.loguniform(1e-7, 1e-3) if not config['model'] == 'CO' else tune.loguniform(1e-5, 1e-3),
        "gamma": tune.loguniform(1e-7, 1e-5),
        "theta": tune.loguniform(1e-7, 1e-3) if 'Cont' in config['model'] else [0]}
    }
    
def get_start_config(config):
    return {"train_loop_config": {
        # Unused parameter for Context-only model
        "num_layers": tune.choice([1, 2, 3, 4, 5]) if not config['model'] == 'CO' else tune.choice([1]),
        "spatial_hidden_size": tune.choice([64, 128, 256]),
        # Unused parameter for Context-only model
        "temporal_hidden_size": tune.choice([128, 256, 512]) if not config['model'] == 'CO' else tune.choice([128]),
       # "lr": tune.loguniform(1e-4, 2e-3),
       "lr": tune.loguniform(1e-4, 2e-3),
        "batch_size": tune.choice([64, 128]),
        # Unused parameter for Context-only model
       # "beta": tune.loguniform(1e-5, 1e-3) if not config['model'] == 'CO' else [0],
       "beta": tune.loguniform(1e-5, 1e-3),
        # Essentially 'beta' for the context-only model
        #"gamma": tune.loguniform(1e-7, 1e-3) if not config['model'] == 'CO' else tune.loguniform(1e-5, 1e-3),
        "gamma": tune.loguniform(1e-7, 1e-3),
        "theta": tune.loguniform(1e-7, 1e-3) if 'Cont' in config['model'] else tune.choice([0]),
        "dropout": tune.choice([0, 0.05, 0.1, 0.25, 0.5])}
    }