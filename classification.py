import torch
import importlib
import numpy as np
from lib.utils import (
    get_default_config, embed_context, generate_version_name)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import glob

config = get_default_config([''])
seeds = [42, 1337, 9999, 1212, 8585, 6767]
models = ['CO', 'DSVAE', 'IDSVAE', 'LVAE']
local_sizes = [0, 2, 4, 8, 16]
context_sizes = [0, 2, 4, 8, 16]
config = config.copy()
for seed in seeds:
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    for model_name in models:
        for local_size in local_sizes:
            for context_size in context_sizes:
                config['seed'] = seed
                config['local_size'] = local_size
                config['model'] = model_name
                config['context_size'] = context_size
                version = generate_version_name(config)
                data_module = importlib.import_module('lib.data')
                dataset_type = getattr(data_module, config['dataset'])
                train_dataset = dataset_type('train', config['seed'])
                valid_dataset = dataset_type('valid', config['seed'])
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                model_module = importlib.import_module('lib.model')
                model_type = getattr(model_module, config['model'])

                model_path = f'ray_results/{version}'

                files = glob.glob(model_path + '/**/*best.ckpt', recursive = True)
                if len(files) == 0:
                    continue
                print(files)
                for ckpt in files:
                    model = model_type.load_from_checkpoint(ckpt)
                    model = model.to(device)
                    model.eval()

                    tr_ge, tr_y_orig = embed_context_data(
                        device, model, train_dataset)
                    va_ge, va_y_orig = embed_context_data(
                        device, model, valid_dataset)

                    num_tr_subjects, tr_num_windows, latent_dim = tr_ge.shape
                    num_va_subjects, va_num_windows, _ = va_ge.shape

                    # Repeat class for each context window
                    tr_y = np.repeat(
                        tr_y_orig[:, np.newaxis], tr_ge.shape[1], axis=1)
                    va_y = np.repeat(
                        va_y_orig[:, np.newaxis], va_ge.shape[1], axis=1)

                    # Reshape context embeddings
                    tr_ge = np.reshape(
                        tr_ge, (num_tr_subjects * tr_num_windows, latent_dim))
                    va_ge = np.reshape(
                        va_ge, (num_va_subjects * va_num_windows, latent_dim))

                    tr_y = tr_y.flatten()
                    va_y = va_y.flatten()

                    # Scale and then predict on test set
                    ssc = StandardScaler()
                    tr_ge = ssc.fit_transform(tr_ge)
                    va_ge = ssc.transform(va_ge)
                    svm = SVC(kernel='rbf')
                    svm.fit(tr_ge, tr_y)
                    print(ckpt, svm.score(va_ge, va_y))
