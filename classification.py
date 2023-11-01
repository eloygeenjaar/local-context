import os
import torch
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
from torch import nn
from pathlib import Path
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from lib.utils import get_default_config, embed_global_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = get_default_config([''])
seeds = [42, 1337, 9999, 1111, 1234]
models = ['TransGLR']
datasets = ['ICAfBIRN']
gammas = [0.0]
normalizations = ['individual']
results = np.zeros((len(seeds),
                    len(normalizations),
                    len(datasets),
                    len(models),
                    len(gammas)))
config = config.copy()
for (s_ix, seed) in enumerate(seeds):
    config['seed'] = seed
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    for (n_ix, normalization) in enumerate(normalizations):
        for (m_ix, model_name) in enumerate(models):
            for (d_ix, dataset) in enumerate(datasets):
                for (g_ix, gamma) in enumerate(gammas):
                    config['batch_size'] = 8
                    config['normalization'] = normalization
                    config['gamma'] = gamma
                    config['model'] = model_name
                    config['dataset'] = dataset
                    version = f'm{config["model"]}_d{config["dataset"]}_g{config["gamma"]}_s{config["seed"]}_n{config["normalization"]}_s{config["local_size"]}_g{config["global_size"]}_f{config["fold_ix"]}'
                    model_module = importlib.import_module('lib.model')
                    model_type = getattr(model_module, config['model'])
                    checkpoint_file = Path(f'lightning_logs/{version}/checkpoints/best.ckpt')
                    print(version)
                    if checkpoint_file.is_file():
                        model = model_type.load_from_checkpoint(checkpoint_file)
                        model = model.to(device)
                        model.eval()
                        print(model.training)
                        data_module = importlib.import_module('lib.data')
                        dataset_type = getattr(data_module, config['dataset'])
                        train_dataset = dataset_type('train', config['normalization'], config["seed"], None, None)
                        valid_dataset = dataset_type('valid', config['normalization'], config["seed"], None, None)
                        test_dataset = dataset_type('test', config['normalization'], config["seed"], None, None)
                        assert train_dataset.data_size == valid_dataset.data_size
                        
                        ((x_train, y_train),
                        (x_valid, y_valid),
                        (x_test, y_test)) = embed_global_data(device, model, train_dataset, valid_dataset, test_dataset)

                        #x_train = np.concatenate((x_train, x_valid), axis=0)
                        #y_train = np.concatenate((y_train, y_valid), axis=0)
                        scaler = StandardScaler()
                        x_train = scaler.fit_transform(x_train)
                        x_valid = scaler.transform(x_valid)
                        lr = LogisticRegression(max_iter=1000)
                        lr.fit(x_train, y_train)
                        print('-- results --')
                        print(seed, model_name, dataset, gamma)
                        print(lr.score(x_train, y_train))
                        acc = lr.score(x_valid, y_valid)
                        print(acc)
                        
                        results[s_ix, n_ix, d_ix, m_ix, g_ix] = acc
                        del lr
                        #del scaler

np.save('results/other_classification.npy', results)
avg_results = results.mean(0)
print('-- air quality ---')
print(avg_results[0])
print('-- physionet ---')
print(avg_results[1])
