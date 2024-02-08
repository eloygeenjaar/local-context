import re
import torch
import importlib
import numpy as np
from lib.utils import (
    get_default_config, embed_all, generate_version_name, init_data_module, init_model, load_hyperparameters)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import csv
import json


config = get_default_config([''])
kernels = ['linear', 'rbf']
datasets = ['UpsampledICAfBIRN', 'ICAfBIRN']
seeds = [42, 1337]
models = ['CFDSVAE', 'DSVAE']
local_sizes = [2]
context_sizes = [2]
config = config.copy()

results_dic_score = {}
results_dic_mse = {}

for dataset_name in datasets:
    dm = init_data_module(config)
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
                    config['dataset'] = dataset_name

                    version = generate_version_name(config)
                    result_p = Path(f'ray_results/{version}')
                    ckpt_p = result_p / 'final.ckpt'

                    if not os.path.isfile(ckpt_p):
                        print(f'{version} ckpt not found')
                        continue

                    with open(result_p / 'result.json') as f:
                        for line in f:
                            pass
                        last_line = line

                    val_mse = json.loads(last_line)['va_mse']

                    hyperparameters = load_hyperparameters(result_p / 'params.json')
                    model = init_model(config, hyperparameters, viz=False, ckpt_path=ckpt_p)
                    

                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')

                    model = model.to(device)
                    model.eval()

                    dataloaders = embed_all(config, model, dm)
                    train_dataloader, val_dataloader, test_dataloader = dataloaders['train'], dataloaders['valid'], dataloaders['test']

                    tr_ge, tr_y_orig = train_dataloader['context_mean'], train_dataloader['target']
                    va_ge, va_y_orig = val_dataloader['context_mean'], val_dataloader['target']

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
                    for kernel in kernels:
                        svm = SVC(kernel=kernel)
                        svm.fit(tr_ge, tr_y)
                        score = svm.score(va_ge, va_y)
                        print(dataset_name, model_name, context_size, 
                              kernel, svm.score(va_ge, va_y))
                        del svm

                    version_no_seed = re.sub('_se\d+', '', version)
                    if version not in results_dic_score:
                        results_dic_score[version_no_seed] = [score]
                        results_dic_mse[version_no_seed] = val_mse
                    else:
                        results_dic_score[version_no_seed].append(score)


with open('classification.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Model", "Kernel", "ACC", "MSE"])

    for key in results_dic_mse.keys():
        average = np.average(results_dic_score[key])
        mse = results_dic_mse[key]
        row = [key, "Linear", str(average), str(mse)]
        csvwriter.writerow(row)
