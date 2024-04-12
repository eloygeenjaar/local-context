import re
import torch
import importlib
import numpy as np
import pandas as pd
from lib.utils import (
    get_default_config, embed_all, generate_version_name, init_data_module, init_model, load_hyperparameters)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch import vmap
from sklearn.decomposition import PCA
import os
import csv
import json


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

config = get_default_config([''])
kernels = ['linear', 'rbf']
datasets = ['ICAfBIRN', 'UpsampledICAfBIRN']
seeds = [42, 1337, 9999, 1212]
models = ['LVAE', 'CO', 'DSVAE', 'IDSVAE']
local_sizes = [2, 4, 8]
context_sizes = [0, 2, 4, 8]
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

                    df = pd.read_json(result_p / 'result.json', lines=True)

                    val_mse = df['va_mse'].min()

                    hyperparameters = load_hyperparameters(result_p / 'params.json')
                    model = init_model(config, hyperparameters, viz=False, ckpt_path=ckpt_p)
                    

                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')

                    model = model.to(device)
                    model.eval()

                    embeddings = embed_all(config, model, dm)
                    train_emb, valid_emb, test_emb = embeddings['train'], embeddings['valid'], embeddings['test']

                    if config['model'] != 'LVAE':
                        tr_ge, tr_y_orig = train_emb['context_mean'], train_emb['target']
                        va_ge, va_y_orig = valid_emb['context_mean'], valid_emb['target']
                        te_ge, te_y_orig = test_emb['context_mean'], test_emb['target']
                    else:
                        # Take mean of local embeddings in the window
                        tr_ge, tr_y_orig = train_emb['local_mean'].mean(-2), train_emb['target']
                        va_ge, va_y_orig = valid_emb['local_mean'].mean(-2), valid_emb['target']
                        te_ge, te_y_orig = test_emb['local_mean'].mean(-2), test_emb['target']
                        
                    tr_ge = np.concatenate((tr_ge, va_ge), axis=0)
                    tr_y_orig = np.concatenate((tr_y_orig, va_y_orig), axis=0)
                    num_tr_subjects, tr_num_windows, latent_dim = tr_ge.shape
                    num_te_subjects, te_num_windows, _ = te_ge.shape

                    # Repeat class for each context window
                    tr_y = np.repeat(
                        tr_y_orig[:, np.newaxis], tr_ge.shape[1], axis=1)
                    te_y = np.repeat(
                        te_y_orig[:, np.newaxis], te_ge.shape[1], axis=1)

                    # Reshape context embeddings
                    tr_ge = np.reshape(
                        tr_ge, (num_tr_subjects * tr_num_windows, latent_dim))
                    te_ge = np.reshape(
                        te_ge, (num_te_subjects * te_num_windows, latent_dim))

                    tr_y = tr_y.flatten()
                    te_y = te_y.flatten()

                    # Scale and then predict on test set
                    ssc = StandardScaler()
                    tr_ge = ssc.fit_transform(tr_ge)
                    te_ge = ssc.transform(te_ge)
                    kernel_results = {kernel: None for kernel in kernels}
                    for kernel in kernels:
                        svm = SVC(kernel=kernel)
                        svm.fit(tr_ge, tr_y)
                        score = svm.score(te_ge, te_y)
                        print(dataset_name, model_name, context_size, 
                              kernel, score)
                        del svm
                        kernel_results[kernel] = score

                    version_no_seed = re.sub('_se\d+', '', version)
                    
                    if version_no_seed not in results_dic_score:
                        results_dic_score[version_no_seed] = {kernel: [kernel_results[kernel]] for kernel in kernels}
                        results_dic_mse[version_no_seed] = [val_mse]
                    else:
                        for kernel in kernels:
                            results_dic_score[version_no_seed][kernel].append(kernel_results[kernel])
                        results_dic_mse[version_no_seed].append(val_mse)
    
    if dataset_name == 'ICAfBIRN':
        # Calculate wFNC score
        tr_data, va_data, te_data = train_emb['input'], valid_emb['input'], test_emb['input']
        tr_target, va_target, te_target = train_emb['target'], valid_emb['target'], test_emb['target']
        
        tr_data = torch.cat((tr_data, va_data), dim=0)
        tr_target = torch.cat((tr_target, va_target), dim=0)
        tr_data = tr_data.permute(0, 1, 3, 2)
        te_data = te_data.permute(0, 1, 3, 2)
        
        tr_data = vmap(vmap(torch.corrcoef))(tr_data)
        te_data = vmap(vmap(torch.corrcoef))(te_data)
        
        ix_row, ix_col = torch.triu_indices(config['data_size'], config['data_size'], offset=1)
        tr_data = tr_data[:, :, ix_row, ix_col]
        te_data = te_data[:, :, ix_row, ix_col]
        print(tr_data.size())
        
        num_tr_subjects, tr_num_windows, latent_dim = tr_data.shape
        num_te_subjects, te_num_windows, _ = te_data.shape

        # Repeat class for each context window
        tr_y = np.repeat(
            tr_target[:, np.newaxis], tr_data.shape[1], axis=1)
        te_y = np.repeat(
            te_target[:, np.newaxis], te_data.shape[1], axis=1)

        # Reshape context embeddings
        tr_data = np.reshape(
            tr_data, (num_tr_subjects * tr_num_windows, latent_dim))
        te_data = np.reshape(
            te_data, (num_te_subjects * te_num_windows, latent_dim))

        tr_y = tr_y.flatten()
        te_y = te_y.flatten()

        for context_size in context_sizes[1:]:
            config['dataset'] = 'ICAfBIRN'
            config['model'] = 'wFNC'
            config['local_size'] = 0
            config['context_size'] = context_size
            
            version = generate_version_name(config)
            pca = PCA(n_components=context_size)
            tr_ge = pca.fit_transform(tr_data)
            te_ge = pca.transform(te_data)
            
            # Scale and then predict on test set
            ssc = StandardScaler()
            tr_ge = ssc.fit_transform(tr_ge)
            te_ge = ssc.transform(te_ge)
            kernel_results = {kernel: None for kernel in kernels}
            for kernel in kernels:
                svm = SVC(kernel=kernel)
                svm.fit(tr_ge, tr_y)
                score = svm.score(te_ge, te_y)
                print(dataset_name, 'wFNC', context_size, 
                        kernel, score)
                del svm
                kernel_results[kernel] = score

            version_no_seed = re.sub('_se\d+', '', version)
            if version_no_seed not in results_dic_score:
                results_dic_score[version_no_seed] = {kernel: [kernel_results[kernel]] for kernel in kernels}
                results_dic_mse[version_no_seed] = [val_mse]
            else:
                for kernel in kernels:
                    results_dic_score[version_no_seed][kernel].append(kernel_results[kernel])
                results_dic_mse[version_no_seed].append(val_mse)
            
            del pca

print(results_dic_score)

with open('results/acc_results.json', 'w') as f:
    f.write(json.dumps(results_dic_score))

with open('results/mse_results.json', 'w') as f:
    f.write(json.dumps(results_dic_mse))

with open('classification.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Model", *(f'Acc_{kernel}' for kernel in kernels), "mse"])

    for key in results_dic_mse.keys():
        accs = []
        for kernel in kernels:
            accs.append(np.mean(results_dic_score[key][kernel]))
        mse = np.mean(results_dic_mse[key])
        row = [key, *accs, str(mse)]
        csvwriter.writerow(row)
