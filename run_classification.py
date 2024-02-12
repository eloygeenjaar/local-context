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
from glob import glob
from sklearn.decomposition import PCA

config = get_default_config([''])
kernels = ['linear', 'rbf']
config = config.copy()

results_dic_score = {}
results_dic_mse = {}
baseline_scores = []
baseline_wFNC_scores = []

runs = glob('ray_results/*')

for i, run in enumerate(runs):
    run = run.split("/")[1]
    config_list = run.split("_")
    config['model'] = config_list[0][1:]
    config['dataset'] = config_list[1][1:]
    config['seed'] = int(config_list[2][2:])
    config['local_size'] = int(config_list[3][2:])
    config['context_size'] = int(config_list[4][2:])

    version = generate_version_name(config)
    result_p = Path(f'ray_results/{version}')
    ckpt_p = result_p / 'final.ckpt'
    dm = init_data_module(config)

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
    scores = []
    for kernel in kernels:
        svm = SVC(kernel=kernel)
        svm.fit(tr_ge, tr_y)
        score = svm.score(va_ge, va_y)
        print(config['dataset'], config['model'], config['context_size'], 
                kernel, score)
        scores.append(score)
        del svm

    version_no_seed = re.sub('_se\d+', '', version, count = 1)
    if version_no_seed not in results_dic_score:
        results_dic_score[version_no_seed] = [scores]
        results_dic_mse[version_no_seed] = [val_mse]
    else:
        results_dic_score[version_no_seed].append(scores)
        results_dic_mse[version_no_seed].append(val_mse)

    if i == len(runs) - 1: #calculate baselines
        #1. Original Input
        tr_ge, tr_y_orig = train_dataloader['input'], train_dataloader['target']
        va_ge, va_y_orig = val_dataloader['input'], val_dataloader['target']

        num_tr_subjects, tr_num_windows, window_size, data_size = tr_ge.shape
        num_va_subjects, va_num_windows, _, _= va_ge.shape

        #2. wFNC with PCA applied
        fncs = torch.empty((num_tr_subjects, tr_num_windows, data_size, data_size))
        for subject_ix in range(num_tr_subjects):
            for window_ix in range(tr_num_windows):
                fnc = torch.corrcoef(
                    tr_ge[subject_ix, window_ix].permute(1, 0))
                fncs[subject_ix, window_ix] = fnc

        fnc_ix_l, fnc_ix_r = torch.triu_indices(data_size, data_size, 1)
        fncs = fncs[..., fnc_ix_l, fnc_ix_r]
        fnc_features = fncs.size(-1)

        fncs = torch.reshape(fncs, (num_tr_subjects * tr_num_windows, fnc_features))

        pca = PCA(n_components=2, svd_solver = 'full')
        fncs_pca = pca.fit_transform(fncs)

        fncs_val = torch.empty((num_va_subjects, va_num_windows, data_size, data_size))
        for subject_ix in range(num_va_subjects):
            for window_ix in range(va_num_windows):
                fnc = torch.corrcoef(
                    va_ge[subject_ix, window_ix].permute(1, 0))
                fncs_val[subject_ix, window_ix] = fnc

        fnc_ix_l, fnc_ix_r = torch.triu_indices(data_size, data_size, 1)
        fncs_val = fncs_val[..., fnc_ix_l, fnc_ix_r]
        fnc_features = fncs_val.size(-1)

        fncs_val = torch.reshape(fncs_val, (num_va_subjects * va_num_windows, fnc_features))

        fncs_val_pca = pca.transform(fncs_val)

        # Repeat class for each context window
        tr_y = np.repeat(
            tr_y_orig[:, np.newaxis], tr_ge.shape[1], axis=1)
        va_y = np.repeat(
            va_y_orig[:, np.newaxis], va_ge.shape[1], axis=1)

        # Reshape context embeddings
        tr_ge = np.reshape(
            tr_ge, (num_tr_subjects * tr_num_windows, window_size * data_size))
        va_ge = np.reshape(
            va_ge, (num_va_subjects * va_num_windows, window_size * data_size))

        tr_y = tr_y.flatten()
        va_y = va_y.flatten()

        # Scale and then predict on test set
        ssc = StandardScaler()
        tr_ge = ssc.fit_transform(tr_ge)
        va_ge = ssc.transform(va_ge)

        ssc2 = StandardScaler()
        fncs_pca = ssc2.fit_transform(fncs_pca)
        fncs_val_pca = ssc2.transform(fncs_val_pca)


        for kernel in kernels:
            svm = SVC(kernel=kernel)
            svm.fit(tr_ge, tr_y)
            score = svm.score(va_ge, va_y)
            print(config['dataset'], config['model'], config['context_size'], 
                    kernel, score)
            baseline_scores.append(score)
            del svm
        for kernel in kernels:
            svm = SVC(kernel=kernel)
            svm.fit(fncs_pca, tr_y)
            score = svm.score(fncs_val_pca, va_y)
            print(config['dataset'], config['model'], config['context_size'], 
                    kernel, score)
            baseline_wFNC_scores.append(score)
            del svm

with open('classification3.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Model", "ACC_Linear", "ACC_RBF", "MSE"])
    for key in results_dic_mse.keys():
        averages = np.average(results_dic_score[key], axis = 0)
        mse = np.average(results_dic_mse[key])
        row = [key, str(averages[0]), str(averages[1]), str(mse)]
        csvwriter.writerow(row)
    row = ["Baseline (original input)", str(baseline_scores[0]), str(baseline_scores[1]), "NA"]
    csvwriter.writerow(row)
    row = ["Baseline (wFNC)", str(scores[0]), str(scores[1]), "NA"]
    csvwriter.writerow(row)