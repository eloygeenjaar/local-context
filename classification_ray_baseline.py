import torch
import importlib
import numpy as np
from lib.utils import (
    get_default_config, embed_all, generate_version_name, init_data_module, init_model, load_hyperparameters)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
from lib.visualizations import visualize_space
import csv


config = get_default_config([''])
seeds = [42, 1337]
models = ['CDSVAE', 'CO', 'DSVAE', 'IDSVAE', 'LVAE']
local_sizes = [0, 2]
context_sizes = [0, 2]
config = config.copy()

config['seed'] = 42
config['local_size'] = 2
config['model'] = 'CDSVAE'
config['context_size'] = 2


version = generate_version_name(config)
result_p = Path(f'ray_results/{version}')
ckpt_p = result_p / 'final.ckpt'
hyperparameters = load_hyperparameters(result_p / 'params.json')
model = init_model(config, hyperparameters, viz=False, ckpt_path=ckpt_p)
dm = init_data_module(config)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model.eval()

dataloaders = embed_all(config, model, dm)
train_dataloader, val_dataloader, test_dataloader = dataloaders['train'], dataloaders['valid'], dataloaders['test']

tr_ge, tr_y_orig = train_dataloader['input'], train_dataloader['target']
va_ge, va_y_orig = val_dataloader['input'], val_dataloader['target']

num_tr_subjects, tr_num_windows, window_size, data_size = tr_ge.shape
num_va_subjects, va_num_windows, _, _= va_ge.shape

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
svm = SVC(kernel='linear')
svm.fit(tr_ge, tr_y)
score = svm.score(va_ge, va_y)
print("Baseline (original input): " , score)


with open('classification.csv', 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    row = ["Baseline (original input)", "Linear", str(score), "NA"]
    csvwriter.writerow(row)