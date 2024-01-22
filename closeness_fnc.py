import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lib.utils import (
    get_default_config, load_hyperparameters,
    generate_version_name, init_model,
    init_data_module, embed_dataloader)
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


config = get_default_config([''])
config['model'] = 'CDSVAE'
config['local_size'] = 2
config['context_size'] = 2
config['seed'] = 42
version = generate_version_name(config)
result_p = Path(f'ray_results/{version}')
ckpt_p = result_p / 'final.ckpt'
hyperparameters = load_hyperparameters(result_p / 'params.json')
model = init_model(config, hyperparameters, viz=False, ckpt_path=ckpt_p)
dm = init_data_module(config)
# The shape out of embed_ functions
# will be:
# (num_subjects, num_windows, ...)
# Where ... depends on the variable we are trying
# to obtain. 
embed_dict = embed_dataloader(config, model, dm.train_dataloader())

original_inputs = embed_dict['input']
context_embeddings = embed_dict['context_mean']

num_subjects, num_windows, window_size, data_size = original_inputs.size()
fncs = torch.empty((num_subjects, num_windows, data_size, data_size))
for subject_ix in range(num_subjects):
    for window_ix in range(num_windows):
        fnc = torch.corrcoef(
            original_inputs[subject_ix, window_ix].permute(1, 0))
        fncs[subject_ix, window_ix] = fnc

fnc_ix_l, fnc_ix_r = torch.triu_indices(data_size, data_size, 1)
fncs = fncs[..., fnc_ix_l, fnc_ix_r]
fnc_features = fncs.size(-1)

fncs = torch.reshape(fncs, (num_subjects * num_windows, fnc_features))
ces = torch.reshape(context_embeddings, (num_subjects * num_windows, -1))

window_ix_l, window_ix_r = torch.triu_indices(
    num_subjects * num_windows, num_subjects * num_windows, 1)

# Calculate distances in the feature spaces
fnc_dist = torch.cdist(fncs, fncs, p=2)[window_ix_l, window_ix_r]
ces_dist = torch.cdist(ces, ces, p=2)[window_ix_l, window_ix_r]

# Normalize and then fit a linear model ax + b to predict 
# the context embedding distances from the FNC distances
fnc_ssc = StandardScaler(with_mean=True)
fnc_dist_s = fnc_ssc.fit_transform(fnc_dist[:, np.newaxis])
ces_ssc = StandardScaler(with_mean=True)
ces_dist_s = ces_ssc.fit_transform(ces_dist[:, np.newaxis])
lr = LinearRegression()
print(lr.fit(fnc_dist_s, ces_dist_s).score(
      fnc_dist_s, ces_dist_s))
