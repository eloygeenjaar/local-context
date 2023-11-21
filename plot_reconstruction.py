import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from lib.utils import get_default_config
plt.rcParams['font.size'] = 12

config = get_default_config([''])
local_sizes = [4, 8, 16]
global_sizes = [2, 4, 8, 16]

ls = []
fig, axs = plt.subplots(len(local_sizes), 2, figsize=(12, 12))
for (l_ix, local_size) in enumerate(local_sizes):
    for (g_ix, global_size) in enumerate(global_sizes):
        version = f'm{config["model"]}_d{config["dataset"]}_g{config["gamma"]}_s{config["seed"]}_n{config["normalization"]}_s{local_size}_g{global_size}_f{config["fold_ix"]}'
        df = pd.read_csv(Path('lightning_logs') / Path(version) / 'metrics.csv')
        tr_df = df.dropna(axis=0, subset='tr_loss_epoch', inplace=False)
        va_df = df.dropna(axis=0, subset='va_loss', inplace=False)
        print(va_df['va_loss'].values)
        epochs = va_df['va_loss'].values.shape[0]
        print(np.arange(epochs)[10:])
        l, = axs[l_ix, 0].plot(np.arange(epochs)[10:], va_df['va_loss'].values[10:], alpha=0.5, linewidth=2)
        axs[l_ix, 1].plot(np.arange(epochs)[10:], tr_df['tr_loss_epoch'].values[10:], alpha=0.5, linewidth=2)
        axs[l_ix, 0].set_title(f'Val - Local size: {local_size}')
        axs[l_ix, 1].set_title(f'Train - Local size: {local_size}')
        axs[l_ix, 0].set_ylabel('Mean squared error')
        if l_ix == 0:
            ls.append(l)
axs[l_ix, 0].set_xlabel('Epochs')
axs[l_ix, 1].set_xlabel('Epochs')
plt.figlegend(ls, global_sizes, loc = 'upper center', ncol=len(global_sizes))
plt.savefig('figures/reconstruction_results.png')
plt.clf()
plt.close(fig)