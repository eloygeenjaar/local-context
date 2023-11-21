import torch
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from lib.utils import get_default_config

config = get_default_config([''])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
version = f'm{config["model"]}_d{config["dataset"]}_g{config["gamma"]}_s{config["seed"]}_n{config["normalization"]}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_module = importlib.import_module('lib.model')
model_type = getattr(model_module, config['model'])
model = model_type.load_from_checkpoint(f'lightning_logs/{version}/checkpoints/best.ckpt')
model = model.to(device)
model.eval()
data_module = importlib.import_module('lib.data')
dataset_type = getattr(data_module, config['dataset'])
train_dataset = dataset_type('train', config["normalization"], config["seed"], fold_ix=0, num_folds=10)
valid_dataset = dataset_type('valid', config["normalization"], config["seed"], fold_ix=0, num_folds=10)
assert train_dataset.data_size == valid_dataset.data_size
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False,
                            num_workers=5)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False,
                            num_workers=5)

ground_truths_tr = []
factors_tr = []
for (i, batch) in enumerate(train_dataloader):
    x, mask, (y_local, y_global) = batch
    x = x.to(device, non_blocking=True)
    with torch.no_grad():
        #p_x_hat, local_dist, global_dist, z_t, z_g = model(x, mask)
        p_x_hat, global_dist, z = model(x, mask)
    break

gd = global_dist.mean[:x.size(0)].cpu().numpy()
x = x.cpu().numpy()
rec = p_x_hat.mean.cpu().numpy()

print(gd.shape)
#y_uni = np.unique(y_global)
y_uni = y_global

fig, axs = plt.subplots(2,2,figsize=(10,5))
for i in range(2):
    for j in range(2):
        ix = i * 2 + j
        #for k in range(20):
            #axs[i, j].plot(rec[y_global==y_uni[ix], :, k][0], alpha=0.5, c='b')
        axs[i, j].plot(rec[y_global==y_uni[ix]][0].mean(-1), c='b', alpha=0.5)
        axs[i, j].plot(x[y_global==y_uni[ix]][0].mean(-1), c='r')
plt.savefig(f'figures/visualization_{version}.png')
plt.clf()
plt.close(fig)

tsne = TSNE(n_components=2)
gd_tsne = tsne.fit_transform(gd)

fig, ax = plt.subplots(1,1,figsize=(10,10))
cmap = plt.get_cmap('inferno')
for i in range(2):
    ax.scatter(gd_tsne[y_global==i, 0], gd_tsne[y_global==i, 1], color=cmap(i / 12))
plt.savefig(f'figures/class_sim.png')
plt.clf()
plt.close(fig)