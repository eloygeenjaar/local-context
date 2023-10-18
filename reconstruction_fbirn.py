import torch
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.utils import get_default_config
from torch.utils.data import DataLoader

config = get_default_config([''])
data_module = importlib.import_module('lib.data')
dataset_type = getattr(data_module, config['dataset'])
valid_dataset = dataset_type('valid', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'])

config = get_default_config([''])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
version = f'm{config["model"]}_d{config["dataset"]}_g{config["gamma"]}_s{config["seed"]}_n{config["normalization"]}_s{config["local_size"]}_g{config["global_size"]}_f{config["fold_ix"]}'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_module = importlib.import_module('lib.model')
model_type = getattr(model_module, config['model'])
model = model_type.load_from_checkpoint(f'lightning_logs/{version}/checkpoints/best.ckpt')
model = model.to(device)
model.eval()

valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False,
                              num_workers=5)
for (i, batch) in enumerate(valid_dataloader):
    x, mask, (y_local, y_global) = batch
    x = x.to(device, non_blocking=True)
    with torch.no_grad():
        p_x_hat, local_dist, global_dist, z_t, z_g = model(x, mask)        
    break

rec = p_x_hat.mean.cpu().numpy()
org = x.cpu().numpy()
fig, axs = plt.subplots(16, 8, figsize=(8, 16))
for i in range(16):
    for j in range(8):
        axs[i, j].plot(rec[j, :, i], c='r', alpha=0.75)
        axs[i, j].plot(org[j, :, i], c='b', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/fbirn_rec.png', dpi=400)
plt.clf()
plt.close(fig)