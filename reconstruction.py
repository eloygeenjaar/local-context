import torch
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.utils import get_default_config, mask_input, mask_input_extra
from torch.utils.data import DataLoader

config = get_default_config([''])
data_module = importlib.import_module('lib.data')
dataset_type = getattr(data_module, config['dataset'])
valid_dataset = dataset_type('valid', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'], window_step=20)

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
    x, task_mask, mask, y = batch
    x = x.to(device, non_blocking=True)
    task_mask = task_mask.to(device, non_blocking=True)
    x_masked = mask_input_extra(x, task_mask, window_step=32)
    x_masked = x_masked.float()
    with torch.no_grad():
        model_output = model(x_masked, x.float())        
    break

#window_size, batch_size, num_windows, input_size = model_output['x_hat'].size()
#print(model_output['x_hat'].size())
#rec = torch.reshape(model_output['x_hat'], (window_size, batch_size * num_windows, input_size))
#rec = rec.cpu().numpy()
#org = torch.reshape(x_masked, (window_size, batch_size * num_windows, input_size))
#org = org.cpu().numpy()
#fig, axs = plt.subplots(16, 8, figsize=(8, 16))
#for i in range(16):
#    for j in range(8):
#        axs[i, j].plot(rec[:, j, i], c='r', alpha=0.75)
#        axs[i, j].plot(org[:, j, i], c='b', alpha=0.5)
#plt.tight_layout()
#plt.savefig('figures/fbirn_rec.png', dpi=400)
#plt.clf()
#plt.close(fig)

print(valid_dataset.df.iloc[2])
window_size, batch_size, num_windows, input_size = model_output['x_hat'].size()
print(model_output['x_hat'].size())
rec = torch.reshape(model_output['x_hat'].permute(2, 0, 1, 3), (-1, batch_size, input_size))
rec = rec.cpu().numpy()
org = torch.reshape(x_masked.permute(2, 0, 1, 3), (-1, batch_size, input_size))
org = org.cpu().numpy()
fig, axs = plt.subplots(4, 2, figsize=(8, 16))
for i in range(4):
    axs[i, 0].imshow(org[:, i, :].T, cmap='jet')
    axs[i, 1].imshow(rec[:, i, :].T, cmap='jet')
plt.tight_layout()
plt.savefig('figures/rec.png', dpi=400)
plt.clf()
plt.close(fig)