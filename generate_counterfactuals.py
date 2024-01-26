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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


config = get_default_config([''])
config['model'] = 'CDSVAE'
config['local_size'] = 2
config['context_size'] = 2
config['seed'] = 1010
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
embed_dict = embed_dataloader(config, model, dm.val_dataloader())

inputs = embed_dict['input']
context_embeddings = embed_dict['context_mean']
targets = embed_dict['target']
y_dict = {
    0: {'name': 'C', 'color': 'b'},
    1: {'name': 'SZ', 'color': 'r'}
}

num_subjects, num_windows, latent_dim = context_embeddings.size()
km = KMeans(n_clusters=2, n_init='auto')
context_embeddings = context_embeddings.view(num_subjects * num_windows, latent_dim)
km.fit(context_embeddings)
print(km.cluster_centers_)

targets = targets[:, np.newaxis]
targets = np.repeat(targets, num_windows, axis=1)
targets = np.reshape(targets, (num_subjects * num_windows, )).numpy()

# Check the percentage of kmeans labels that align
# with the actual diagnosis
labels = km.labels_.copy()
if (labels == targets).sum() / (num_subjects * num_windows) < 0.5:
    labels = (labels == 0).astype(int)

#sz_mean = context_embeddings[(labels == 1) & (targets == 1)].mean(0)
#c_mean = context_embeddings[(labels == 0) & (targets == 0)].mean(0)

# From: https://stackoverflow.com/questions/21660937/get-nearest-point-to-centroid-scikit-learn
closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, context_embeddings)
# Make sure the two windows closest to the cluster centers are from two different classes
assert labels[closest].sum() == 1

print(inputs.size())
num_subjects, num_windows, num_timesteps, data_size = inputs.size()
inputs = inputs.view(num_subjects * num_windows, num_timesteps, data_size)
x = inputs[closest].permute(1, 0, 2).to(model.device)
flipped_closest = closest[::-1].copy()
cf_context = context_embeddings[flipped_closest].to(model.device)
print(cf_context)
model.eval()
with torch.no_grad():
    local, x_hat = model.generate_counterfactual(x, cf_context)
local = local.cpu()
x_hat = x_hat.cpu()
# Shape:
# Local: (num_timesteps, 2, batch_size, local_size)
# X_hat: (num_timesteps, 2, batch_size, data_size)
num_timesteps, _, batch_size, _ = local.size()
fig, axs = plt.subplots(batch_size, 2)
for i in range(batch_size):
    minmax = x_hat[:, :, i].abs().max()
    axs[i, 0].imshow(x_hat[:, 0, i].T, vmin=-minmax, vmax=minmax, cmap='jet')
    axs[i, 1].imshow(x_hat[:, 1, i].T, vmin=-minmax, vmax=minmax, cmap='jet')
axs[0, 0].set_title('Original')
axs[0, 1].set_title('Counterfactual')
plt.savefig('results/counterfactual_rec.png', dpi=400, bbox_inches="tight")
plt.clf()
plt.close(fig)

fig, axs = plt.subplots(batch_size, 2)
for i in range(batch_size):
    minmax = local[:, :, i].abs().max()
    axs[i, 0].plot(local[:, 0, i, 0], local[:, 0, i, 1], c='red' if labels[closest][i] else 'blue')
    axs[i, 1].plot(local[:, 1, i, 0], local[:, 1, i, 1], c='red' if labels[closest][i] else 'blue')
axs[0, 0].set_title('Original')
axs[0, 1].set_title('Counterfactual')
plt.savefig('results/counterfactual_local.png', dpi=400, bbox_inches="tight")
plt.clf()
plt.close(fig)