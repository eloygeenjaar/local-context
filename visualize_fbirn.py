import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from lib.utils import get_default_config, embed_data

config = get_default_config([''])
data_module = importlib.import_module('lib.data')
dataset_type = getattr(data_module, config['dataset'])
train_dataset = dataset_type('train', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'])
valid_dataset = dataset_type('valid', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'])
test_dataset = dataset_type('test', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'])

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

((x_train, x_tr_time, y_train),
 (x_valid, x_va_time, y_valid),
 (x_test, x_te_time, y_test)) = embed_data(device, model, train_dataset, valid_dataset, test_dataset)


train_subjects = x_train.shape[0]
print(x_train.min(0), x_train.max(0), x_train.mean(0), x_train.std(0))

x = np.concatenate((x_train, x_valid), axis=0)
y = np.concatenate((y_train, y_valid), axis=0)

kmeans = KMeans(n_clusters=2, random_state=config['seed'], n_init='auto')
kmeans.fit(x_train)
labels = kmeans.predict(x) + 1
if np.sum(labels == y) > np.sum(labels != y):
    c_mean = x[(labels == 1) & (y == 1)].mean(0)
    sz_mean = x[(labels == 2) & (y == 2)].mean(0)
else:
    c_mean = x[(labels == 2) & (y == 1)].mean(0)
    sz_mean = x[(labels == 1) & (y == 2)].mean(0)

if x.shape[-1] > 2:
    tsne = PCA(n_components=2)
    x_tsne = np.concatenate((x, c_mean[np.newaxis], sz_mean[np.newaxis]))
    x_tsne = tsne.fit_transform(x_tsne)
    x_train_tsne = x_tsne[:train_subjects]
    x_valid_tsne = x_tsne[train_subjects:-2]
    c_mean_tsne = x_tsne[-2]
    sz_mean_tsne = x_tsne[-1]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(x_train_tsne[y_train==1, 0], x_train_tsne[y_train==1, 1], alpha=0.25, color='r')
ax.scatter(x_train_tsne[y_train==2, 0], x_train_tsne[y_train==2, 1], alpha=0.25, color='b')
ax.scatter(x_valid_tsne[y_valid==1, 0], x_valid_tsne[y_valid==1, 1], alpha=0.25, color='r', marker='D')
ax.scatter(x_valid_tsne[y_valid==2, 0], x_valid_tsne[y_valid==2, 1], alpha=0.25, color='b', marker='D')
ax.scatter(c_mean_tsne[0], c_mean_tsne[1], color='k', s=50, alpha=1.0)
ax.scatter(sz_mean_tsne[0], sz_mean_tsne[1], color='k', s=50, alpha=1.0)
plt.savefig('figures/global_visualization.png')
plt.clf()
plt.close(fig)

x_tr_time_flat = np.reshape(x_tr_time, (np.product(x_tr_time.shape[:2]), x_tr_time.shape[-1]))
x_va_time_flat = np.reshape(x_va_time, (np.product(x_va_time.shape[:2]), x_va_time.shape[-1]))
scores, clusters = [], []
n_clusters_ls = list(range(2, 21))
for n_clusters in n_clusters_ls:
    kmeans = KMeans(n_clusters=n_clusters, random_state=config['seed'], n_init='auto')
    kmeans.fit(x_tr_time_flat)
    labels = kmeans.predict(x_va_time_flat)
    scores.append(silhouette_score(x_va_time_flat, labels, metric='euclidean'))
    clusters.append(kmeans.cluster_centers_)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(n_clusters_ls, scores, alpha=0.75, linewidth=5)
ax.set_xticks(n_clusters_ls[::2])
plt.savefig('figures/clustering_scores.png')
plt.clf()
plt.close(fig)

num_clusters = 6

if x_tr_time.shape[-1] > 2:
    tsne = PCA(n_components=2)
    train_subjects, num_timesteps, latent_dim = x_tr_time.shape
    valid_subjects = x_va_time.shape[0]
    x_time_tsne = np.concatenate((x_tr_time_flat, x_va_time_flat, clusters[num_clusters - 2]))
    x_time_tsne = tsne.fit_transform(x_time_tsne)
    x_tr_time_tsne = x_time_tsne[:(train_subjects * num_timesteps)]
    x_tr_time_tsne = np.reshape(x_tr_time_tsne, (train_subjects, num_timesteps, 2))
    x_va_time_tsne = x_time_tsne[(train_subjects * num_timesteps):-num_clusters]
    x_va_time_tsne = np.reshape(x_va_time_tsne, (valid_subjects, num_timesteps, 2))
    clusters_tsne = x_tsne[-num_clusters:]


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

x_tr_time_c = x_tr_time_tsne[y_train==1]
x_tr_time_c = np.reshape(x_tr_time_c, (np.product(x_tr_time_c.shape[:2]), x_tr_time_c.shape[-1]))
x_tr_time_sz = x_tr_time_tsne[y_train==2]
x_tr_time_sz = np.reshape(x_tr_time_sz, (np.product(x_tr_time_sz.shape[:2]), x_tr_time_sz.shape[-1]))
x_va_time_c = x_va_time_tsne[y_valid==1]
x_va_time_c = np.reshape(x_va_time_c, (np.product(x_va_time_c.shape[:2]), x_va_time_c.shape[-1]))
x_va_time_sz = x_va_time_tsne[y_valid==2]
x_va_time_sz = np.reshape(x_va_time_sz, (np.product(x_va_time_sz.shape[:2]), x_va_time_sz.shape[-1]))
ax.scatter(x_tr_time_c[:, 0], x_tr_time_c[:, 1], alpha=0.25, color='r')
ax.scatter(x_tr_time_sz[:, 0], x_tr_time_sz[:, 1], alpha=0.25, color='b')
ax.scatter(x_va_time_c[:, 0], x_va_time_c[:, 1], alpha=0.25, color='r', marker='D')
ax.scatter(x_va_time_sz[:, 0], x_va_time_sz[:, 1], alpha=0.25, color='b', marker='D')
ax.scatter(clusters_tsne[:, 0], clusters_tsne[:, 1], color='k', s=50, alpha=1.0)
plt.savefig('figures/local_visualization.png')
plt.clf()
plt.close(fig)

local_clusters = clusters[num_clusters - 2]
fig, ax = plt.subplots(3, num_clusters, figsize=((2 * num_clusters), 15))
z_t = torch.from_numpy(local_clusters).float().to(device).unsqueeze(1)
z_g_c = torch.from_numpy(c_mean).float().to(device).unsqueeze(0).repeat(num_clusters, 1)
z_g_sz = torch.from_numpy(sz_mean).float().to(device).unsqueeze(0).repeat(num_clusters, 1)
with torch.no_grad():
    rec_c = model.decoder(z_t, z_g_c, output_len=model.window_size)
    rec_sz = model.decoder(z_t, z_g_sz, output_len=model.window_size)
print(rec_c.size(), rec_sz.size())
for cluster_ix in range(num_clusters):
    rec_c_np = rec_c[cluster_ix].cpu().numpy()
    rec_sz_np = rec_sz[cluster_ix].cpu().numpy()
    print(np.abs(rec_c_np - rec_sz_np).mean())
    diff = rec_c_np - rec_sz_np
    vminmax = np.array([np.abs(rec_c_np).max(), np.abs(rec_sz_np).max()]).max()
    ax[0, cluster_ix].imshow(rec_c_np.T, vmin=-vminmax, vmax=vminmax, cmap='jet')
    ax[1, cluster_ix].imshow(rec_sz_np.T, vmin=-vminmax, vmax=vminmax, cmap='jet')
    ax[2, cluster_ix].imshow(diff.T, cmap='jet')
    ax[2, cluster_ix].set_xlabel('Time ->')
    ax[0, cluster_ix].set_title(f'Control brain state: {cluster_ix}')
    ax[1, cluster_ix].set_title(f'SZ brain state: {cluster_ix}')
    ax[2, cluster_ix].set_title(f'Diff brain state: {cluster_ix}')
ax[0, 0].set_ylabel('ICA components')
ax[1, 0].set_ylabel('ICA components')
plt.tight_layout()

plt.savefig('figures/reconstruction.png')
plt.clf()
plt.close(fig)

fig, ax = plt.subplots(3, num_clusters, figsize=((5 * num_clusters), 15))
for cluster_ix in range(num_clusters):
    rec_c_np = rec_c[cluster_ix].cpu().numpy()
    dfnc_c = np.corrcoef(rec_c_np.T)
    rec_sz_np = rec_sz[cluster_ix].cpu().numpy()
    dfnc_sz = np.corrcoef(rec_sz_np.T)
    print(np.abs(dfnc_c - dfnc_sz).mean())
    diff = dfnc_c - dfnc_sz
    vminmax = np.array([np.abs(dfnc_c).max(), np.abs(dfnc_sz).max()]).max()
    ax[0, cluster_ix].imshow(dfnc_c, vmin=-vminmax, vmax=vminmax, cmap='jet')
    ax[1, cluster_ix].imshow(dfnc_sz, vmin=-vminmax, vmax=vminmax, cmap='jet')
    ax[2, cluster_ix].imshow(diff, cmap='jet')
    ax[2, cluster_ix].set_xlabel('ICA components')
    ax[0, cluster_ix].set_title(f'Control dFNC state: {cluster_ix}')
    ax[1, cluster_ix].set_title(f'SZ dFNC state: {cluster_ix}')
    ax[2, cluster_ix].set_title(f'C - SZ dFNC state: {cluster_ix}')
ax[0, 0].set_ylabel('ICA components')
ax[1, 0].set_ylabel('ICA components')
plt.tight_layout()

plt.savefig('figures/state_fncs.png')
plt.clf()
plt.close(fig)