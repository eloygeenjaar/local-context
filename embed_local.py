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
from sklearn.metrics import silhouette_score
plt.rcParams['font.size'] = 20

config = get_default_config([''])
data_module = importlib.import_module('lib.data')
dataset_type = getattr(data_module, config['dataset'])
train_dataset = dataset_type('train', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'], window_step=20)
valid_dataset = dataset_type('valid', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'], window_step=20)
test_dataset = dataset_type('test', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'], window_step=20)

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

print(x_tr_time.shape)
x_tr_time_shared = x_tr_time[..., :model.shared]
x_tr_time_ns = x_tr_time[..., model.shared:]
#x = np.reshape(x_tr_time_shared, (-1, x_tr_time_shared.shape[-1]))

#n_clusters_ls = list(range(2, 21))
#scores = []
#for n_clusters in n_clusters_ls:
#    kmeans = KMeans(n_clusters=n_clusters, random_state=config['seed'], n_init='auto')
#    kmeans.fit(x)
#    labels = kmeans.predict(x)
#    scores.append(silhouette_score(x, labels))
#    print(f'Num clusters: {n_clusters}, {scores[-1]}')

#if x.shape[-1] > 2:
#    tsne = TSNE(n_components=2)
#    x_tsne = tsne.fit_transform(x)
#else:
#    x_tsne = x

#x_tsne_subj = np.reshape(x_tsne, (len(train_dataset), -1, 2))
#print(x_tsne_subj.shape)
#fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#ax.scatter(x_tsne[:, 0], x_tsne[:, 1], alpha=0.1, color='b', s=10)
#cmap = plt.get_cmap('jet')
#for i in range(len(train_dataset)):
#    ax.scatter(x_tsne_subj[i, :, 0], x_tsne_subj[i, :, 1], alpha=0.2, c=cmap(np.linspace(0.0, 1.0, x_tsne_subj.shape[1])), s=20)
    #ax.scatter(x_tsne_subj[i, :, 0], x_tsne_subj[i, :, 1], alpha=0.2, color=cmap(i / len(train_dataset)), s=20)
#ax.set_xlabel('Latent dimension 1')
#ax.set_ylabel('Latent dimension 2')
#plt.savefig('figures/local_viz.png')
#plt.clf()
#plt.close(fig)

x = np.reshape(x_tr_time_ns, (-1, x_tr_time_ns.shape[-1]))

#n_clusters_ls = list(range(2, 21))
#scores = []
#for n_clusters in n_clusters_ls:
#    kmeans = KMeans(n_clusters=n_clusters, random_state=config['seed'], n_init='auto')
#    kmeans.fit(x)
#    labels = kmeans.predict(x)
#    scores.append(silhouette_score(x, labels))
#    print(f'Num clusters: {n_clusters}, {scores[-1]}')

#if x.shape[-1] > 2:
#    tsne = TSNE(n_components=2)
#    x_tsne = tsne.fit_transform(x)
#else:
#    x_tsne = x

x_tsne_subj = np.reshape(x, (len(train_dataset), -1, x.shape[-1]))
print(x_tsne_subj.shape)
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
#ax.scatter(x_tsne[:, 0], x_tsne[:, 1], alpha=0.1, color='b', s=10)
cmap = plt.get_cmap('jet')
for i in range(10):
    #ax.scatter(np.arange(x_tsne_subj.shape[-1]), x_tsne_subj[i, :], alpha=0.2, c=cmap(np.linspace(0.0, 1.0, x_tsne_subj.shape[1])), s=20)
    for j in range(4):
        for k in range(4):
            ix = j * 4 + k
            axs[j, k].plot(np.arange(x_tsne_subj.shape[1]), x_tsne_subj[i, :, ix], alpha=0.2, color=cmap(i / 10))
            axs[j, k].set_xlabel('Latent dimension 1')
            axs[j, k].set_ylabel('Latent dimension 2')
plt.savefig('figures/local_viz_ns.png')
plt.clf()
plt.close(fig)