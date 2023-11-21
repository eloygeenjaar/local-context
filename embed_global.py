import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from lib.utils import get_default_config, embed_global_data, embed_sfncs
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
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

((x_train, y_train),
 (x_valid, y_valid),
 (x_test, y_test)) = embed_global_data(device, model, train_dataset, valid_dataset, test_dataset)

#sfnc_train, sfnc_valid, sfnc_test = embed_sfncs(device, model, train_dataset, valid_dataset, test_dataset)
#pca = PCA(n_components=8)
#sfnc_train = pca.fit_transform(sfnc_train)

print(x_train.shape)
x = np.reshape(x_train, (-1, model.global_size))

#n_clusters_ls = list(range(2, 21))
#scores = []
#for n_clusters in n_clusters_ls:
#    kmeans = KMeans(n_clusters=n_clusters, random_state=config['seed'], n_init='auto')
#    kmeans.fit(x)
#    labels = kmeans.predict(x)
#    scores.append(silhouette_score(x, labels))
#    print(f'Num clusters: {n_clusters}, {scores[-1]}')

if x.shape[-1] > 3:
    tsne = TSNE(n_components=3)
    x_tsne = tsne.fit_transform(x)
else:
    x_tsne = x

x_tsne_subj = np.reshape(x_tsne, (len(train_dataset), -1, 3))
print(x_tsne_subj.shape)
print(x_tsne_subj.shape)

#cols = ['Age', 'Gender', 'WM_Task_Acc', 'WM_Task_Median_RT']
#for col in cols:
#    y_train = train_dataset.df[col].values.copy()
#    nans = np.isnan(y_train)
#    y_train = y_train[~nans]
#    x_tr = x_train.mean(1)[~nans].copy()
#    sc = StandardScaler(with_mean=True)
#    x_tr = sc.fit_transform(x_tr)
#    sfnc_tr = sfnc_train[~nans].copy()
#    lr = LinearRegression()
#    lr.fit(x_tr, y_train)
#    print(col, lr.score(x_tr, y_train), pearsonr(lr.predict(x_tr), y_train)[0])
#    del lr
#    del sc
#    lr = LinearRegression()
#    sc = StandardScaler(with_mean=True)
#    sfnc_tr = sc.fit_transform(sfnc_tr)
#    print(col, lr.fit(sfnc_tr, y_train).score(sfnc_tr, y_train))

fig, ax = plt.subplots(1, 1, figsize=(10, 10),subplot_kw=dict(projection='3d'))
cmap = plt.get_cmap('jet')
for i in range(50):
    #ax.scatter(x_tsne_subj[i, :, 0], x_tsne_subj[i, :, 1], alpha=0.2, c=cmap(np.linspace(0.2, 1., x_tsne_subj.shape[1])), s=20)
    ax.plot(x_tsne_subj[i, 20:100, 0], x_tsne_subj[i, 20:100, 1], x_tsne_subj[i, 20:100, 2], alpha=0.4, c=cmap(i / 50))
    ax.scatter(x_tsne_subj[i, 20:100, 0], x_tsne_subj[i, 20:100, 1], x_tsne_subj[i, 20:100, 2], alpha=0.5, c=cmap(i / 50), s=20)
ax.set_xlabel('Latent dimension 1')
ax.set_ylabel('Latent dimension 2')
plt.savefig('figures/global_viz.png')
plt.clf()
plt.close(fig)
