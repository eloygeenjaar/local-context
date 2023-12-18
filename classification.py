import umap
import torch
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import misvm
from lib.utils import get_default_config, embed_global_data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

config = get_default_config([''])
seeds = [42, 1337, 9999, 1212]
models = ['CDSVAE', 'DSVAE']
local_sizes = [2, 4]
global_sizes = [2, 4, 8, 16]
config = config.copy()
for seed in seeds:
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    for model_name in models:
        for local_size in local_sizes:
            for global_size in global_sizes:
                config['seed'] = seed
                config['local_size'] = local_size
                config['model'] = model_name
                config['global_size'] = global_size
                version = f'm{config["model"]}_d{config["dataset"]}_s{config["seed"]}_s{config["local_size"]}_g{config["global_size"]}_nl{config["num_layers"]}'
                data_module = importlib.import_module('lib.data')
                dataset_type = getattr(data_module, config['dataset'])
                train_dataset = dataset_type('train', config['seed'])
                valid_dataset = dataset_type('valid', config['seed'])
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_module = importlib.import_module('lib.model')
                model_type = getattr(model_module, config['model'])
                model = model_type.load_from_checkpoint(f'lightning_logs/{version}/checkpoints/best.ckpt')
                model = model.to(device)
                model.eval()
                
                tr_ge, tr_y_orig = embed_global_data(device, model, train_dataset)
                va_ge, va_y_orig = embed_global_data(device, model, valid_dataset)

                classifier = misvm.MISVM(kernel='linear', C=10, max_iters=1000, verbose=False)
                tr_y_orig = tr_y_orig.numpy()
                va_y_orig = va_y_orig.numpy()
                tr_y_bag = (tr_y_orig * 2 - 1).astype(int)
                va_y_bag = (va_y_orig * 2 - 1).astype(int)
                classifier.fit(tr_ge, tr_y_bag)
                bag_labels, instances = classifier.predict(va_ge, instancePrediction=True)
                print(va_y_orig)
                print(np.sign(bag_labels))
                print(np.sign(classifier.predict(tr_ge)))
                print(f'Bags: {accuracy_score(bag_labels >=0, va_y_bag>=0)}')
                print(np.sign(instances).shape)

                num_tr_subjects, tr_num_windows, latent_dim = tr_ge.shape
                num_va_subjects, va_num_windows, _ = va_ge.shape

                tr_y = np.repeat(tr_y_orig[:, np.newaxis], tr_ge.shape[1], axis=1)
                va_y = np.repeat(va_y_orig[:, np.newaxis], va_ge.shape[1], axis=1)

                tr_ge = np.reshape(tr_ge, (num_tr_subjects * tr_num_windows, latent_dim))
                va_ge = np.reshape(va_ge, (num_va_subjects * va_num_windows, latent_dim))

                tr_y = tr_y.flatten()
                va_y = va_y.flatten()

                ssc = StandardScaler()
                tr_ge = ssc.fit_transform(tr_ge)
                va_ge = ssc.transform(va_ge)
                svm = SVC(kernel='rbf')
                svm.fit(tr_ge, tr_y)
                print(model_name, global_size, svm.score(va_ge, va_y))
                
                tr_ge = np.reshape(tr_ge, (num_tr_subjects, tr_num_windows, latent_dim))
                va_ge = np.reshape(va_ge, (num_va_subjects, va_num_windows, latent_dim))
                quit()