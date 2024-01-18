import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

def prep_viz(x, num_subjects):
    # We assume that x comes in the following shape:
    # (num_subjects, num_timesteps, latent_dimensions)
    if num_subjects == -1:
        num_subjects = x.shape[0]
    pca = False
    assert len(x.shape) == 3 or len(x.shape) == 4
    latent_dimensions = x.shape[-1]
    x = x[:num_subjects]
    if len(x.shape) == 4:
        x = np.reshape(x, (num_subjects, -1, latent_dimensions))
    num_timesteps = x.shape[1]
    if latent_dimensions > 2:
        x_flat = np.reshape(x, (num_subjects * num_timesteps, latent_dimensions))
        pca = PCA(n_components=2)
        x = pca.fit_transform(x_flat)
        x = np.reshape(x, (num_subjects, num_timesteps, 2))
        pca = True
    return x, pca

def visualize_trajectory(x: np.ndarray, p: Path, name: str, num_subjects=2):
    x, pca_done = prep_viz(x, num_subjects)    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(num_subjects):
        ax.scatter(x[i, :, 0], x[i, :, 1], alpha=0.9)
        ax.plot(x[i, :, 0], x[i, :, 1], alpha=0.7)
    ax.set_aspect(1)
    ax.set_xlabel(
        'PCA component 1' if pca_done else 'Latent dimension 1'
        )
    ax.set_ylabel(
        'PCA component 2' if pca_done else 'Latent dimension 2'
                )
    ax.set_title(f'{name} trajectory')
    plt.savefig(p, dpi=400)
    plt.clf()
    plt.close(fig)

def visualize_space(x: np.ndarray, p: Path, name: str, y=None, y_dict=None, num_subjects=-1):
    x, pca_done = prep_viz(x, num_subjects)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if y is None:
        for i in range(num_subjects):
            ax.scatter(x[i, :, 0], x[i, :, 1], alpha=0.5)
    else:
        unique_y_values = np.unique(y).astype(int)
        print(unique_y_values)
        y_names = []
        for unique_y_value in unique_y_values:
            if y_dict is not None:
                y_name = y_dict[unique_y_value]['name']
                y_color = y_dict[unique_y_value]['color']
                ax.scatter(
                    x[y == unique_y_value, :, 0].flatten(),
                    x[y == unique_y_value, :, 1].flatten(),
                    alpha=0.5, color=y_color)
            else:
                y_name = unique_y_value
                ax.scatter(
                    x[y == unique_y_value, :, 0].flatten(),
                    x[y == unique_y_value, :, 1].flatten(),
                    alpha=0.5)
            y_names.append(y_name)
        ax.legend(y_names)
    ax.set_aspect(1)
    ax.set_xlabel(
        'PCA component 1' if pca_done else 'Latent dimension 1'
        )
    ax.set_ylabel(
        'PCA component 2' if pca_done else 'Latent dimension 2'
                )
    ax.set_title(f'{name} space')
    plt.savefig(p, dpi=400)
    plt.clf()
    plt.close(fig)
