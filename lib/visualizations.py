import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


def visualize_trajectory(x: np.ndarray, p: Path, name: str, num_subjects=2):
    # We assume that x comes in the following shape:
    # (num_subjects, num_timesteps, latent_dimensions)
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
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(num_subjects):
        ax.scatter(x[i, :, 0], x[i, :, 1], alpha=0.9)
        ax.plot(x[i, :, 0], x[i, :, 1], alpha=0.7)
    ax.set_aspect(1)
    ax.set_xlabel(
        'Latent dimension 1' if latent_dimensions == 2 else 'PCA component 1'
                )
    ax.set_ylabel(
        'Latent dimension 2' if latent_dimensions == 2 else 'PCA component 2'
                )
    ax.set_title(f'{name.capitalize()} trajectory')
    plt.savefig(p, dpi=400)
    plt.clf()
    plt.close(fig)
