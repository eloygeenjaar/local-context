import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Union, List
from pathlib import Path
from sklearn.decomposition import PCA
plt.rcParams.update({'font.size': 22})


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

def visualize_trajectory(x: np.ndarray, p: Path, name: str, num_subjects=5):
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
    kwargs_dict = {
        'alpha': 0.5,
        's': 50
    }
    if y is None:
        for i in range(num_subjects):
            ax.scatter(
                x[i, :, 0].flatten(),
                x[i, :, 1].flatten(),
                **kwargs_dict)
    else:
        unique_y_values = np.unique(y).astype(int)
        y_names = []
        for unique_y_value in unique_y_values:
            if y_dict is not None:
                y_name = y_dict[unique_y_value]['name']
                y_color = y_dict[unique_y_value]['color']
                kwargs_dict['color'] = y_color
            else:
                y_name = unique_y_value
                
            y_names.append(y_name)
            ax.scatter(x[y == unique_y_value, :, 0].flatten(),
                       x[y == unique_y_value, :, 1].flatten(),
                       **kwargs_dict)
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

def visualize_reconstruction(original_inputs, reconstructions, p: Path, name: str,
                             rows=2, cols=2, window_ixs: Union[List, int] = None,
                             seed=42):
    # We assume that x comes in the following shape:
    # (num_subjects, num_windows, window_size, data_size)
    num_subjects, num_windows, window_size, data_size = original_inputs.shape
    fig, axs = plt.subplots(2 * rows, cols * 2, figsize=(20, 20))
    
    if window_ixs is None:
        rng = np.random.default_rng(seed)
        window_ixs = rng.integers(low=0, high=num_windows, size=(rows * cols))
    elif type(window_ixs) == int:
        window_ixs = [window_ixs] * (rows * cols)
    elif len(window_ixs) < (rows * cols):
        raise NotImplementedError

    for i in range(rows):
        for j in range(cols):
            ix = i * cols + j
            og = original_inputs[ix, window_ixs[ix]]
            og_fnc = np.corrcoef(og.T)
            rec = reconstructions[ix, window_ixs[ix]]
            rec_fnc = np.corrcoef(rec.T)
            axs[2 * i, j * 2].imshow(og, cmap='jet')
            axs[2 * i, j * 2].set_title('OG')
            axs[2 * i + 1, j * 2].imshow(og_fnc, cmap='jet')
            axs[2 * i + 1, j * 2].set_title('OG FNC')
            axs[2 * i, (j * 2) + 1].imshow(rec, cmap='jet')
            axs[2 * i, (j * 2) + 1].set_title('Rec')
            axs[2 * i + 1, (j * 2) + 1].imshow(rec_fnc, cmap='jet')
            axs[2 * i + 1, (j * 2) + 1].set_title('Rec FNC')
    fig.suptitle(f'{name} Input - Reconstructions')
    plt.savefig(p, dpi=400)
    plt.clf()
    plt.close(fig)
