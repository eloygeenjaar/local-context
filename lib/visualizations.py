import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union, List
from lapjv import lapjv
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
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

def visualize_space_inputs(original_input, x: np.ndarray, p: Path, name: str, y=None, y_dict=None, num_subjects=-1,
                           jonker_volgenant=True):
    x, pca_done = prep_viz(x, num_subjects)
    num_subjects, num_windows, window_size, _ = original_input.shape
    x = np.reshape(x, (num_subjects * num_windows, -1))
    original_input = np.reshape(original_input,
                                (num_subjects * num_windows, window_size, -1))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scale = 250
    ax.set_xlim(x[:, 0].min() * scale - 50, x[:, 0].max() * scale + 53 + 50)
    ax.set_ylim(x[:, 1].min() * scale - 50, x[:, 1].max() * scale + 53 + 50)
    for i in range(num_subjects):
        for j in range(num_windows):
            ix = i * num_windows + j
            coordinates = x[ix]
            og = original_input[ix]
            fnc = np.corrcoef(og.T)
            y_value = y[i]
            ax.imshow(
                fnc,
                extent=(coordinates[0] * scale, coordinates[0] * scale + 53,
                        coordinates[1] * scale, coordinates[1] * scale + 53),
                cmap='jet')
            square_linewidth = 2
            square = patches.Rectangle(
                (coordinates[0] * scale - square_linewidth // 2, coordinates[1] * scale - square_linewidth // 2),
                53 + np.ceil(square_linewidth / 2), 53 + np.ceil(square_linewidth / 2),
                # (72 / dpi) https://stackoverflow.com/questions/57657419/how-to-draw-a-figure-in-specific-pixel-size-with-matplotlib
                linewidth=square_linewidth * 72/400,
                edgecolor='r' if y_value else 'b',
                facecolor='none')
            ax.add_patch(square)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_xlabel(
        'PCA component 1' if pca_done else 'Latent dimension 1'
        )
    ax.set_ylabel(
        'PCA component 2' if pca_done else 'Latent dimension 2'
                )
    ax.set_title(f'{name} space')
    plt.savefig(p, dpi=500)
    plt.clf()
    plt.close(fig)
    
def jonker_volgenant(original_input, x: np.ndarray, p: Path, name: str, y=None, y_dict=None, num_subjects=-1):
    # Follows: https://github.com/kylemcdonald/CloudToGrid/blob/master/CloudToGrid.ipynb
    x, pca_done = prep_viz(x, num_subjects)
    num_subjects, num_windows, window_size, data_size = original_input.shape
    # We are creating a square grid
    side_subjects = int(np.sqrt(num_subjects))
    side_windows = int(np.sqrt(num_windows))
    side = side_subjects * side_windows
    num_subjects = int(side_subjects ** 2)
    num_windows = int(side_windows ** 2)
    x = x[:num_subjects, :num_windows]
    original_input = original_input[:num_subjects, :num_windows]
    x = np.reshape(x, (num_subjects * num_windows, -1))
    # Calculate all the FNCs
    original_input = np.reshape(original_input,
                                (num_subjects * num_windows, window_size, -1))
    fncs = np.empty((num_subjects, num_windows, data_size, data_size))
    subject_ix = np.arange(num_subjects)[:, np.newaxis]
    subject_ix = np.repeat(subject_ix, num_windows, axis=1)
    subject_ix = np.reshape(subject_ix, (num_subjects * num_windows, ))
    for i in range(num_subjects):
        for j in range(num_windows):
            ix = i * num_windows + j
            og = original_input[ix]
            fnc = np.corrcoef(og.T)
            fncs[i, j] = fnc
    fncs = np.reshape(fncs, (num_subjects * num_windows, data_size, data_size))
    # Create the grid
    _x = np.linspace(0, 1, side)
    _y = np.linspace(0, 1, side)
    xv, yv = np.meshgrid(
         _x,
         _y)
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    # Calculate cost based on embeddings
    cost = cdist(grid, x, "sqeuclidean")
    # See reference why this is necessary:
    # "I've noticed if you normalize to a maximum value that is too large, 
    # this can also cause the Hungarian implementation to crash."
    cost = cost * (10000000. / cost.max())
    row_ind, col_ind, _ = lapjv(cost)
    grid_jv = grid[col_ind]
    fig, axs = plt.subplots(side, side, figsize=(20, 20))
    vminmax = np.abs(fncs).max()
    for (subj_ix, fnc, ixs) in zip(subject_ix, fncs, grid_jv):
        ix = (np.abs(_x - ixs[0]).argmin(),
              np.abs(_y - ixs[1]).argmin())
        axs[ix[0], ix[1]].imshow(fnc, cmap='jet', vmin=-vminmax, vmax=vminmax)
        axs[ix[0], ix[1]].set_xticks([])
        axs[ix[0], ix[1]].set_yticks([])
        axs[ix[0], ix[1]].set_aspect('equal')
        axs[ix[0], ix[1]].set_xticklabels([])
        axs[ix[0], ix[1]].set_yticklabels([])
        for spine in axs[ix[0], ix[1]].spines.values():
            spine.set_edgecolor(y_dict[int(y[subj_ix])]['color'])
            spine.set_linewidth(1.75)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle(f'{name} space')
    #plt.tight_layout()
    plt.savefig(p, dpi=400, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
