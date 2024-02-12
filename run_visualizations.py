from pathlib import Path
from lib.utils import (
    get_default_config, load_hyperparameters,
    generate_version_name, init_model,
    init_data_module, embed_dataloader)
from lib.visualizations import (
    visualize_trajectory,
    visualize_space,
    visualize_reconstruction,
    visualize_space_inputs,
    jonker_volgenant)
from sklearn.cluster import KMeans


visualizations_possible = [
    'context_space', 'trajectory', 'reconstruction',
    'reconstruction_fnc', 'space_inputs',
    'jonker_volgenant'
]
visualizations = ['context_space']

config = get_default_config([''])
config['model'] = 'CFDSVAE'
config['local_size'] = 2
config['context_size'] = 2
config['seed'] = 42
config['dataset'] = 'UpsampledICAfBIRN'
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

if 'trajectory' in visualizations:
    local_embeddings = embed_dict['local_mean']
    visualize_trajectory(local_embeddings, Path(f'results/local_{config["model"]}_trajectory.png'),
                        f'Local {config["model"]}')
    context_embeddings = embed_dict['context_mean']
    visualize_trajectory(context_embeddings, Path(f'results/context_{config["model"]}_trajectory.png'),
                        f'Context {config["model"]}')

if 'context_space' in visualizations:
    context_embeddings = embed_dict['context_mean']
    targets = embed_dict['target']
    y_dict = {
        0: {'name': 'C', 'color': 'b'},
        1: {'name': 'SZ', 'color': 'r'}
    }
    visualize_space(context_embeddings, Path(f'results/context_{config["model"]}_space.png'),
                    f'Context {config["model"]}', y=targets, y_dict=y_dict)
    #import numpy as np
    #km = KMeans(n_clusters=2, n_init='auto')
    #km.fit(np.reshape(context_embeddings, (context_embeddings.shape[0] * context_embeddings.shape[1], context_embeddings.shape[2])))
    #print(km.cluster_centers_)

if 'reconstruction' in visualizations:
    # Shape: (num_subjects, num_windows, window_size, data_size)
    original_inputs = embed_dict['input']
    reconstructions = embed_dict['reconstruction']
    visualize_reconstruction(original_inputs, reconstructions, Path(f'results/reconstruction_{config["model"]}.png'),
                             f'{config["model"]}')

if 'space_inputs' in visualizations:
    original_inputs = embed_dict['input']
    context_embeddings = embed_dict['context_mean']
    targets = embed_dict['target']
    y_dict = {
        0: {'name': 'C', 'color': 'b'},
        1: {'name': 'SZ', 'color': 'r'}
    }
    visualize_space_inputs(
        original_inputs,
        context_embeddings, Path(f'results/context_{config["model"]}_space_inputs.png'),
        f'Context {config["model"]}', y=targets, y_dict=y_dict)

if 'jonker_volgenant' in visualizations:
    original_inputs = embed_dict['input']
    context_embeddings = embed_dict['context_mean']
    targets = embed_dict['target']
    y_dict = {
        0: {'name': 'C', 'color': 'b'},
        1: {'name': 'SZ', 'color': 'r'}
    }
    jonker_volgenant(
        original_inputs,
        context_embeddings, Path(f'results/context_{config["model"]}_jonker_volgenant.png'),
        f'Context {config["model"]}', y=targets, y_dict=y_dict)
    original_inputs = embed_dict['input']
    context_embeddings = embed_dict['context_mean']
    targets = embed_dict['target']
    y_dict = {
        0: {'name': 'C', 'color': 'b'},
        1: {'name': 'SZ', 'color': 'r'}
    }
    jonker_volgenant(
        original_inputs,
        original_inputs, Path(f'results/input_jonker_volgenant.png'),
        f'Input', y=targets, y_dict=y_dict)
