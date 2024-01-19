from pathlib import Path
from lib.utils import (
    get_default_config, load_hyperparameters,
    generate_version_name, init_model,
    init_data_module, embed_dataloader)
from lib.visualizations import visualize_trajectory

config = get_default_config([''])
config['model'] = 'CDSVAE'
config['local_size'] = 2
config['context_size'] = 2
config['seed'] = 42
version = generate_version_name(config)
result_p = Path(f'ray_results/{version}')
hyperparameters = load_hyperparameters(result_p / 'params.json')
model = init_model(config, hyperparameters, viz=False)
dm = init_data_module(config)
# The shape out of embed_ functions
# will be:
# (num_subjects, num_windows, ...)
# Where ... depends on the variable we are trying
# to obtain. 
embed_dict = embed_dataloader(config, model, dm.train_dataloader())
local_embeddings = embed_dict['local_mean']
visualize_trajectory(local_embeddings, Path(f'results/local_{config["model"]}_trajectory.png'),
                     f'Local {config["model"]}')
context_embeddings = embed_dict['context_mean']
visualize_trajectory(context_embeddings, Path(f'results/context_{config["model"]}_trajectory.png'),
                     f'Context {config["model"]}')
