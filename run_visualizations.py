from pathlib import Path
from lib.utils import (
    get_default_config, load_hyperparameters,
    generate_version_name, init_model,
    init_data_module, embed_dataloader)
from lib.visualizations import visualize_trajectory

config = get_default_config([''])
config['model'] = 'DSVAE'
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
visualize_trajectory(local_embeddings, Path('results/local_DSVAE_trajectory.png'),
                     'Local DSVAE')
context_embeddings = embed_dict['context_mean']
visualize_trajectory(context_embeddings, Path('results/context_DSVAE_trajectory.png'),
                     'Local DSVAE')
