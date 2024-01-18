from pathlib import Path
from lib.utils import (
    get_default_config, load_hyperparameters,
    generate_version_name, init_model,
    init_data_module, embed_dataloader)
from lib.visualizations import (
    visualize_trajectory,
    visualize_space)

visualizations_possible = [
    'context_space', 'trajectory'
]
visualizations = ['context_space']

config = get_default_config([''])
config['model'] = 'CDSVAE'
config['local_size'] = 2
config['context_size'] = 2
config['seed'] = 42
version = generate_version_name(config)
print(version)
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
    print(context_embeddings.size(), targets.size())
    visualize_space(context_embeddings, Path(f'results/context_{config["model"]}_space.png'),
                    f'Context {config["model"]}', y=targets)
