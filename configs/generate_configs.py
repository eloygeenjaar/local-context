import yaml
from pathlib import Path

with Path('./default.yaml').open('r') as f:
    default_conf = dict(yaml.safe_load(f))

i = 0
seeds = [42, 1337, 9999, 1212]
models = ['CDSVAE', 'DSVAE', 'SCDSVAE']
local_sizes = [2]
global_sizes = [2]
lrs = [0.001]
model_dict = default_conf.copy()
for seed in seeds:
    for model in models:
        for local_size in local_sizes:
            for global_size in global_sizes:
                for lr in lrs:
                    model_dict['seed'] = seed
                    model_dict['local_size'] = local_size
                    model_dict['global_size'] = global_size
                    model_dict['model'] = model
                    model_dict['lr'] = lr
                    with open(f'config_{i}.yaml', 'w') as file:
                        documents = yaml.dump(model_dict, file)
                    print(i,
                            model_dict['global_size'])
                    i += 1
seeds = [42, 1337, 9999, 1212]
models = ['LVAE']
local_sizes = [2, 4]
model_dict = default_conf.copy()
for seed in seeds:
    for model in models:
        for local_size in local_sizes:
            model_dict['seed'] = seed
            model_dict['local_size'] = local_size
            model_dict['global_size'] = 0
            model_dict['model'] = model
            with open(f'config_{i}.yaml', 'w') as file:
                documents = yaml.dump(model_dict, file)
            print(i,
                    model_dict['global_size'])
            i += 1
