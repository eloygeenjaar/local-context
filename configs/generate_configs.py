import yaml
from pathlib import Path

with Path('./default.yaml').open('r') as f:
    default_conf = dict(yaml.safe_load(f))

i = 0
seeds = [42, 1337, 9999, 1212]
models = ['CFDSVAE']
local_sizes = [2, 4]
context_sizes = [2]
datasets = ['ICAfBIRN', 'UpsampledICAfBIRN']
model_dict = default_conf.copy()
for seed in seeds:
    for model in models:
        for local_size in local_sizes:
            for context_size in context_sizes:
                for dataset in datasets:
                    model_dict['dataset'] = dataset
                    model_dict['seed'] = seed
                    model_dict['local_size'] = local_size
                    model_dict['context_size'] = context_size
                    model_dict['model'] = model
                    with open(f'config_{i}.yaml', 'w') as file:
                        documents = yaml.dump(model_dict, file)
                    print(i,
                            model_dict['model'],
                            model_dict['seed'],
                            model_dict['local_size'],
                            model_dict['context_size'],
                            model_dict['dataset'])
                    i += 1

quit()
seeds = [42, 1337, 9999, 1212]
models = ['DSVAE', 'CDSVAE', 'CIDSVAE', 'IDSVAE']
local_sizes = [2, 4, 8]
context_sizes = [2, 4, 8, 16]
model_dict = default_conf.copy()
for seed in seeds:
    for model in models:
        for local_size in local_sizes:
            for context_size in context_sizes:
                model_dict['seed'] = seed
                model_dict['local_size'] = local_size
                model_dict['context_size'] = context_size
                model_dict['model'] = model
                with open(f'config_{i}.yaml', 'w') as file:
                    documents = yaml.dump(model_dict, file)
                print(i,
                        model_dict['model'],
                        model_dict['seed'],
                        model_dict['local_size'],
                        model_dict['context_size'])
                i += 1

seeds = [42, 1337, 9999, 1212]
models = ['LVAE']
local_sizes = [2, 4, 8]
model_dict = default_conf.copy()
for seed in seeds:
    for model in models:
        for local_size in local_sizes:
            model_dict['seed'] = seed
            model_dict['local_size'] = local_size
            model_dict['context_size'] = 0
            model_dict['model'] = model
            with open(f'config_{i}.yaml', 'w') as file:
                documents = yaml.dump(model_dict, file)
            print(i,
                  model_dict['model'],
                  model_dict['seed'],
                  model_dict['local_size'],
                  model_dict['context_size'])
            i += 1

seeds = [42, 1337, 9999, 1212, 8585, 6767]
models = ['CO']
context_sizes = [2, 4, 8, 16]
model_dict = default_conf.copy()
for seed in seeds:
    for model in models:
        for context_size in context_sizes:
            model_dict['seed'] = seed
            model_dict['local_size'] = 0
            model_dict['context_size'] = model_dict['window_size'] * context_size
            model_dict['model'] = model
            with open(f'config_{i}.yaml', 'w') as file:
                documents = yaml.dump(model_dict, file)
            print(i,
                  model_dict['model'],
                  model_dict['seed'],
                  model_dict['local_size'],
                  model_dict['context_size'])
            i += 1
