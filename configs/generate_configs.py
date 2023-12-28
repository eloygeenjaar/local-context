import yaml
from pathlib import Path

with Path('./default.yaml').open('r') as f:
    default_conf = dict(yaml.safe_load(f))

i = 0
seeds = [42, 1337, 9999, 1111, 1234]
models = ['GLR', 'TransGLR', 'mGLR']
datasets = ['ICAfBIRN']
gammas = [0.0]
normalizations = ['individual']
sizes = [8, 16, 32]
model_dict = default_conf.copy()
for normalization in normalizations:
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                for gamma in gammas:
                    for size in sizes:
                        model_dict['local_size'] = 32
                        model_dict['global_size'] = size
                        model_dict['batch_size'] = 2
                        model_dict['normalization'] = normalization
                        model_dict['gamma'] = gamma
                        model_dict['model'] = model
                        model_dict['seed'] = seed
                        model_dict['dataset'] = dataset
                        with open(f'config_{i}.yaml', 'w') as file:
                            documents = yaml.dump(model_dict, file)
                        print(i,
                                model_dict['normalization'],
                                model_dict['gamma'],
                                model_dict['model'],
                                model_dict['seed'],
                                model_dict['dataset'])
                        i += 1

quit()                        
seeds = [42, 1337, 9999, 1111, 1234]
models = ['ContGLR']
datasets = ['ICAfBIRN']
gammas = [0.01]
normalizations = ['individual']
sizes = [4, 8, 16, 32]
model_dict = default_conf.copy()
for normalization in normalizations:
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                for gamma in gammas:
                    for size in sizes:
                        model_dict['local_size'] = size
                        model_dict['global_size'] = size
                        model_dict['batch_size'] = 8
                        model_dict['normalization'] = normalization
                        model_dict['gamma'] = gamma
                        model_dict['model'] = model
                        model_dict['seed'] = seed
                        model_dict['dataset'] = dataset
                        with open(f'config_{i}.yaml', 'w') as file:
                            documents = yaml.dump(model_dict, file)
                        print(i,
                                model_dict['normalization'],
                                model_dict['gamma'],
                                model_dict['model'],
                                model_dict['seed'],
                                model_dict['dataset'])
                        i += 1

seeds = [42, 1337, 9999, 1111, 1234]
models = ['GLR', 'TransGLR', 'mGLR']
datasets = ['ICAfBIRN']
gammas = [0.0]
normalizations = ['individual']
sizes = [4, 8, 16, 32]
model_dict = default_conf.copy()
for normalization in normalizations:
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                for gamma in gammas:
                    for size in sizes:
                        model_dict['local_size'] = size
                        model_dict['global_size'] = 2
                        model_dict['batch_size'] = 8
                        model_dict['normalization'] = normalization
                        model_dict['gamma'] = gamma
                        model_dict['model'] = model
                        model_dict['seed'] = seed
                        model_dict['dataset'] = dataset
                        with open(f'config_{i}.yaml', 'w') as file:
                            documents = yaml.dump(model_dict, file)
                        print(i,
                                model_dict['normalization'],
                                model_dict['gamma'],
                                model_dict['model'],
                                model_dict['seed'],
                                model_dict['dataset'])
                        i += 1

quit()
seeds = [42, 1337, 9999, 1111, 1234]
models = ['VAE']
datasets = ['ICAUKBiobank']
normalizations = ['individual']
model_dict = default_conf.copy()
for normalization in normalizations:
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                model_dict['batch_size'] = 8
                model_dict['normalization'] = normalization
                model_dict['gamma'] = 0.
                model_dict['model'] = model
                model_dict['seed'] = seed
                model_dict['dataset'] = dataset
                with open(f'config_{i}.yaml', 'w') as file:
                    documents = yaml.dump(model_dict, file)
                print(i,
                        model_dict['normalization'],
                        model_dict['gamma'],
                        model_dict['model'],
                        model_dict['seed'],
                        model_dict['dataset'])
                i += 1

quit()
models = ['LinmGLR', 'LinGLR']
datasets = ['fBIRN']
gammas = [0.1, 0.01, 0.0]
normalizations = ['individual']
sizes = [8, 16, 32]
fold_ixs = list(range(10))
model_dict = default_conf.copy()
for normalization in normalizations:
    for fold_ix in fold_ixs:
        for model in models:
            for dataset in datasets:
                for gamma in gammas:
                    for size in sizes:
                        model_dict['local_size'] = size
                        model_dict['global_size'] = size
                        model_dict['batch_size'] = 8
                        model_dict['normalization'] = normalization
                        model_dict['gamma'] = gamma
                        model_dict['model'] = model
                        model_dict['seed'] = 42
                        model_dict['dataset'] = dataset
                        model_dict['input_size'] = 128
                        model_dict['fold_ix'] = fold_ix
                        with open(f'config_{i}.yaml', 'w') as file:
                            documents = yaml.dump(model_dict, file)
                        print(i,
                                model_dict['normalization'],
                                model_dict['gamma'],
                                model_dict['model'],
                                model_dict['seed'],
                                model_dict['dataset'],
                                model_dict['input_size'])
                        i += 1

datasets = ['fBIRN']
models = ['LinVAE']
normalizations = ['individual']
sizes = [8, 16, 32]
fold_ixs = list(range(10))
model_dict = default_conf.copy()
for normalization in normalizations:
    for fold_ix in fold_ixs:
        for model in models:
            for dataset in datasets:
                for size in sizes:
                    model_dict['local_size'] = size
                    model_dict['global_size'] = size
                    model_dict['batch_size'] = 8
                    model_dict['normalization'] = normalization
                    model_dict['gamma'] = 0.
                    model_dict['model'] = model
                    model_dict['seed'] = 42
                    model_dict['dataset'] = dataset
                    model_dict['input_size'] = 128
                    model_dict['fold_ix'] = fold_ix
                    with open(f'config_{i}.yaml', 'w') as file:
                        documents = yaml.dump(model_dict, file)
                    print(i,
                            model_dict['normalization'],
                            model_dict['gamma'],
                            model_dict['model'],
                            model_dict['seed'],
                            model_dict['dataset'],
                            model_dict['input_size'])
                    i += 1