import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lib.utils import (
    get_default_config, load_hyperparameters,
    generate_version_name, init_model,
    init_data_module, embed_dataloader)
from sklearn.linear_model import LinearRegression

local_sizes = [2, 4, 8]
context_sizes = [2, 4, 8]
seeds = [42, 1212, 1337, 9999]
models = ['DSVAE', 'IDSVAE']
results = np.zeros((
    len(local_sizes), len(context_sizes), len(models), (len(seeds) * len(seeds) - len(seeds))
))
for (l_ix, local_size) in enumerate(local_sizes):
    for (c_ix, context_size) in enumerate(context_sizes):
        for (m_ix, model_name) in enumerate(models):
            tr_seeds, te_seeds = [], []
            for (s_ix, seed) in enumerate(seeds):
                config = get_default_config([''])
                config['model'] = model_name
                config['local_size'] = local_size
                config['context_size'] = context_size
                config['seed'] = seed
                config['dataset'] = 'ICAfBIRN'
                version = generate_version_name(config)
                result_p = Path(f'ray_results/{version}')
                ckpt_p = result_p / 'final.ckpt'
                if not ckpt_p.is_file():
                    continue
                hyperparameters = load_hyperparameters(result_p / 'params.json')
                print(hyperparameters)
                model = init_model(config, hyperparameters, viz=False, ckpt_path=ckpt_p)
                dm = init_data_module(config)
                # The shape out of embed_ functions
                # will be:
                # (num_subjects, num_windows, ...)
                # Where ... depends on the variable we are trying
                # to obtain.
                tr_embed_dict = embed_dataloader(config, model, dm.train_dataloader())
                va_embed_dict = embed_dataloader(config, model, dm.val_dataloader())
                te_embed_dict = embed_dataloader(config, model, dm.test_dataloader())

                tr_context = np.concatenate((
                    tr_embed_dict['context_mean'].cpu().numpy(), va_embed_dict['context_mean'].cpu().numpy()
                ), axis=0)
                
                tr_seeds.append(np.reshape(
                    tr_context, (-1, model.context_size))
                )
                te_seeds.append(np.reshape(
                    te_embed_dict['context_mean'].cpu().numpy(), (-1, model.context_size))
                )

            print(local_size, context_size)
            if len(seeds) == len(te_seeds):
                seed_results = []
                for i in range(len(seeds)):
                    for j in range(len(seeds)):
                        if i != j:
                            lr = LinearRegression()
                            lr.fit(tr_seeds[i], tr_seeds[j])
                            score = lr.score(te_seeds[i], te_seeds[j])
                            seed_results.append(score)
                            print(i, j, score)
                results[l_ix, c_ix, m_ix, :] = np.asarray(seed_results)
                        



fig, axs = plt.subplots(len(local_sizes), len(context_sizes), figsize=(10, 10))
for (l_ix, local_size) in enumerate(local_sizes):
    for (c_ix, context_size) in enumerate(context_sizes):
        print(results[l_ix, c_ix].shape)
        bplot = axs[l_ix, c_ix].boxplot(results[l_ix, c_ix].T, vert=True, patch_artist=True, labels=models)
        colors = ["#93278F", "#03A678"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        axs[l_ix, c_ix].set_title(f'L{local_size}, C{context_size}')
        axs[-1, c_ix].set_xlabel('Models')
    axs[l_ix, 0].set_ylabel('Accuracy')

fig.savefig('results/reliability_analysis.png', bbox_inches=0, transparent=False, dpi=400)
plt.clf()
plt.close(fig)
