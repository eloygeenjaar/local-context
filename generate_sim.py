import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
matplotlib.rcParams.update({'font.size': 20})

seed = 42
np.random.seed(seed)
rng = np.random.default_rng(seed)

num_subjects = 1000
c_means = np.array(
    [[-6, -0.5],
     [-2, 0.5],
     [2, -0.5],
     [6, 0.5]]
)
sz_means = c_means + np.array([0, 2.0])
data_ls, states_ls, latent_ls = [], [], []
W = rng.normal(size=(3, 53))
covars = np.ones((4, 2)) * 1.0
covars[:, 1] = 0.5
sz_arr = rng.integers(low=0, high=2, size=(num_subjects, ))

transition_matrix = np.array([
    [0.85, 0.05, 0.05, 0.05],
    [0.05, 0.85, 0.05, 0.05],
    [0.05, 0.05, 0.85, 0.05],
    [0.05, 0.05, 0.05, 0.85]
])

for subj in range(num_subjects):
    sz = sz_arr[subj]
    means = sz_means if sz else c_means
    #model = hmm.GaussianHMM(n_components=4, covariance_type="diag")
    #model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])
    #model.means_ = sz_means if sz else c_means
    #model.covars_ = covars
    # Staying in the same state is more likely for simulated schizophrenia-diagnosed 
    # subjects
    #min_rand = 1.0 if sz else 0.1
    #max_rand = 3.0 if sz else 0.3
    #prob_retain_state = rng.random(size=(4, ))
    #prob_retain_state = prob_retain_state * (max_rand - min_rand) + min_rand
    #transition_matrix = rng.random(size=(4, 4)) * (1 - 0.1) + 0.1
    #transition_matrix = rng.random(size=(4, 4))
    # Normalize rows
    #transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:, None]
    #np.fill_diagonal(transition_matrix, prob_retain_state)
    # Re-normalize rows
    #transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:, None]
    #model.transmat_ = transition_matrix
    #latents, states = model.sample(100)
    states = rng.integers(low=0, high=4, size=(5, ))
    print(states)
    latents_ls = []
    for state in states:
        latents_ls.append(rng.normal(means[state], covars[state], size=(20, 2)))
    latents = np.concatenate(latents_ls, axis=0)
    print(latents.shape)
    latents = np.concatenate((latents, rng.normal(0, 1, size=(100, 1))), axis=-1)
    print(latents.shape)
    inputs = latents @ W
    #inputs -= inputs.mean(0)
    #inputs /= inputs.std(0)
    data_ls.append(inputs)
    states_ls.append(states)
    latent_ls.append(latents)

data = np.stack(data_ls, axis=0)
states = np.stack(states_ls, axis=0)
latents = np.stack(latent_ls, axis=0)

print((np.diff(states[sz_arr == 0], axis=1) != 0).sum(1).mean(0),
      (np.diff(states[sz_arr == 1], axis=1) != 0).sum(1).mean(0))

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for k in range(10):
    axs.scatter(latents[k, :, 0], latents[k, :, 1], alpha=0.75)
    axs.scatter(c_means[:, 0], c_means[:, 1], color='b', s=100, marker='s')
    axs.scatter(sz_means[:, 0], sz_means[:, 1], color='r', s=100, marker='s')
axs.set_xlabel('Latent 1')
axs.set_ylabel('Latent 2')
axs.set_title('Simulated latent states')
plt.savefig('figures/simulation.png', dpi=300)
plt.clf()
plt.close(fig)

ix = np.arange(data.shape[0])
ix_trainval, ix_test, y_trainval, y_test = train_test_split(
    ix, sz_arr, test_size=0.2, shuffle=True, stratify=sz_arr, random_state=seed)
ix_train, ix_val, y_train, y_val = train_test_split(
    ix_trainval, y_trainval, test_size=0.1, shuffle=True, stratify=y_trainval, random_state=seed)

np.save('data/simulated/x_train.npy', data[ix_train])
np.save('data/simulated/s_train.npy', states[ix_train])
np.save('data/simulated/l_train.npy', latents[ix_train])
np.save('data/simulated/y_train.npy', y_train)
np.save('data/simulated/x_valid.npy', data[ix_val])
np.save('data/simulated/s_valid.npy', states[ix_val])
np.save('data/simulated/l_valid.npy', latents[ix_val])
np.save('data/simulated/y_valid.npy', y_val)
np.save('data/simulated/x_test.npy', data[ix_test])
np.save('data/simulated/s_test.npy', states[ix_test])
np.save('data/simulated/l_test.npy', latents[ix_test])
np.save('data/simulated/y_test.npy', y_test)