import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)
def normal_sampling(current_ix: int, length: int, sd: float):
    sample = int(np.round(rng.normal(0, sd), 0))
    # We want to sample other timesteps not the same
    if sample == 0 and (current_ix != length - 1) and (current_ix != 0):
        sampled_ix = current_ix + rng.integers(low=0, high=2) * 2 - 1
    # If the current index is zero then we can only sample up
    elif current_ix == 0:
        sampled_ix = abs(sample)
        if sampled_ix == 0:
            sampled_ix = 1
    # If we have the highest ix then we can only sample down
    elif current_ix == length - 1:
        sampled_ix = length - abs(sample)
        # This means sample = 0
        if sampled_ix == length - 1:
            sampled_ix = length - 2
    else:
        sampled_ix = current_ix + sample
    
    if sampled_ix < 0:
        sampled_ix = 0
    elif sampled_ix > length - 1:
        sampled_ix = length -1
    return sampled_ix

sampled_ixs = []
for i in range(1000):
    sampled_ixs.append(normal_sampling(20, 150, sd=3))

print(sampled_ixs)
plt.hist(sampled_ixs)
plt.savefig('figures/distribution_samples.png')