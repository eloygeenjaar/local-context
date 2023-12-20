import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Load classification results for our method
# LVAE (baseline), and windowed-FNC (wFNC)
proposed_arr = np.load('results/proposed.npy')
lvae = np.load('results/lvae.npy')
wfnc = np.load('results/wfnc.npy')

print(proposed_arr[:, 1, 0].shape, lvae[:, 0, 0].shape)
results = np.stack((
    proposed_arr[:, 1, 0],
    lvae[:, 0, 0],
    np.ones((proposed_arr.shape[0], )) * wfnc[0]), axis=-1)
print(ttest_ind(proposed_arr[:, 1, 0], lvae[:, 0, 0]))
# Plot a barplot across seeds
fig, ax = plt.subplots(1, 1)
bplot = ax.boxplot(
    results, labels=['Local-global', 'Local only', 'dFNC'], patch_artist=True)
ax.set_ylabel('Window classification accuracy')

# Set the colors for LVAE and wFNC
colors = ['#93278F', '#0071BC', '#03A678']
for (patch, color) in zip(bplot['boxes'], colors):
    patch.set(facecolor=color)
plt.tight_layout()
plt.savefig('figures/boxplot.png', dpi=300)
