import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.data import ICAUKBiobank, ICAfBIRN

train_dataset = ICAfBIRN('train', 'individual', 42, None, None)
train_loader = DataLoader(train_dataset, num_workers=5, pin_memory=True,
                        batch_size=64, shuffle=True,
                        persistent_workers=True, prefetch_factor=5, drop_last=True)

data = []
for (ix, (x, _, _)) in enumerate(train_loader):
    break

x = x.cpu().numpy()
print(x.shape)
fig, axs = plt.subplots(7, 7, figsize=(10, 10))
for i in range(7):
    for j in range(7):
        for k in range(5):
            axs[i, j].plot(np.arange(0, 30), x[k, i * 4 + j], alpha=0.5)

plt.savefig('figures/fbirn_viz.png')
plt.clf()
plt.close(fig)
