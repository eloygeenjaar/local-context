import torch
import numpy as np
import matplotlib.pyplot as plt


class_dict = {
    1: (0.5, 0.05, 1.8, 40, -1.5),
    2: (0.5, -0.05, 1.8, 40, 1.5),
    3: (0.8, 0.05, 0.8, 20, -1.5),
    4: (0.8, -0.05, 0.8, 20, 1.5)
}

def generate_data(alpha, gamma, a, b, c):
    t = np.linspace(0, 2*np.pi, 100)
    classes = np.random.randint(1, 5, size=(500, ))
    X_ls = []
    for i in range(500):
        alpha, gamma, a, b, c = class_dict[classes[i]]
        x = alpha * (gamma * t + a * np.sin((b * t)/(2*np.pi)) + c)
        X_ls.append(x)
    X = np.stack(X_ls, axis=0)
    X = X + np.random.normal(0, 0.1, size=(500, 100))
    return X, classes

X, y = generate_data(0.5, 0.05, 1.8, 40, -1.5)
fig, axs = plt.subplots(2, 2, figsize=(10, 2))
for i in range(2):
    for j in range(2):
        ix = i * 2 + j + 1
        axs[i, j].plot(X[(y==ix)][0])
        axs[i, j].set_ylim([-2, 2])
plt.show()