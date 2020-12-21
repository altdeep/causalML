import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader

from pyro import poutine

import torch


class DonutsDataset(Dataset):
    def __init__(self, n_samples, hole, noise=None, factor=None):
        super().__init__()
        self.data = self.make_dataset(n_samples, hole, noise, factor)

    def make_dataset(self, n_samples, hole, noise=None, factor=None):
        output = None
        if not noise:
            output = hole.model()
            noise = output[0][1].item()
        if not factor:
            output = hole.model() if output is None else output
            factor = output[1][1].item()
        self.noise_scalar = noise
        self.factor_scalar = factor
        X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=factor)
        dataset = {'noise': (torch.ones([n_samples]).view(-1, 1)*noise).float(), 
                   'factor': (torch.ones([n_samples]).view(-1, 1)*factor).float(),
                   'label': torch.tensor(y).view(-1, 1).float(),
                   'x1': torch.tensor(X[:, 0]).view(-1, 1).float(),
                   'x2': torch.tensor(X[:, 1]).view(-1, 1).float()}
        return dataset

    def __getitem__(self, idx):
        return {name: val[idx] for name, val in self.data.items()}

    def __len__(self):
        return len(self.data.values())


def plot(x1, x2, y):
    X = StandardScaler().fit_transform(np.concatenate([x1, x2], axis=-1))

    plt.title(r'Samples from $p(x_1,x_2)$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.scatter(X[:,0], X[:,1], alpha=0.5, color=['red' if y_ == 1 else 'green' for y_ in y])
    plt.show()

    plt.subplot(1, 2, 1)
    sns.distplot(X[:,0], hist=False, kde=True,
                 bins=None,
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2})
    plt.title(r'$p(x_1)$')
    plt.subplot(1, 2, 2)
    sns.distplot(X[:,1], hist=False, kde=True,
                 bins=None,
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2})
    plt.title(r'$p(x_2)$')
    plt.show()

def compare(model, dataset, n=1000):
    if isinstance(model, tuple):
        x1, x2, y = model
    else:
        x1, x2, y, _, _ = sample_trained(model.plate_model, n)
    X1, X2, Y, = dataset.data['x1'], dataset.data['x2'], dataset.data['label']

    #plot(x1, x2, y)
    plt.title(r'Joint Distribution')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.scatter(X1, X2, label='data', alpha=0.5, color=['red' if y_ == 1 else 'green' for y_ in Y])
    plt.scatter(x1, x2, label='flow', alpha=0.5, color=['firebrick' if y_ == 1 else 'blue' for y_ in y])
    plt.legend()
    plt.show()

    plt.subplot(1, 2, 1)
    sns.distplot(X1, hist=False, kde=True, 
                 bins=None,
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='data')
    sns.distplot(x1, hist=False, kde=True, 
                 bins=None, color='firebrick',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='flow')
    plt.title(r'$p(x_1)$')
    plt.subplot(1, 2, 2)
    sns.distplot(X1, hist=False, kde=True, 
                 bins=None,
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='data')
    sns.distplot(x1, hist=False, kde=True, 
                 bins=None, color='firebrick',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='flow')
    plt.title(r'$p(x_2)$')
    plt.show()

def factor_mechanism(noise, N_factor):
    return sigmoid(N_factor + noise)

def sample_trained(plate_model, n=1000):
    handler = poutine.trace(plate_model)
    trace = handler.get_trace(n)
    nodes = ['x1', 'x2', 'label', 'factor', 'noise']
    x1, x2, y, factor, noise = [trace.nodes[node]['value'].detach().numpy() for node in nodes]
    return x1, x2, y, factor, noise

