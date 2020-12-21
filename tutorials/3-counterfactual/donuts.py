#!/usr/bin/env python
# coding: utf-8

# # Concentric Circles - "Donuts" Dataset

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from pyro import poutine
import torch
import pyro


# # Donuts Dataset Class

# #### Data Generator for the Donuts Tutorial

# From sklearn's [make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) function, we create a dataset with chosen noise and factor variables:
# * noise - Standard deviation of Gaussian noise added to the data as a double value
# * factor - Scale factor between inner and outer circle as a double value ranging from 0 to 1

# DonutsDataset's make_dataset returns a dataset containing:
# * x1 - array of the generated samples of x1 from the pairs of X
# * x2 - array of the generated samples of x2 from the pairs of X
# * label - array of the the integer labels (0 or 1) for class membership of each X sample
# * noise - the noise value defined in data creation
# * factor - the factor value defined in data creation

# This is the data generating process that is modeled in the counterfactual_donuts_tutorial and that counterfactual inference is performed for.

# In[25]:


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


# # Plot Function for Visualization of the Dataset

# In[26]:


def plot(x1, x2, y):
    X = StandardScaler().fit_transform(np.concatenate([x1, x2], axis=-1))

    plt.title(r'Samples from $p(x_1,x_2)$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.scatter(X[:,0], X[:,1], alpha=0.5, color=['red' if y_ == 1 else 'green' for y_ in y])
    ax = plt.gca()
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
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


# # Plot Function for Visualization of the Predicted (x1,x2) pairs and the Original Dataset

# In[27]:


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
    plt.scatter(X1, X2, label='data', alpha=0.5, color=['red' if y_ == 1 else 'red' for y_ in Y])
    ax = plt.gca()
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    plt.scatter(x1, x2, label='flow', alpha=0.5, color=['blue' if y_ == 1 else 'blue' for y_ in y])
    ax = plt.gca()
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    plt.legend()
    plt.show()

    plt.subplot(1, 2, 1)
    sns.distplot(X1, hist=False, kde=True, 
                 bins=None,color='red',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='data')
    sns.distplot(x1, hist=False, kde=True, 
                 bins=None, color='blue',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='flow')
    plt.title(r'$p(x_1)$')
    plt.subplot(1, 2, 2)
    sns.distplot(X2, hist=False, kde=True, 
                 bins=None,color='red',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='data')
    sns.distplot(x2, hist=False, kde=True, 
                 bins=None, color='blue',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 2},
                 label='flow')
    plt.title(r'$p(x_2)$')
    plt.show()


# # Flow-Based PyroModule Framework Helper Functions

# In[28]:


from functools import reduce
import sys
from functools import partial

def sumall(tens):
    reified = list(tens)  # NOTE: Need the lookahead for initializer, this is an iterator
    initializer = torch.zeros_like(reified[0])
    reducer = lambda l, memo: memo + l
    return reduce(reducer, reified, initializer)

def get_logprobs(tr, k):
    site = tr.nodes[k]
    assert site["type"] == "sample" and site["is_observed"], f'{k} is type {site["type"]} and is_observed: {site["is_observed"]}'
    return site["log_prob"]

def visualize(dataset, model):
    with pyro.plate('samples', 1000):
        outs = model.model()
        x1_flow = outs[0]
        x2_flow = outs[1]
        y_flow = torch.ones_like(x1_flow)
    compare((x1_flow.detach(), x2_flow.detach(), y_flow.detach()), dataset)
    
def smoke_test(model):
    print(model.model())
    with pyro.plate('test', 13):
        out = model.model()
        print(out[0].shape)
    sleep(0.5)


# ## Helper function for Conditional Affine Transforms made by [N. Pawlowski+, D. C. Castro+, B. Glocker.](https://github.com/biomedia-mira/deepscm)

# In[31]:


from pyro.distributions.conditional import ConditionalTransformModule, ConditionalTransformedDistribution

# from deepscm.distributions.transforms.affine.py
class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)

        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context):
        loc, log_scale = self.context_nn(context)
        scale = torch.exp(log_scale)

        ac = T.AffineTransform(loc, scale, event_dim=self.event_dim)
        return ac

