"""
From https://github.com/henrhoi/realnvp-pytorch/blob/0ecf65c9aa366b982932ed132c3d916a465826d8/utils.py
"""
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torchvision.utils import make_grid


def make_scatterplot(points, title=None):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    if title is not None:
        plt.title(title)


# 2D dimension
def load_smiley_face(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]


def load_half_moons(n):
    return make_moons(n_samples=n, noise=0.1)


def smiley_sample_data():
    train_data, train_labels = load_smiley_face(2000)
    test_data, test_labels = load_smiley_face(1000)
    return train_data, train_labels, test_data, test_labels


def half_moons_sample_data():
    train_data, train_labels = load_half_moons(2000)
    test_data, test_labels = load_half_moons(1000)
    return train_data, train_labels, test_data, test_labels


def show_2d_samples(samples, title='Samples'):
    plt.figure()
    plt.title(title)
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()


def show_2d_latents(latents, labels, title='Latent Space'):
    plt.figure()
    plt.title(title)
    plt.scatter(latents[:, 0], latents[:, 1], s=1, c=labels)
    plt.xlabel('z1')
    plt.ylabel('z2')

    plt.show()


def show_2d_densities(densities, dset_type, title='Densities'):
    plt.figure()
    plt.title(title)
    dx, dy = 0.025, 0.025
    if dset_type == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    elif dset_type == 2:  # moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    else:
        raise Exception('Invalid dset_type:', dset_type)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]

    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.xlabel('z1')
    plt.ylabel('z2')

    plt.show()


def show_results_2d(dset_type, fn):
    if dset_type == 1:
        train_data, train_labels, test_data, test_labels = smiley_sample_data()
    elif dset_type == 2:
        train_data, train_labels, test_data, test_labels = half_moons_sample_data()
    else:
        raise Exception('Invalid dset_type:', dset_type)

    train_losses, test_losses, densities, latents = fn(train_data, test_data, dset_type)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')

    show_training_plot(train_losses, test_losses, f'Dataset {dset_type} Train Plot')
    show_2d_densities(densities, dset_type)
    show_2d_latents(latents, train_labels)


# Multiple dimensions
def show_results_shapes(fn):
    data_dir = "data"
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))

    train_losses, test_losses, samples = fn(train_data, test_data)
    samples = np.clip(samples.astype('float') * 2.0, 0, 1.9999)
    floored_samples = np.floor(samples)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    show_training_plot(train_losses, test_losses, f'Dataset Train Plot')
    show_samples(samples * 255.0 / 2.0)
    show_samples(floored_samples * 255.0, title='Samples with Flooring')


def get_celeb_data():
    data_dir = "data"
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    return train_data, test_data


def show_results_celeb_a(fn):
    data_dir = "data"
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))

    train_losses, test_losses, samples, interpolations = fn(train_data, test_data)
    samples = samples.astype('float')
    interpolations = interpolations.astype('float')

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    show_training_plot(train_losses, test_losses, f'Dataset Train Plot')
    show_samples(samples * 255.0)
    show_samples(interpolations * 255.0, nrow=6, title='Interpolations')


# General utils

def show_training_plot(train_losses, test_losses, title):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.show()


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def show_samples(samples, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
