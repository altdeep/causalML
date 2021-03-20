# dSprites - Disentanglement testing Sprites dataset

This repository contains the dSprites dataset, used to assess the
disentanglement properties of unsupervised learning methods.

If you use this dataset in your work, please cite it as follows:

## Bibtex

```
@misc{dsprites17,
author = {Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner},
title = {dSprites: Disentanglement testing Sprites dataset},
howpublished= {https://github.com/deepmind/dsprites-dataset/},
year = "2017",
}
```

## Description

![dsprite_gif](dsprites.gif)

dSprites is a dataset of 2D shapes procedurally generated from 6 ground truth
independent latent factors. These factors are *color*, *shape*, *scale*,
*rotation*, *x* and *y* positions of a sprite.

All possible combinations of these latents are present exactly once,
generating N = 737280 total images.

### Latent factor values

*   Color: white
*   Shape: square, ellipse, heart
*   Scale: 6 values linearly spaced in [0.5, 1]
*   Orientation: 40 values in [0, 2 pi]
*   Position X: 32 values in [0, 1]
*   Position Y: 32 values in [0, 1]

We varied one latent at a time (starting from Position Y, then Position X, etc),
and sequentially stored the images in fixed order.
Hence the order along the first dimension is fixed and allows you to map back to
the value of the latents corresponding to that image.

We chose the latents values deliberately to have the smallest step changes
while ensuring that all pixel outputs were different. No noise was added.

The data is a NPZ NumPy archive with the following fields:

*   `imgs`: (737280 x 64 x 64, uint8) Images in black and white.
*   `latents_values`: (737280 x 6, float64) Values of the latent factors.
*   `latents_classes`: (737280 x 6, int64) Integer index of the latent factor
    values. Useful as classification targets.
*   `metadata`: some additional information, including the possible latent
    values.

Alternatively, a HDF5 version is also available, containing the same data,
packed as Groups and Datasets.

## Disentanglement metric

This dataset was created as a unit test of disentanglement properties of
unsupervised models. It can be used to determine how well models recover the
ground truth latents presented above.

You find our proposed disentanglement metric assessing the disentanglement
quality of a model (along with an example usage of this dataset) in:

[Higgins, Irina, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot,
Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. "beta-VAE: Learning
basic visual concepts with a constrained variational framework." In *Proceedings
of the International Conference on Learning Representations (ICLR).
2017.*](https://openreview.net/forum?id=Sy2fzU9gl)

## Disclaimers

This is not an official Google product.

The images were generated using the LOVE framework, which is licenced under
zlib/libpng licence:

```
LOVE is Copyright (c) 2006-2016 LOVE Development Team

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.

3. This notice may not be removed or altered from any source
distribution.
```
