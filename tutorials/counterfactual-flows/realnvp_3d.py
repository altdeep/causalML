"""
From https://github.com/henrhoi/realnvp-pytorch/blob/0ecf65c9aa366b982932ed132c3d916a465826d8/realnvp_3d.py
"""

import torch
import torch.nn as nn
import random
import time
from utils import *


class ActNorm(nn.Module):
    """
    ActNorm-class for activation normalization as described in Section 1.3 in Glow by Kingma & Dhariwal
    Found inspiration for class implementation <a href="https://github.com/axium/Glow-Pytorch/blob/master/actnorm.py">here</a>.
    """

    def __init__(self, no_channels):
        super(ActNorm, self).__init__()
        shape = (1, no_channels, 1, 1)
        self.initialized = False
        self.log_std = torch.nn.Parameter(torch.zeros(shape).float())
        self.mean = torch.nn.Parameter(torch.zeros(shape).float())

    def initialize(self, x):
        with torch.no_grad():
            x_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            variance = ((x - x_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            log_std = torch.log(torch.sqrt(variance))

            self.log_std.data.copy_(-log_std.data)
            self.mean.data.copy_(-x_mean.data)
            self.initialized = True

    def apply_bias(self, x, backward):
        """
        Subtracting bias if forward, addition if backward
        """

        direction = -1 if backward else 1
        return x + direction * self.mean

    def apply_scale(self, x, backward):
        """
        Applying scale
        """

        direction = -1 if backward else 1
        return x * torch.exp(direction * self.log_std)

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)

        x = self.apply_bias(x, False)
        x = self.apply_scale(x, False)
        return x, self.log_std

    def backward(self, z):
        x = self.apply_scale(z, True)
        x = self.apply_bias(x, True)
        return x


class WeightNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x):
        return self.conv(x)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input x is (N, C, H, W)
        # Want output to be (N, 4*C, H/2, W/2)
        # with the squeezing operation described in realNVP paper
        N, C, H, W = x.shape
        x = x.view(N, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(N, 4 * C, H // 2, W // 2)

    def backward(self, x):
        return UnSqueeze()(x)


class UnSqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape  # C will be 4*C of the original, and H and W are H//2 and W//2
        x = x.view(N, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(N, C // 4, 2 * H, 2 * W)

    def backward(self, x):
        return Squeeze()(x)


class ResidualConv2d(nn.Module):
    """
    Residual Links between MaskedConv2d-layers
    As described in Figure 5 in "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_dim):
        super(ResidualConv2d, self).__init__()
        self.net = nn.Sequential(
            WeightNormConv2d(in_dim, in_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            WeightNormConv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            WeightNormConv2d(in_dim, in_dim, kernel_size=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        return self.net(x) + x


class ResidualCNN(nn.Module):
    """
     Residual CNN-class using residual blocks from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_channels, out_channels, conv_filters=128, residual_blocks=8):
        super().__init__()
        modules = [WeightNormConv2d(in_channels, conv_filters, kernel_size=3, padding=1), nn.ReLU()]
        modules += [ResidualConv2d(conv_filters) for _ in range(residual_blocks)]
        modules += [WeightNormConv2d(conv_filters, out_channels, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class CheckboardAffineCouplingLayer(nn.Module):
    """
    Coupling layer for RealNVP-class with checkboard mask
    """

    def __init__(self, in_channels, conv_filters=128, residual_blocks=8, top_condition=True, input_shape=(32, 32)):
        super(CheckboardAffineCouplingLayer, self).__init__()
        self.register_buffer('mask', self.get_mask(input_shape, top_condition))
        self.cnn = ResidualCNN(in_channels, 2 * in_channels, conv_filters, residual_blocks)
        self.scale = nn.Parameter(torch.tensor([1.]),
                                  requires_grad=True)  # log_scale is is scale*x + scale_shift, i.e. affine transformation
        self.scale_shift = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def get_mask(self, input_shape, top_condition):
        """
        Get checkboard mask
        """
        H, W = input_shape
        mask = np.arange(H).reshape(-1, 1) + np.arange(W)
        mask = np.mod(top_condition + mask, 2)
        mask = mask.reshape(-1, 1, H, W)
        return torch.from_numpy(mask).float()

    def forward(self, x):
        x_ = x * self.mask
        s, t = torch.chunk(self.cnn(x_), 2, dim=1)
        log_scale = self.scale * torch.tanh(s) + self.scale_shift

        t = t * (1.0 - self.mask)  # Will be zero for the non-dependant
        log_scale = log_scale * (1.0 - self.mask)  # Will be zero for the non-dependant

        z = x * torch.exp(log_scale) + t
        return z, log_scale

    def backward(self, z):
        z_ = z * self.mask
        s, t = torch.chunk(self.cnn(z_), 2, dim=1)

        log_scale = self.scale * torch.tanh(s) + self.scale_shift

        t = t * (1.0 - self.mask)  # Will be zero for the non-dependant
        log_scale = log_scale * (1.0 - self.mask)  # Will be zero for the non-dependant

        x = (z - t) * torch.exp(-log_scale)
        return x


class ChannelAffineCouplingLayer(nn.Module):
    """
    Coupling layer for RealNVP-class with channel-wise masking
    """

    def __init__(self, in_channels, conv_filters=128, residual_blocks=8, top_condition=True, input_shape=(32, 32)):
        super(ChannelAffineCouplingLayer, self).__init__()
        self.top_condition = top_condition

        self.cnn = ResidualCNN(2 * in_channels, 4 * in_channels, conv_filters, residual_blocks)
        self.scale = nn.Parameter(torch.tensor([1.]),
                                  requires_grad=True)  # log_scale is is scale*x + scale_shift, i.e. affine transformation
        self.scale_shift = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, x):
        N, C, H, W = x.shape
        first_channels, second_channels = x[:, :C // 2], x[:, C // 2:]

        if self.top_condition:
            s, t = torch.chunk(self.cnn(first_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            z = torch.cat((first_channels, second_channels * torch.exp(log_scale) + t), dim=1)
            jacobian = torch.cat((torch.zeros_like(log_scale), log_scale), dim=1)  # We only condition on firs
        else:
            s, t = torch.chunk(self.cnn(second_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            z = torch.cat((first_channels * torch.exp(log_scale) + t, second_channels), dim=1)
            jacobian = torch.cat((log_scale, torch.zeros_like(log_scale)),
                                 dim=1)  # We only condition on first 1/2 channels, so we get the identity matrix I of shape S

        return z, jacobian

    def backward(self, z):
        N, C, H, W = z.shape
        first_channels, second_channels = z[:, :C // 2], z[:, C // 2:]

        if self.top_condition:
            s, t = torch.chunk(self.cnn(first_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            x = torch.cat((first_channels, (second_channels - t) * torch.exp(-log_scale)), dim=1)

        else:
            s, t = torch.chunk(self.cnn(second_channels), 2, dim=1)
            log_scale = self.scale * torch.tanh(s) + self.scale_shift

            x = torch.cat(((first_channels - t) * torch.exp(-log_scale), second_channels), dim=1)

        return x


class RealNVP(nn.Module):
    """
    RealNVP implemented with coupling layers
    """

    def __init__(self, input_shape):
        super().__init__()
        C, H, W = input_shape
        modules = [[CheckboardAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(C)] for i in range(4)]
        modules.append([Squeeze()])
        modules += [[ChannelAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(4 * C)] for i in range(3)]
        modules.append([UnSqueeze()])
        modules += [[CheckboardAffineCouplingLayer(C, top_condition=i % 2 == 0), ActNorm(C)] for i in range(3)]

        modules = [layer for layer_types in modules for layer in layer_types]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        z = x
        log_det_jacobian = torch.zeros_like(x)

        for layer in self.net:
            if isinstance(layer, (Squeeze, UnSqueeze)):
                z = layer(z)
                log_det_jacobian = layer(log_det_jacobian)  # Need to reshape jacobian as well
            else:
                z, new_log_det_jacobian = layer(z)
                log_det_jacobian += new_log_det_jacobian

        return z, log_det_jacobian

    def backward(self, z):
        x = z
        for layer in list(self.net)[::-1]:
            x = layer.backward(x)

        return x


class ParallellWrapper(nn.DataParallel):
    def __init__(self, model, device_ids=[0, 1]):
        super(ParallellWrapper, self).__init__(model, device_ids=device_ids)

    def backward(self, z):
        return self.module.backward(z)


def train_realvnp_3d(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """
    start_time = time.time()
    torch.cuda.empty_cache()

    N, H, W, C = train_data.shape

    def logit_transform(x, dequantize=True, alpha=.05):
        """
        X is {0, 1, 2, 3}
        Dequantazion trick from Density Estimation using RealNVP paper
        """
        if dequantize:
            normal_noise = torch.distributions.uniform.Uniform(low=.0, high=1.)  # Gets values in [0, 1)
            x = x + normal_noise.rsample(x.shape).float().cuda()

        p = alpha + (1. - alpha) * x / 4  # P is in [0.05, 1)
        logit = torch.log(p) - torch.log(1. - p + 1e-9)
        log_logit_det = torch.log(torch.abs((1 / p + 1 / (1 - p + 1e-9)) * (1. - alpha) / 4.) + 1e-7)

        return logit, log_logit_det

    def nll_loss(batch, output, logit_log_determinant, alpha=.05):
        z, log_determinant = output

        # Adding the contribution from the dequantization
        log_determinant += logit_log_determinant

        gaussian = torch.distributions.normal.Normal(loc=.0, scale=1.)

        log_pz = gaussian.log_prob(z)
        log_p_theta = torch.sum((log_pz + log_determinant).view(batch.shape[0], -1), dim=1) / (C * H * W)
        return -torch.mean(log_p_theta)

    def get_batched_loss(data_loader, model):
        test_loss = []
        for batch in data_loader:
            batch, logit_log_determinant = logit_transform(batch.cuda())
            out = model(batch)
            loss = nll_loss(batch, out, logit_log_determinant)
            test_loss.append(loss.cpu().item())

        return np.mean(np.array(test_loss))

    # Create data loaders
    multiple_gpus = True
    batch_size = 64
    dataset_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    print("[INFO] Creating model and data loaders")
    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float()

    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    test_loader = torch.utils.data.DataLoader(test_data, **dataset_params)

    # Model
    n_epochs = 40
    input_shape = (C, H, W)
    model = RealNVP(input_shape=input_shape)
    realnvp = ParallellWrapper(model, device_ids=[0, 1]).cuda() if multiple_gpus else model.cuda()

    optimizer = torch.optim.Adam(realnvp.parameters(), lr=5e-4)

    # Training
    train_losses = []
    test_losses = [get_batched_loss(test_loader, realnvp).item()]

    print("[INFO] Training")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            batch, logit_log_determinant = logit_transform(batch.cuda())
            output = realnvp(batch)
            loss = nll_loss(batch, output, logit_log_determinant)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().item())

        test_loss = get_batched_loss(test_loader, realnvp)
        test_losses.append(test_loss.item())
        print(
            f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Test loss: {test_loss:.2f} - Time elapsed: {time.time() - epoch_start:.2f}")

    # Sampling and interpolation
    def sample(num_samples):
        z = torch.zeros(size=(num_samples, C, H, W)).normal_(0, 1).float().cuda()
        x = torch.zeros(size=(num_samples, C, H, W)).float().cuda()

        for i in range(num_samples):
            x[i] = torch.sigmoid(realnvp.backward(z[[i]])[0])

        x = x.detach().cpu().numpy()
        return np.transpose(x, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    def sample_interpolations(num_interpolation):
        """
        Returns {num_interpolation} interpolation of two images. Resulting in 6 * {num_interpolation} images
        """
        samples = torch.zeros(size=(num_interpolation * 6, C, H, W)).float().cuda()
        counter = 0
        for _ in range(num_interpolation):
            x_a, x_b = random.choice(train_data).unsqueeze(0).float().cuda(), random.choice(train_data).unsqueeze(
                0).float().cuda()
            x_a, _ = logit_transform(x_a, dequantize=False)
            x_b, _ = logit_transform(x_b, dequantize=False)

            z_a, log_det_a = realnvp(x_a)
            z_b, log_det_b = realnvp(x_b)

            samples[counter] = torch.sigmoid(x_a)
            counter += 1

            for weight in range(2, 10, 2):
                z_interpolated = (1.0 - weight * 0.1) * z_a + (weight * 0.1) * z_b
                x_interpolated = realnvp.backward(z_interpolated)

                samples[counter] = torch.sigmoid(x_interpolated[0])
                counter += 1

            samples[counter] = torch.sigmoid(x_b)
            counter += 1

        print(f"[INFO] Pixel range: max: {torch.max(samples).item()}, min: {torch.min(samples).item()}")
        samples = samples.detach().cpu().numpy()
        return np.transpose(samples, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    torch.cuda.empty_cache()
    realnvp.eval()
    with torch.no_grad():
        print("[INFO] Sampling")
        samples = sample(100)

        print("[INFO] Interpolating images")
        interpolation_samples = sample_interpolations(5)

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")

    return np.array(train_losses), np.array(test_losses), samples, interpolation_samples


def train_and_show_results_celeb_a():
    """
    Trains RealVNP for images and displays samples and training plot for Celeb A dataset
    """
    show_results_celeb_a(train_realvnp_3d)
