import torch
import torch.nn as nn
from utils.utils import Flatten


def get_cnn_encoder(image_channels=3):
    return nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=2, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            Flatten()
        )

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=1024, num_labels=17):
        super().__init__()
        self.cnn = get_cnn_encoder(image_channels=3) # Currently this returns only for 1024 hidden dimensions. Need to change that
        # setup the two linear transformations used
        self.fc21 = nn.Linear(hidden_dim+num_labels, z_dim)
        self.fc22 = nn.Linear(hidden_dim+num_labels, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x,y):
        '''
        Here if i get an array of [xs, ys] what should i do ?
        xs is gonna be of the shape (32, 3, 400,400) and ys is gonna be of the shape (32,10)
        '''
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.reshape(-1, 40000)
        # then compute the hidden units
        hidden = self.cnn(x)
        hidden = self.softplus(hidden) # This should return a [1, 1024] vector.
        # then return a mean vector and a (positive) square root covariance

        # each of size batch_size x z_dim
        hidden = torch.cat([hidden, y], dim=-1)
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale