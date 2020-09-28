import torch
import torch.nn as nn
from utils.utils import UnFlatten



def get_seq_decoder(hidden_dim=1024, image_channels=3):
    return nn.Sequential(
            UnFlatten(), # (32, 1024, 1, 1)
            nn.ConvTranspose2d(hidden_dim, 512, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=13, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, image_channels, kernel_size=2, stride=1),
            nn.Sigmoid() # (32, 3, 400,400)
        )


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_labels=17):
        super().__init__()
        self.cnn_decoder = get_seq_decoder(hidden_dim, 3) # image_channels is 3
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim+num_labels, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc21 = nn.Linear(hidden_dim, 400)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z, y):
        # define the forward computation on the latent z
        # first compute the hidden units
        concat_z = torch.cat([z, y], dim=-1)
        hidden = self.softplus(self.fc1(concat_z))
        #hidden = self.softplus(self.fc2(hidden))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.cnn_decoder(hidden)
        return loc_img