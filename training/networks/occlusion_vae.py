import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from training.networks.utils import init

"""
Label_VAE / Sensor_VAE, ported from PaS_CrowdNav's rl/pas_rnn_model.py (the mechanism
that produced the 94% success / 6% collision standalone PaS baseline).

v1 scope cut: PaS's Sensor_VAE takes a stack of config.pas.sequence (default 4) grid
frames as input channels, to accumulate information over time despite occlusion. HEIGHT's
port uses a single frame (1 input channel) to avoid rolling-buffer management across
parallel envs, so Sensor_VAE's encoder is architecturally identical to Label_VAE's here.
Revisit (restore multi-channel input + a per-env frame buffer) if a single frame proves
too little signal.

Training split (mirrors PaS's rl/ppo.py): Label_VAE is pretrained standalone
(pretrain_occlusion_vae.py) on privileged ground-truth grids, then loaded frozen
(requires_grad=False) into the policy network. Sensor_VAE is trained jointly with PPO,
pulled toward Label_VAE's latent via an MSE loss (see training/algo/ppo.py).
"""


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 1, -1)


class _View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Label_VAE(nn.Module):
    """Ground-truth grid autoencoder. Pretrained standalone, then frozen."""

    def __init__(self, config):
        super().__init__()
        latent_size = config.SRNN.occlusion_latent_size
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.encoder = nn.Sequential(
            init_(nn.Conv2d(1, 32, 4, 2, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, 2, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, 2, 1)), nn.BatchNorm2d(128), nn.ReLU(),
            init_(nn.Conv2d(128, 64, 4, 2, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 4, 2, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            _Flatten(),
            init_(nn.Linear(32 * 3 * 3, latent_size * 2)),
        )

        self.decoder = nn.Sequential(
            init_(nn.Linear(latent_size, 32 * 3 * 3)),
            _View((-1, 32, 3, 3)),
            nn.ReLU(), nn.BatchNorm2d(32),
            init_(nn.ConvTranspose2d(32, 64, 4, 2, 1)), nn.ReLU(), nn.BatchNorm2d(64),
            init_(nn.ConvTranspose2d(64, 128, 4, 2, 1)), nn.ReLU(), nn.BatchNorm2d(128),
            init_(nn.ConvTranspose2d(128, 64, 4, 2, 1, 1)), nn.ReLU(), nn.BatchNorm2d(64),
            init_(nn.ConvTranspose2d(64, 32, 4, 2, 1)), nn.ReLU(), nn.BatchNorm2d(32),
            init_(nn.ConvTranspose2d(32, 1, 4, 2, 1)), nn.Sigmoid(),
        )

        self.linear_mu = nn.Linear(latent_size * 2, latent_size)
        self.linear_var = nn.Linear(latent_size * 2, latent_size)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def encode(self, grid):
        x = self.encoder(grid)
        z_mu = self.linear_mu(x)
        z_log_variance = torch.log(self.softplus(self.linear_var(x)))
        z = self.reparameterize(z_mu, z_log_variance)
        return z_mu.squeeze(1), z_log_variance.squeeze(1), z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, grid):
        z_mu, z_log_variance, z = self.encode(grid)
        decoded = self.decode(z)
        return z_mu.squeeze(1), z_log_variance.squeeze(1), z, decoded


class Sensor_VAE(nn.Module):
    """Partial/occluded grid encoder. Trained jointly with PPO, distilled toward Label_VAE."""

    def __init__(self, config):
        super().__init__()
        latent_size = config.SRNN.occlusion_latent_size
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.encoder = nn.Sequential(
            init_(nn.Conv2d(1, 32, 4, 2, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, 2, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, 2, 1)), nn.BatchNorm2d(128), nn.ReLU(),
            init_(nn.Conv2d(128, 64, 4, 2, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 4, 2, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            _Flatten(),
            init_(nn.Linear(32 * 3 * 3, latent_size * 2)),
        )

        self.linear_mu = nn.Linear(latent_size * 2, latent_size)
        self.linear_var = nn.Linear(latent_size * 2, latent_size)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def encode(self, grid):
        x = self.encoder(grid)
        z_mu = self.linear_mu(x)
        z_log_variance = torch.log(self.softplus(self.linear_var(x)))
        z = self.reparameterize(z_mu, z_log_variance)
        return z_mu.squeeze(1), z_log_variance.squeeze(1), z
