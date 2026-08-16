# Stage 2: pretrain Label_VAE on ground-truth occupancy grids collected by
# collect_occlusion_data.py. Port of PaS-CrowdNav's vae_pretrain.py (same recipe: MSE
# reconstruction + KL, beta=0.00025, Adam lr=1e-3, batch=16, 300 epochs, checkpoint every
# 20 epochs). The resulting label_vae_weight_300.pth is loaded frozen by
# selfAttn_srnn_merge.py when config.sim.occlusion_method == 'vae'.
#
# Must reach epoch 300 -- train.py's Label_VAE loader hard-requires the exact _300.pth
# file (mirrors PaS's own train.py, which does the same for the same reason: a partial
# checkpoint from an interrupted run is not a validated VAE).

import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from crowd_nav.configs.config import Config
from training.networks.occlusion_vae import Label_VAE

NUM_EPOCHS = 300
BETA = 0.00025
LR = 1e-3
BATCH_SIZE = 16
CKPT_INTERVAL = 20


class GridDataset(Dataset):
    def __init__(self, phase_dir):
        files = sorted(glob.glob(os.path.join(phase_dir, '*_label_grid.npy')))
        if not files:
            raise FileNotFoundError(
                f'No *_label_grid.npy files found in {phase_dir}. Run collect_occlusion_data.py first.')
        frames = [np.load(f) for f in files]  # each (T, 1, grid_size, grid_size)
        self.data = np.concatenate(frames, axis=0).astype(np.float32)  # (N, 1, grid_size, grid_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def vae_loss(decoded, target, mu, logvar, batch_size):
    recon_loss = F.mse_loss(decoded, target, reduction='sum').div(batch_size)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss, kl_loss


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = 'data/occlusion_vae/VAEdata'
    train_loader = DataLoader(GridDataset(os.path.join(data_root, 'train')),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(GridDataset(os.path.join(data_root, 'val')),
                            batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = Label_VAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ckpt_dir = 'data/occlusion_vae/label_vae_ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Training Label_VAE for {NUM_EPOCHS} epochs on {len(train_loader.dataset)} grid frames '
          f'({len(train_loader)} batches/epoch)...')

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        model.train()
        recon_epoch, kl_epoch = 0., 0.
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z_mu, z_logvar, z, decoded = model(batch)
            recon_loss, kl_loss = vae_loss(decoded, batch, z_mu, z_logvar, batch.size(0))
            loss = recon_loss + BETA * kl_loss
            loss.backward()
            optimizer.step()
            recon_epoch += recon_loss.item()
            kl_epoch += kl_loss.item()
        recon_epoch /= len(train_loader)
        kl_epoch /= len(train_loader)

        model.eval()
        val_recon = 0.
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z_mu, z_logvar, z, decoded = model(batch)
                r, _ = vae_loss(decoded, batch, z_mu, z_logvar, batch.size(0))
                val_recon += r.item()
        val_recon /= max(1, len(val_loader))

        print(f'epoch {epoch:3d}/{NUM_EPOCHS}  train_recon={recon_epoch:.4f}  train_kl={kl_epoch:.4f}  '
              f'val_recon={val_recon:.4f}  ({time.time()-start:.1f}s)')

        if epoch % CKPT_INTERVAL == 0:
            ckpt_path = os.path.join(ckpt_dir, f'label_vae_weight_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  saved {ckpt_path}')

    final_ckpt = os.path.join(ckpt_dir, f'label_vae_weight_{NUM_EPOCHS}.pth')
    if not os.path.exists(final_ckpt):
        raise RuntimeError(
            f'Training loop completed but {final_ckpt} is missing -- something is wrong '
            f'with the checkpoint schedule (CKPT_INTERVAL={CKPT_INTERVAL} must divide NUM_EPOCHS={NUM_EPOCHS}).')
    print(f'\nDone. {final_ckpt} is ready to be set as config.pas.label_vae_ckpt.')


if __name__ == '__main__':
    main()
