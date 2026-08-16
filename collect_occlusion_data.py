# Stage 2: collect ground-truth occupancy grids for pretrain_occlusion_vae.py.
#
# Only label_grid (privileged ground truth) is collected here -- Label_VAE is the only
# network pretrained standalone (mirrors PaS-CrowdNav's vae_pretrain.py, which also only
# trains Label_VAE). Sensor_VAE is trained jointly with PPO later, directly against
# sensor_grid generated live during the main train.py run, so there's nothing to
# pre-collect for it.
#
# The robot has no scripted policy available yet at this stage (robot.policy is meant to
# be RL-trained, and no checkpoint exists before Stage 2's own training even starts), so
# episodes are driven by uniformly random actions -- this is the data-collection-time
# equivalent of PaS's ORCA-driven collection, just adapted since HEIGHT has no scripted
# robot option.

import os
import numpy as np

from crowd_nav.configs.config import Config
from crowd_sim import *
import gym


def collect_phase(env, phase, num_episodes, output_dir):
    phase_dir = os.path.join(output_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)

    existing = len([f for f in os.listdir(phase_dir) if f.endswith('_label_grid.npy')])
    if existing >= num_episodes:
        print(f'{phase}: {existing} episodes already collected -- skipping.')
        return

    for ep in range(existing, num_episodes):
        env.unwrapped.phase = phase
        obs = env.reset()
        label_grids = [obs['label_grid']]
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            label_grids.append(obs['label_grid'])

        label_grids = np.stack(label_grids, axis=0).astype(np.float32)  # (T, 1, grid_size, grid_size)
        np.save(os.path.join(phase_dir, f'{ep:05d}_label_grid.npy'), label_grids)

        if (ep + 1) % 20 == 0 or ep + 1 == num_episodes:
            print(f'{phase}: collected {ep + 1}/{num_episodes} episodes')


def main():
    config = Config()
    if config.sim.occlusion_method != 'vae':
        raise ValueError(
            "config.sim.occlusion_method must be 'vae' to collect occlusion grids "
            "(it controls whether generate_ob() populates ob['label_grid'] at all).")

    output_dir = 'data/occlusion_vae/VAEdata'
    os.makedirs(output_dir, exist_ok=True)

    try:
        env = gym.make(config.env.env_name, disable_env_checker=True)
    except TypeError:
        env = gym.make(config.env.env_name)
    env.configure(config)
    env.unwrapped.thisSeed = config.env.seed
    env.unwrapped.nenv = 1

    collect_phase(env, 'train', config.pas.collect_train_episodes, output_dir)
    collect_phase(env, 'val', config.pas.collect_val_episodes, output_dir)
    collect_phase(env, 'test', config.pas.collect_test_episodes, output_dir)

    print('Done. Data saved under', output_dir)


if __name__ == '__main__':
    main()
