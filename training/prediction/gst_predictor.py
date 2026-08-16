import os
import pickle

import numpy as np
import torch

from training.prediction.gumbel_social_transformer.st_model import st_model
from training.prediction.mgnn.utils import seq_to_graph

"""
Wraps a pretrained GST (Gumbel Social Transformer) trajectory predictor for per-environment,
synchronous use inside calc_human_future_traj(method='inferred') in crowd_sim_var_human.py.

Ported from CrowdNav_Prediction_AttnGraph/gst_updated/scripts/wrapper/
crowd_nav_interface_multi_env_parallel.py's CrowdNavPredInterfaceMultiEnv, simplified for:
- single-env (batch=1) calls, instead of that class's multi-env block-diagonal batching.
- the "always-valid, left-padded" history convention used by the caller here: unlike the
  source repo (which models partially-observed pedestrians via a -999 sentinel + validity
  mask), callers always provide a full obs_seq_len history (left-padded by repeating the
  earliest known position when a human/episode is too new to have real history yet), so the
  source's invalid-value/masking machinery collapses to "always valid" and is omitted here.
The core prediction math -- relative-displacement graph construction via the vendored
seq_to_graph(), and converting the model's predicted per-step deltas back to absolute
positions via cumsum + last observed position -- is kept exactly as the source computes it.
"""


class GSTPredictor:
    def __init__(self, model_dir, device):
        checkpoint_dir = os.path.join(model_dir, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)
        self.device = device
        self.obs_seq_len = self.args.obs_seq_len
        self.pred_seq_len = self.args.pred_seq_len

        self.model = st_model(self.args, device=device).to(device)
        ckpt_path = os.path.join(checkpoint_dir, f'epoch_{self.args.num_epochs}.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, history_pos):
        """
        history_pos: numpy array (obs_seq_len, num_humans, 2), absolute positions,
            oldest first / most recent last, already subsampled and left-padded by the
            caller so every entry is a real (possibly repeated) position -- never a
            sentinel/invalid value.
        Returns: numpy array (pred_seq_len, num_humans, 2), predicted absolute positions.
        """
        obs_seq_len = self.obs_seq_len
        num_peds = history_pos.shape[1]
        assert history_pos.shape == (obs_seq_len, num_peds, 2), \
            f'expected ({obs_seq_len}, num_peds, 2), got {history_pos.shape}'

        obs_traj = np.transpose(history_pos, (1, 2, 0))  # (num_peds, 2, obs_seq_len)
        obs_traj_rel = obs_traj[:, :, 1:] - obs_traj[:, :, :-1]
        obs_traj_rel = np.concatenate((np.zeros((num_peds, 2, 1)), obs_traj_rel), axis=2)

        obs_traj_t = torch.from_numpy(obs_traj).float()
        obs_traj_rel_t = torch.from_numpy(obs_traj_rel).float()

        # seq_to_graph expects 3D (num_peds, 2, obs_seq_len) here (matches how the source
        # wrapper calls it, before its own later batch-dim unsqueeze).
        v_obs, A_obs = seq_to_graph(obs_traj_t, obs_traj_rel_t, attn_mech='rel_conv')
        # v_obs: (obs_seq_len, num_peds, 2), A_obs: (obs_seq_len, num_peds, num_peds, 2)

        attn_mask = torch.ones(obs_seq_len, num_peds, num_peds)
        loss_mask_rel = torch.ones(1, num_peds, obs_seq_len + self.pred_seq_len)

        v_obs = v_obs.unsqueeze(0).to(self.device)
        A_obs = A_obs.unsqueeze(0).to(self.device)
        attn_mask = attn_mask.unsqueeze(0).to(self.device)
        loss_mask_rel = loss_mask_rel.to(self.device)

        with torch.no_grad():
            gaussian_params_pred, _, _ = self.model(
                v_obs, A_obs, attn_mask, loss_mask_rel,
                tau=0.03, hard=True, sampling=False, device=self.device)
        mu, _, _, _ = gaussian_params_pred  # (1, pred_seq_len, num_peds, 2), per-step deltas

        # model predicts per-step deltas; cumsum -> offset from the last observed position,
        # then add that position back to get absolute predicted positions (matches source
        # wrapper's mu_cumsum = mu.cumsum(1) + last_observed_position).
        mu_cumsum = mu.detach().cpu().numpy().cumsum(axis=1)[0]  # (pred_seq_len, num_peds, 2)
        last_obs_pos = history_pos[-1]  # (num_peds, 2)
        predicted_pos = mu_cumsum + last_obs_pos[np.newaxis, :, :]
        return predicted_pos
