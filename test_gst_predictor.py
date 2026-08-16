# Standalone smoke test for GSTPredictor -- run this on the DGX BEFORE setting
# sim.predict_method='inferred' and launching a full training run. Isolates
# checkpoint-loading/shape bugs from a 28-process PyBullet startup.
#
# Synthetic scenario (matches the source repo's own __main__ self-test in
# crowd_nav_interface_multi_env_parallel.py): 4 pedestrians walking in straight lines
# (right, left, up, down) toward/away from the origin at a constant step size. A sane
# predictor should continue each pedestrian roughly along its observed line.

import numpy as np
import torch

from training.prediction.gst_predictor import GSTPredictor

MODEL_DIR = 'data/gst_predictor'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    predictor = GSTPredictor(MODEL_DIR, device)
    print('Loaded GST predictor. obs_seq_len =', predictor.obs_seq_len,
          'pred_seq_len =', predictor.pred_seq_len)

    obs_seq_len = predictor.obs_seq_len
    step = 0.56
    x_init = 2.0
    t = np.arange(obs_seq_len - 1, -1, -1) * step  # oldest -> newest, decreasing distance

    ped_right = np.stack((x_init + t, np.zeros(obs_seq_len)), axis=1)   # walking toward +x -> -x... i.e. approaching origin from +x
    ped_left = -ped_right
    ped_up = np.stack((np.zeros(obs_seq_len), x_init + t), axis=1)
    ped_down = -ped_up

    history_pos = np.stack((ped_right, ped_left, ped_up, ped_down), axis=1)  # (obs_seq_len, 4, 2)
    print('history_pos shape:', history_pos.shape)

    predicted_pos = predictor.predict(history_pos)
    print('predicted_pos shape:', predicted_pos.shape)
    assert predicted_pos.shape == (predictor.pred_seq_len, 4, 2), \
        f'unexpected output shape: {predicted_pos.shape}'
    assert not np.isnan(predicted_pos).any(), 'predicted_pos contains NaNs'

    print('\nLast observed positions:')
    print(history_pos[-1])
    print('\nPredicted future positions (should continue roughly along each line):')
    print(predicted_pos)

    print('\nGSTPredictor smoke test PASSED.')


if __name__ == '__main__':
    main()
