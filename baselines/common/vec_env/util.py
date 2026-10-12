from collections import OrderedDict
import numpy as np
from gym.spaces import Dict

def copy_obs_dict(obs):
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])
def dict_to_obs(*args):
    obs_dict = args[-1]
    return obs_dict if None not in obs_dict else obs_dict[None]
def obs_to_dict(obs):
    return obs if isinstance(obs, dict) else {None: obs}
def obs_space_info(obs_space):
    if isinstance(obs_space, Dict):
        keys = list(obs_space.spaces.keys())
        shapes = {k: obs_space.spaces[k].shape for k in keys}
        dtypes = {k: obs_space.spaces[k].dtype for k in keys}
    else:
        keys = [None]
        shapes = {None: obs_space.shape}
        dtypes = {None: obs_space.dtype}
    return keys, shapes, dtypes
