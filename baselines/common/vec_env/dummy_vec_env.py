import numpy as np
from .vec_env import VecEnv
from .util import obs_space_info, dict_to_obs
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)
        self.keys, self.shapes, self.dtypes = obs_space_info(env.observation_space)
        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
    def _save_obs(self, e, obs):
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]
        if isinstance(obs, dict):
            for k in self.keys:
                self.buf_obs[k][e] = obs[k]
        else:
            self.buf_obs[None][e] = obs
    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, {k: np.copy(v) for k, v in self.buf_obs.items()})
    def reset(self):
        for e, env in enumerate(self.envs):
            self._save_obs(e, env.reset())
        return self._obs_from_buf()
    def step_async(self, actions):
        self.actions = actions
    def step_wait(self):
        for e, env in enumerate(self.envs):
            action = self.actions[e] if np.ndim(self.actions) > 0 else self.actions
            result = env.step(action)
            if isinstance(result, tuple) and len(result) == 5:
                obs, rew, term, trunc, info = result
                done = term or trunc
            else:
                obs, rew, done, info = result
            self.buf_rews[e] = rew
            self.buf_dones[e] = done
            self.buf_infos[e] = info
            if done:
                obs = env.reset()
            self._save_obs(e, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), list(self.buf_infos)
    def close(self):
        for env in self.envs:
            env.close()
    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)
