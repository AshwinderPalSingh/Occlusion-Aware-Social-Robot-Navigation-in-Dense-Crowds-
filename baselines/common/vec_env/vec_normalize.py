from .vec_env import VecEnvWrapper
class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, *args, **kwargs):
        super().__init__(venv)
        self.ob_rms = None
        self.ret_rms = None
        self.clipob = 10.0
        self.epsilon = 1e-8
    def _obfilt(self, obs, update=True):
        return obs
