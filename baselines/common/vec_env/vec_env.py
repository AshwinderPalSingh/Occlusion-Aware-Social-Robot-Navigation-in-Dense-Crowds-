from contextlib import contextmanager
class VecEnv:
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    def reset(self):
        raise NotImplementedError
    def step_async(self, actions):
        raise NotImplementedError
    def step_wait(self):
        raise NotImplementedError
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    def close(self):
        pass
class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(venv.num_envs, observation_space or venv.observation_space, action_space or venv.action_space)
    def reset(self):
        return self.venv.reset()
    def step_async(self, actions):
        return self.venv.step_async(actions)
    def step_wait(self):
        return self.venv.step_wait()
    def close(self):
        return self.venv.close()
    def render(self, *args, **kwargs):
        return self.venv.render(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.venv, name)
class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x
    def __call__(self):
        return self.x()
@contextmanager
def clear_mpi_env_vars():
    yield
