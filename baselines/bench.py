import gym
class Monitor(gym.Wrapper):
    def __init__(self, env, filename=None, allow_early_resets=True):
        super().__init__(env)
        self.filename = filename
        self.allow_early_resets = allow_early_resets
        self.rewards = []
    def reset(self, **kwargs):
        self.rewards = []
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]
        return result
    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        self.rewards.append(float(reward))
        if done:
            info = dict(info)
            info.setdefault('episode', {'r': sum(self.rewards), 'l': len(self.rewards)})
        return obs, reward, done, info
