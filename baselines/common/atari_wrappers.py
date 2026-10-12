def make_atari(env_id):
    import gym
    return gym.make(env_id)
def wrap_deepmind(env):
    return env
