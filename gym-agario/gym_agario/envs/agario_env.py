import gym
from gym import error, spaces, utils
from gym.utils import seeding


class AgarioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        return ['observation', 'reward', 'done', 'info']

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass
