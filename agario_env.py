import gym
from gym import error, spaces, utils
from gym.utils import seeding
from agario import AgarioGame


class AgarioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game = AgarioGame(render=True, speed_scale=2, display_text=True)
        self.skip_frames = 1

    def seed(self, seed=None):
        self.game.seed(seed)

    def step(self, action):
        old_mass = self.game.player.mass
        for _ in range(self.skip_frames):
            self.game.step(action)
        state = self.game.get_player_state()
        new_mass = self.game.player.mass
        reward = new_mass - old_mass
        done = self.game.game_ended
        return [state, reward, done, 'info']

    def reset(self):
        self.game.reset()
        return self.game.get_player_state()

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        self.game.close()
