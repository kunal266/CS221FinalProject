import gym
from gym import error, spaces, utils
from gym.utils import seeding
from agario import AgarioGame, Player
import numpy as np
import math
from collections import namedtuple

from typing import List, Tuple, Any

State = namedtuple(typename='State', field_names=['grid', 'mass', 'zoom'])
Reward = int
Done = bool
Info = Any


class AgarioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 should_render: bool,
                 speed_scale: float,
                 display_text: bool,
                 should_display: bool,
                 grid_resolution: int,
                 skip_frames: int,
                 max_steps: int):
        self.should_render = should_render
        self.should_display = should_display
        self.game = AgarioGame(should_render=should_render,
                               speed_scale=speed_scale,
                               display_text=display_text,
                               should_display=should_display)
        self.grid_resolution = grid_resolution
        self.action_space = spaces.Discrete(16)
        self.skip_frames = skip_frames
        self.max_steps = max_steps
        self.steps = 0

    def seed(self, seed=None):
        self.game.seed(seed)

    def get_angle_from_action(self, action: int) -> float:
        return action * 2 * math.pi / (self.action_space.n + 1)

    #step and reward function is defined here 
    def step(self, action: int) -> Tuple[State, Reward, Done, Info]:
        angle = self.get_angle_from_action(action)
        old_mass = self.game.player.mass
        for _ in range(self.skip_frames):
            if self.should_display:
                self.render()
            self.game.step(angle)
        cells, adversaries, player = self.game.get_player_state()
        state = self.get_state(cells, adversaries, player)
        new_mass = self.game.player.mass
        reward = new_mass - old_mass
        done = self.game.game_ended
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
        return state, reward, done, 'info'

    def reset(self):
        self.game.reset()
        self.steps = 0
        cells, adversaries, player = self.game.get_player_state()
        return self.get_state(cells, adversaries, player)

    def get_state(self, cells, adversaries, player) -> State:
        grid = self.build_vision_grid(cells, adversaries, player.camera)
        return State(grid, player.mass, player.camera.zoom)

    def build_vision_grid(self, cells, adversaries, camera) -> np.array:
        grid_step_x = int(math.ceil(camera.width / self.grid_resolution))
        grid_step_y = int(math.ceil(camera.height / self.grid_resolution))
        all_cells = cells + adversaries
        grid = np.zeros((2, self.grid_resolution, self.grid_resolution))
        for cell in all_cells:
            x_rel, y_rel = camera.transform_pos(cell.x, cell.y)
            x_grid = int(x_rel // grid_step_x)
            y_grid = int(y_rel // grid_step_y)
            channel = 1 if isinstance(cell, Player) else 0
            grid[channel, y_grid, x_grid] += cell.mass
        return grid

    def render(self, mode='human'):
        return self.game.render()

    def close(self):
        self.game.close()
