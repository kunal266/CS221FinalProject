import math
import matplotlib.pyplot as plt
from agario_env import AgarioEnv
import torch

RENDER = True
DISPLAY_TEXT = True

class Agent:
    def __init__(self, initial_state):
        _, _, self.player = initial_state
        self.last_angle = 0
        self.mass_path = []

    def get_action(self, state):
        cells, adversaries, player = state
        self.mass_path += [player.mass]
        all_cells = cells + adversaries
        all_smaller_cells = list(filter(lambda cell: cell.mass < self.player.mass, all_cells))
        if not all_smaller_cells:
            return self.last_angle
        best_cell = max(all_smaller_cells, key=lambda cell: cell.mass / (cell.distance(cell, self.player) + 0.001))
        best_x, best_y = self.player.camera.transform_and_center_pos(best_cell.x, best_cell.y)
        angle = -math.atan2(best_y, best_x)
        return angle

    def plot_mass_path(self):
        plt.plot(self.mass_path)
        plt.title('Mass vs time')
        plt.xlabel('timestep')
        plt.ylabel('mass')
        plt.show()


def main():
    env = AgarioEnv(render=RENDER, speed_scale=1, display_text=DISPLAY_TEXT)
    env.seed(40)
    state = env.reset()
    done = False
    agent = Agent(state)
    cumulative_reward = 0
    while not done:
        if RENDER:
            env.render()
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        if reward != 0:
            cumulative_reward += reward
            print(f'reward = {reward}')
    print(f'ALL DONE: total_reward = {cumulative_reward}')
    agent.plot_mass_path()
    env.close()


if __name__ == '__main__':
    main()
