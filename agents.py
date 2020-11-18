import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from agario_env import State
from collections import namedtuple


class Agent:
    def __init__(self):
        self.mass_path = []
        self.max_masses = []

    def save_performance(self, path):
        np.savetxt(path, self.max_masses)

    def print_final_stats(self):
        mean_max_mass = np.mean(self.max_masses)
        median_max_mass = np.median(self.max_masses)
        std_max_mass = np.std(self.max_masses)
        print(f'{type(self)} mean = {mean_max_mass}, std: {std_max_mass}, median = {median_max_mass}')

    def get_action(self, state):
        raise NotImplementedError()

    def plot_mass_path(self):
        plt.plot(self.mass_path)
        plt.title('Mass vs time')
        plt.xlabel('timestep')
        plt.ylabel('mass')
        plt.show()

    def plot_episoded_max_mass(self):
        plt.plot(self.max_masses)
        plt.xlabel('Episodes')
        plt.ylabel('Max mass')
        plt.show()


class RandomAgent(Agent):
    def get_action(self, state):
        return random.uniform(0, 2 * np.pi)


class Greedy(Agent):
    def __init__(self):
        super().__init__()
        self.last_angle = 0

    def get_angle_from_grid(self, shape, i, j):
        h, w = shape
        y_grid, x_grid = i + 0.5, j + 0.5
        y_centered, x_centered = y_grid - h / 2, x_grid - w / 2
        if y_centered == x_centered == 0:
            return self.last_angle
        angle = -math.atan2(y_centered, x_centered)
        return angle

    @staticmethod
    def get_best_indices(grid):
        h, w = grid.shape

        def mass_distance_ratio(flattened_index) -> float:
            i, j = np.unravel_index(flattened_index, (h, w))
            y_centered, x_centered = (i + 0.5) - (h / 2), (j + 0.5) - (w / 2)
            vector = np.array([x_centered, y_centered])
            distance = math.sqrt(vector @ vector)
            return grid[i, j] / distance

        best_index = max(range(h * w), key=lambda i: mass_distance_ratio(i))

        return np.unravel_index(best_index, grid.shape)

    def get_action(self, state: State) -> float:
        grid, mass, zoom = state
        # grid = grid[0] + grid[1]
        grid = grid[0]
        self.mass_path.append(mass)
        i, j = self.get_best_indices(grid)
        angle = self.get_angle_from_grid(grid.shape, i, j)
        self.last_angle = angle
        return angle


class DQN(nn.Module):
    def __init__(self, height, width, input_channels, num_outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv2d_size_out_final(size, kernel_sizes, strides):
            assert len(kernel_sizes) == len(strides)
            output_size = size
            for i in range(len(kernel_sizes)):
                output_size = conv2d_size_out(output_size, kernel_sizes[i], strides[i])
            return output_size

        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        kernel_sizes = (8, 4)
        strides = (4, 2)
        convw = conv2d_size_out_final(width, kernel_sizes, strides)
        convh = conv2d_size_out_final(height, kernel_sizes, strides)
        linear_input_size = (convw * convh) * 32 + 2
        self.fully_connected = nn.Linear(linear_input_size, num_outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state_batch: np.array):
        grids, masses, zooms = State(*zip(*state_batch))
        x = torch.from_numpy(np.array(grids)).float()
        masses = torch.tensor(masses).unsqueeze(1)
        zooms = torch.tensor(zooms).unsqueeze(1)
        # x.size() -> (N, 1, width, height)
        x = F.relu(self.bn1(self.conv1(x)))
        # x.size() -> (N, 16, ?, ?)
        x = F.relu(self.bn2(self.conv2(x)))
        # x.size() -> (N, 32, ?, ?)
        # x = F.relu(self.bn3(self.conv3(x)))
        # x.size() -> (N, 32, ?, ?)
        x = x.view(x.size(0), -1)
        # print(x.size())
        # print(masses.size())
        x = torch.cat([x, masses, zooms], dim=1)
        # print(f'x.size() = {x.size()}')
        preds = self.fully_connected(x)
        # print(f'preds.size() = {preds.size()}')
        return preds


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent):
    # BATCH_SIZE = 128
    BATCH_SIZE = 128
    GAMMA = 0.85
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    MEMORY_CAPACITY = 10000
    TARGET_UPDATE = 100
    LEARNING_RATE = 1e-2

    def __init__(self, height, width, input_channels, num_actions, loadpath: str):
        super().__init__()
        self.policy_net = DQN(height, width, input_channels, num_actions)
        self.target_net = DQN(height, width, input_channels, num_actions)
        if loadpath != '':
            self.policy_net.load_state_dict(torch.load(loadpath))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.LEARNING_RATE)
        self.memory = ReplayMemory(self.MEMORY_CAPACITY)
        self.num_actions = num_actions
        self.steps_done = 0

    def seed(self, seed):
        random.seed(seed)

    def epsilon(self):
        return self.EPS_END + (self.EPS_START - self.EPS_END) * \
               math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def action_to_angle(self, action):
        return (action / self.num_actions) * 2 * np.pi

    def angle_to_action(self, angle):
        action_raw = int((angle / (2 * np.pi)) * self.num_actions)
        return action_raw if angle > 0 else self.num_actions - 1 - abs(action_raw)

    def get_action(self, state: State):
        should_explore = random.random() > self.epsilon()
        # print(f'epsilon = {self.epsilon()}')
        if should_explore:
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                preds = self.policy_net(np.array([state], dtype=object))
                best_action = torch.argmax(preds).item()
                action = best_action
                # angle = (best_action / self.num_actions) * 2 * np.pi
                # action = angle
        self.steps_done += 1
        return action

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        # print(f'OPTIMIZING')
        transitions = self.memory.sample(self.BATCH_SIZE)
        # print(f'transitions = {transitions, transitions}')
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        states, actions, next_states, rewards = Transition(*zip(*transitions))
        non_final_mask = np.array(list(map(lambda s: s is not None, next_states)), dtype=np.bool)
        # print(f'non_final_mask.shape = {non_final_mask.shape}')
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_next_states = np.array([s for s in next_states if s is not None], dtype=object)
        # print(f'non_final_next_states.shape = {non_final_next_states.shape}')
        state_batch = np.array(states, dtype=object)
        # print(f'actions = {actions}')
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.int64).unsqueeze(1)
        # print(f'state_batch.shape = {state_batch.shape}')
        # print(f'action_batch.size() = {action_batch.size()}')
        # print(f'reward_batch.size() = {reward_batch.size()}')

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # preds = self.policy_net(state_batch)
        # print(f'preds.size() = {preds.size()}')
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # print(f'state_action_values.size() = {state_action_values.size()}')

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(1)
        # print(f'next_state_values.size() \t= {next_state_values.size()}')
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # print(f'expected_state_action_values.size() \t= {expected_state_action_values.size()}')

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # print(f'loss = {loss}')

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # print(f'FINISHED OPTIMIZING')
