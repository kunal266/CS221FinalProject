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



class DQN(nn.Module):
    def __init__(self, height, width, input_channels, num_outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        
        

        
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv2d_size_out_final(size, kernel_sizes, strides):
            assert len(kernel_sizes) == len(strides)
            output_size = size
            for i in range(len(kernel_sizes)):
                output_size = conv2d_size_out(output_size, kernel_sizes[i], strides[i])
            return output_size

        
        
        kernel_sizes = (8, 4)
        strides = (4, 2)
        convw = conv2d_size_out_final(width, kernel_sizes, strides)
        convh = conv2d_size_out_final(height, kernel_sizes, strides)
        linear_input_size = (convw * convh) * 32 + 2
        self.fully_connected = nn.Linear(linear_input_size, num_outputs)

    
    
    def forward(self, state_batch: np.array):
        grids, masses, zooms = State(*zip(*state_batch))
        x = torch.from_numpy(np.array(grids)).float()
        masses = torch.tensor(masses).unsqueeze(1)
        zooms = torch.tensor(zooms).unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        
        
        x = x.view(x.size(0), -1)
        
        
        x = torch.cat([x, masses, zooms], dim=1)
        
        preds = self.fully_connected(x)
        
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
        
        if should_explore:
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                preds = self.policy_net(np.array([state], dtype=object))
                best_action = torch.argmax(preds).item()
                action = best_action
                
                
        self.steps_done += 1
        return action

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        
        
        
        states, actions, next_states, rewards = Transition(*zip(*transitions))
        non_final_mask = np.array(list(map(lambda s: s is not None, next_states)), dtype=np.bool)
        
        
        non_final_next_states = np.array([s for s in next_states if s is not None], dtype=object)
        
        state_batch = np.array(states, dtype=object)
        
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.int64).unsqueeze(1)
        
        
        

        
        
        
        
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        

        
        
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(1)
        
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        

        
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        

        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
