import math
import matplotlib.pyplot as plt
from agario_env import AgarioEnv, State
from agents import DQNAgent
import torch
from datetime import datetime
import numpy as np

RENDER = False
DISPLAY_TEXT = False
SPEED_SCALE = 3

GRID_RESOLUTION = 32
NUM_SKIP_FRAMES = 10
ACTION_DISCRETIZATION = 16
NUM_EPISODES = 20
WEIGHTS_SAVE_EPISODE_STEP = 5
MAX_STEPS = int(200)

def main_DQN():
    env = AgarioEnv(should_render=RENDER,
                    speed_scale=SPEED_SCALE,
                    display_text=DISPLAY_TEXT,
                    grid_resolution=GRID_RESOLUTION,
                    should_display=False,
                    skip_frames=10,
                    max_steps=100)
    agent = DQNAgent(height=GRID_RESOLUTION,
                     width=GRID_RESOLUTION,
                     input_channels=2,
                     num_actions=ACTION_DISCRETIZATION,
                     loadpath='DQN_weights/model_0_episodes.model')
    
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        new_state = None
        reward = 0
        num_steps = 0
        while not done:
            raw_action = agent.get_action(state)
            action = agent.action_to_angle(raw_action)
            for _ in range(NUM_SKIP_FRAMES):
                if RENDER:
                    env.render()
                new_state, reward, done, _ = env.step(action)
            num_steps += 1
            
            if done or num_steps > MAX_STEPS:
                new_state = None
                done = True
            agent.memory.push(state, raw_action, new_state, reward)
            agent.optimize()
            if done:
                print(f'Episode {episode} done, max_mass = {state.mass}')
                agent.max_masses.append(state.mass)
                agent.print_final_stats()
            if num_steps % agent.TARGET_UPDATE == 0:
                
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            state = new_state
        if episode % WEIGHTS_SAVE_EPISODE_STEP == 0:
            torch.save(agent.policy_net.state_dict(),
               f'DQN_weights/model_{episode}_episodes.model')
            np.savetxt(f'DQN_weights/model_{episode}_episodes.performance',
                        np.array(agent.max_masses))
    print(f'Complete')
    torch.save(agent.policy_net.state_dict(),
               f'DQN_weights/model_{NUM_EPISODES}_episodes.model')
    np.savetxt(f'DQN_weights/model_{NUM_EPISODES}_episodes.performance',
                        np.array(agent.max_masses))
    agent.print_final_stats()
    env.close()

if __name__ == '__main__':
    
    main_DQN()
    
