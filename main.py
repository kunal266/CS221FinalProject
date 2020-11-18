import math
import matplotlib.pyplot as plt
from agario_env import AgarioEnv, State
from agents import Greedy, DQNAgent, RandomAgent
import torch
from datetime import datetime

RENDER = True
DISPLAY_TEXT = False
SPEED_SCALE = 2
# GRID_RESOLUTION = 84
GRID_RESOLUTION = 32
NUM_SKIP_FRAMES = 10
ACTION_DISCRETIZATION = 16
NUM_EPISODES = 500
MAX_STEPS = int(200)


def main_random():
    env = AgarioEnv(render=RENDER,
                    speed_scale=SPEED_SCALE,
                    display_text=DISPLAY_TEXT,
                    grid_resolution=GRID_RESOLUTION)
    agent = RandomAgent()
    for episode in range(NUM_EPISODES):
        state = env.reset()
        num_steps = 0
        done = False
        while True:
            action = agent.get_action(state)
            for _ in range(NUM_SKIP_FRAMES):
                if RENDER:
                    env.render()
                state, reward, done, _ = env.step(action)
            if done or num_steps >= MAX_STEPS:
                print(f'epoch: {episode}, max_mass = {state.mass}')
                agent.max_masses.append(state.mass)
                break
            num_steps += 1
    agent.save_performance(path='random.performance')
    agent.print_final_stats()
    env.close()


def main_DQN():
    env = AgarioEnv(render=RENDER,
                    speed_scale=SPEED_SCALE,
                    display_text=DISPLAY_TEXT,
                    grid_resolution=GRID_RESOLUTION)
    agent = DQNAgent(height=GRID_RESOLUTION,
                     width=GRID_RESOLUTION,
                     input_channels=2,
                     num_actions=ACTION_DISCRETIZATION,
                     loadpath='model_210_2020-11-17_18:20:07.154462_episodes.model')
    # env.seed(41)
    # agent.seed(41)
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
            # print(f'step = {num_steps}')
            if done or num_steps > MAX_STEPS:
                new_state = None
                done = True
            agent.memory.push(state, raw_action, new_state, reward)
            agent.optimize()
            if done:
                print(f'Episode done, max_mass = {state.mass}')
                agent.max_masses.append(state.mass)
            if num_steps % agent.TARGET_UPDATE == 0:
                # print(f'UPDATING TARGET')
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            state = new_state
    print(f'Complete')
    torch.save(agent.policy_net.state_dict(),
               f'model_{210 + NUM_EPISODES}_{str(datetime.now()).replace(" ", "_")}_episodes.model')
    # agent.plot_episoded_max_mass()
    agent.print_final_stats()
    env.close()


def main_DQN_plus_greedy():
    GREEDY_TOTAL_NUM_EPISODES = 10
    GREEDY_NUM_EPISODES = GREEDY_TOTAL_NUM_EPISODES // 3
    env = AgarioEnv(render=RENDER,
                    speed_scale=SPEED_SCALE,
                    display_text=DISPLAY_TEXT,
                    grid_resolution=GRID_RESOLUTION)
    agent = DQNAgent(height=GRID_RESOLUTION,
                     width=GRID_RESOLUTION,
                     input_channels=2,
                     num_actions=ACTION_DISCRETIZATION,
                     loadpath='')
    greedy = Greedy()
    env.seed(41)
    agent.seed(41)
    for episode in range(GREEDY_TOTAL_NUM_EPISODES):
        state = env.reset()
        done = False
        new_state = None
        raw_action, action = None, None
        reward = 0
        num_steps = 0
        is_greedy_episode = episode < GREEDY_NUM_EPISODES
        while not done:
            if is_greedy_episode:
                action = greedy.get_action(state)
                raw_action = agent.angle_to_action(action)
                # print(f'angle: {action}, raw_action: {raw_action}')
            else:
                raw_action = agent.get_action(state)
                action = agent.action_to_angle(raw_action)
            for _ in range(NUM_SKIP_FRAMES):
                if RENDER:
                    env.render()
                new_state, reward, done, _ = env.step(action)
            num_steps += 1
            # print(f'step = {num_steps}')
            if done or num_steps > MAX_STEPS:
                new_state = None
                done = True
            agent.memory.push(state, raw_action, new_state, reward)
            agent.optimize()
            if done:
                print(f'{"Greedy" if is_greedy_episode else "DQN" } episode done, max_mass: {state.mass}')
                if not is_greedy_episode:
                    agent.max_masses.append(state.mass)
            if num_steps % agent.TARGET_UPDATE == 0:
                # print(f'UPDATING TARGET')
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            state = new_state
    print(f'Complete')
    torch.save(agent.policy_net.state_dict(),
               f'model_GREEDY_DQN_{NUM_EPISODES}_{str(datetime.now()).replace(" ", "_")}_episodes.model')
    agent.print_final_stats()
    env.close()


if __name__ == '__main__':
    main_DQN()
    # main_DQN_plus_greedy()
