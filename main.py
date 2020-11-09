import gym

env = gym.make('gym_agario:agario-v0')
# env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    # action = env.action_space.sample()
    received = env.step(None)  # take a random action
    print(received)
env.close()
