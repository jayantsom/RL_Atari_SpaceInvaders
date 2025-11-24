import ale_py
import gymnasium as gym
import random

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

print(f'\nACTIONS: {env.unwrapped.get_action_meanings()}')

episodes = 5
for episode in range(1, episodes+1):
    state, info = env.reset()
    done = False
    score = 0

    while not done:
        action = random.randint(0, actions - 1)
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()