"""
Script for testing and visualizing trained DQN model performance.
Loads a saved model and plays Space Invaders to evaluate learned behavior.
"""

import ale_py
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import config
import model
import utils

# Initializing the game state by resetting environment and stacking frames.
def init_state(env):
    # Resetting the game environment to start state
    state, _ = env.reset()
    # Preprocessing the raw frame for neural network input
    state = utils.preprocess_frame(state)
    # Creating a queue with multiple frames for temporal context
    state_queue = deque([state] * config.STACK_FRAMES, maxlen=config.STACK_FRAMES)
    # Stacking frames along channel dimension for network input
    return np.stack(state_queue, axis=0), state_queue

# Updating the state by adding new frame and removing oldest frame.
def update_state(state_queue, next_frame):
    # Adding new preprocessed frame to the queue
    state_queue.append(utils.preprocess_frame(next_frame))
    # Returning the updated stacked state
    return np.stack(state_queue, axis=0)

# Playing the game using the trained model for a specified number of episodes.
def play_game(model_path, episodes=10, render=True):
    # Creating the game environment with rendering enabled
    env = gym.make(
        config.ENV_NAME,
        render_mode='human' if render else None,
        frameskip=config.FRAME_SKIP,
        repeat_action_probability=config.REPEAT_ACTION_PROBABILITY
    )
    action_size = env.action_space.n
    
    # Loading the trained neural network model
    policy_net = model.create_dqn(action_size)
    policy_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    # Setting model to evaluation mode for inference
    policy_net.eval()
    
    print(f"Available actions: {env.unwrapped.get_action_meanings()}")
    
    all_scores = []
    
    # Playing multiple episodes to evaluate model performance
    for episode in range(episodes):
        state, state_queue = init_state(env)
        total_reward = 0
        steps = 0
        # Tracking which actions the model chooses
        action_counts = {i: 0 for i in range(action_size)}
        
        # Running a single episode until game ends
        while True:
            # Converting state to tensor format for neural network
            state_tensor = torch.tensor(state, dtype=torch.uint8, device=config.DEVICE).unsqueeze(0)
            
            # Getting Q-values and selecting best action
            with torch.no_grad():
                q_values = model.dqn_forward(policy_net, state_tensor)
                action = q_values.max(1)[1].item()
            
            # Recording the chosen action
            action_counts[action] += 1
            # Taking action in the environment
            next_frame, reward, done, truncated, _ = env.step(action)
            # Updating state with new frame
            next_state = update_state(state_queue, next_frame)
            
            # Accumulating reward and step count
            total_reward += reward
            steps += 1
            state = next_state
            
            # Ending episode when game is over
            if done or truncated:
                break
        
        # Storing episode results for analysis
        all_scores.append(total_reward)
        print(f'Episode {episode + 1}: Score = {total_reward}, Steps = {steps}')
        print(f'Action distribution: {action_counts}')
    
    # Closing the game environment
    env.close()
    
    # Displaying summary statistics across all episodes
    print(f"\nSummary - Average Score: {np.mean(all_scores):.1f} ± {np.std(all_scores):.1f}")
    print(f"Min Score: {min(all_scores)}, Max Score: {max(all_scores)}")

if __name__ == "__main__":
    # Running the game with the trained model
    play_game('trained_model.pth', episodes=10)