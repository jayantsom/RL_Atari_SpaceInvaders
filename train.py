"""
Main training script for Deep Q-Network (DQN) on Space Invaders.
Implements the complete DQN algorithm with experience replay and target networks.
"""

import ale_py
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import config
import model
import memory
import utils

# Initializing the game state by resetting environment and stacking frames.
def init_state(env):
    # Resetting environment and preprocessing the first frame
    state, _ = env.reset()
    state = utils.preprocess_frame(state)
    # Creating a queue with multiple identical frames to start
    state_queue = deque([state] * config.STACK_FRAMES, maxlen=config.STACK_FRAMES)
    # Stacking frames along the channel dimension for network input
    return np.stack(state_queue, axis=0), state_queue

# Updating the state by adding new frame and removing oldest frame.
def update_state(state_queue, next_frame):
    # Adding new preprocessed frame to the queue
    state_queue.append(utils.preprocess_frame(next_frame))
    # Returning the updated stacked state
    return np.stack(state_queue, axis=0)

# Performing one optimization step using a batch of experiences.
def optimize_model(policy_net, target_net, optimizer):
    # Sampling a batch of experiences from replay memory
    batch = memory.sample_memory(config.BATCH_SIZE)
    if batch is None:
        return 0.0, 0.0
    
    # Converting batch data to PyTorch tensors on the correct device
    state_batch = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.uint8, device=config.DEVICE)
    action_batch = torch.tensor([e[1] for e in batch], dtype=torch.long, device=config.DEVICE)
    reward_batch = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=config.DEVICE)
    next_state_batch = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.uint8, device=config.DEVICE)
    done_batch = torch.tensor([e[4] for e in batch], dtype=torch.bool, device=config.DEVICE)

    # Calculating Q-values for taken actions
    current_q_values = model.dqn_forward(policy_net, state_batch).gather(1, action_batch.unsqueeze(1))
    
    # Calculating target Q-values using target network
    with torch.no_grad():
        next_q_values = model.dqn_forward(target_net, next_state_batch).max(1)[0]
    
    # Computing expected Q-values using Bellman equation
    expected_q_values = reward_batch + (config.GAMMA * next_q_values * ~done_batch)
    
    # Calculating loss between current and expected Q-values
    loss = utils.huber_loss(current_q_values.squeeze(), expected_q_values)
    
    # Performing gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Returning loss and average Q-value for monitoring
    avg_q = current_q_values.mean().item()
    return loss.item(), avg_q

# Main training loop for DQN algorithm.
def train():
    # Opening log file to record training progress
    log_file = open('training_logs.txt', 'w')
    log_file.write("Frame,Reward,Epsilon,Loss,QValue\n")
    
    # Creating the Space Invaders game environment
    env = gym.make(
        config.ENV_NAME,
        frameskip=config.FRAME_SKIP,
        repeat_action_probability=config.REPEAT_ACTION_PROBABILITY
    )
    action_size = env.action_space.n
    
    # Creating policy network and target network
    policy_net = model.create_dqn(action_size)
    target_net = model.create_dqn(action_size)
    # Initializing target network with policy network weights
    target_net.load_state_dict(policy_net.state_dict())
    
    # Setting up RMSprop optimizer for training the policy network
    optimizer = optim.RMSprop(policy_net.parameters(), lr=config.LEARNING_RATE, alpha=0.95, eps=0.01)

    # Initializing experience replay memory
    memory.init_memory(config.MEMORY_SIZE)
    
    frame_idx = 0
    episode_rewards = []
    loss_history = []
    q_history = []
    
    print(f"Starting training for {config.TOTAL_FRAMES} frames...")
    
    # Main training loop - running until target frame count is reached
    while frame_idx < config.TOTAL_FRAMES:
        state, state_queue = init_state(env)
        episode_reward = 0
        done = False
        
        # Running a single episode
        while not done and frame_idx < config.TOTAL_FRAMES:
            # Calculating current exploration rate
            epsilon = utils.get_epsilon(frame_idx, config)
            
            if random.random() > epsilon:
                # Exploitation: choosing best action according to policy network
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.uint8, device=config.DEVICE).unsqueeze(0)
                    q_values = model.dqn_forward(policy_net, state_tensor)
                    action = q_values.max(1)[1].item()
            else:
                # Exploration: choosing random action
                action = random.randint(0, action_size - 1)
            
            # Taking action in the environment
            next_frame, reward, done, truncated, _ = env.step(action)
            next_state = update_state(state_queue, next_frame)
            # Storing experience in replay memory
            memory.push_memory(state, action, reward, next_state, done)
            
            # Updating episode statistics
            episode_reward += reward
            state = next_state
            frame_idx += 1
            
            # Training the network periodically after enough experiences collected
            if frame_idx > config.REPLAY_START_SIZE and frame_idx % 4 == 0:
                loss, avg_q = optimize_model(policy_net, target_net, optimizer)
                loss_history.append(loss)
                q_history.append(avg_q)
            
            # Updating target network periodically for stable learning
            if frame_idx % config.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Logging progress every 1% of total training frames
            if frame_idx % (config.TOTAL_FRAMES // 100) == 0:
                avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                avg_loss = np.mean(loss_history[-100:]) if loss_history else 0
                avg_q_value = np.mean(q_history[-100:]) if q_history else 0
                
                # Writing metrics to log file
                log_file.write(f"{frame_idx},{avg_reward:.2f},{epsilon:.3f},{avg_loss:.4f},{avg_q_value:.3f}\n")
                log_file.flush()
                
                # Printing progress to console
                progress = (frame_idx / config.TOTAL_FRAMES) * 100
                print(f"Progress: {progress:.1f}% | Frame: {frame_idx} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f} | Avg Q: {avg_q_value:.3f}")
        
        # Storing episode reward and maintaining rolling window
        episode_rewards.append(episode_reward)
        if len(episode_rewards) > 100:
            episode_rewards.pop(0)
    
    # Saving the trained model and cleaning up
    torch.save(policy_net.state_dict(), 'trained_model.pth')
    log_file.close()
    env.close()
    
    print("Training completed!")

if __name__ == "__main__":
    # Starting the training process
    train()