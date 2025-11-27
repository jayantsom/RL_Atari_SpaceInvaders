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

def init_state(env):
    state, _ = env.reset()
    state = utils.preprocess_frame(state)
    state_queue = deque([state] * config.STACK_FRAMES, maxlen=config.STACK_FRAMES)
    return np.stack(state_queue, axis=0), state_queue

def update_state(state_queue, next_frame):
    state_queue.append(utils.preprocess_frame(next_frame))
    return np.stack(state_queue, axis=0)

def optimize_model(policy_net, target_net, optimizer):
    batch = memory.sample_memory(config.BATCH_SIZE)
    if batch is None:
        return 0.0, 0.0
    
    state_batch = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.uint8, device=config.DEVICE)
    action_batch = torch.tensor([e[1] for e in batch], dtype=torch.long, device=config.DEVICE)
    reward_batch = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=config.DEVICE)
    next_state_batch = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.uint8, device=config.DEVICE)
    done_batch = torch.tensor([e[4] for e in batch], dtype=torch.bool, device=config.DEVICE)

    current_q_values = model.dqn_forward(policy_net, state_batch).gather(1, action_batch.unsqueeze(1))
    
    with torch.no_grad():
        next_q_values = model.dqn_forward(target_net, next_state_batch).max(1)[0]
    
    expected_q_values = reward_batch + (config.GAMMA * next_q_values * ~done_batch)
    
    loss = utils.huber_loss(current_q_values.squeeze(), expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    avg_q = current_q_values.mean().item()
    return loss.item(), avg_q

def train():
    log_file = open('training_logs.txt', 'w')
    log_file.write("Frame,Reward,Epsilon,Loss,QValue\n")
    
    env = gym.make(
        config.ENV_NAME,
        frameskip=config.FRAME_SKIP,
        repeat_action_probability=config.REPEAT_ACTION_PROBABILITY
    )
    action_size = env.action_space.n
    
    policy_net = model.create_dqn(action_size)
    target_net = model.create_dqn(action_size)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.RMSprop(policy_net.parameters(), lr=config.LEARNING_RATE, alpha=0.95, eps=0.01)
    memory.init_memory(config.MEMORY_SIZE)
    
    frame_idx = 0
    episode_rewards = []
    loss_history = []
    q_history = []
    
    print(f"Starting training for {config.TOTAL_FRAMES} frames...")
    
    while frame_idx < config.TOTAL_FRAMES:
        state, state_queue = init_state(env)
        episode_reward = 0
        done = False
        
        while not done and frame_idx < config.TOTAL_FRAMES:
            epsilon = utils.get_epsilon(frame_idx, config)
            
            if random.random() > epsilon:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.uint8, device=config.DEVICE).unsqueeze(0)
                    q_values = model.dqn_forward(policy_net, state_tensor)
                    action = q_values.max(1)[1].item()
            else:
                action = random.randint(0, action_size - 1)
            
            next_frame, reward, done, truncated, _ = env.step(action)
            next_state = update_state(state_queue, next_frame)
            memory.push_memory(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            frame_idx += 1
            
            if frame_idx > config.REPLAY_START_SIZE and frame_idx % 4 == 0:
                loss, avg_q = optimize_model(policy_net, target_net, optimizer)
                loss_history.append(loss)
                q_history.append(avg_q)
            
            if frame_idx % config.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if frame_idx % (config.TOTAL_FRAMES // 100) == 0:
                avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                avg_loss = np.mean(loss_history[-100:]) if loss_history else 0
                avg_q_value = np.mean(q_history[-100:]) if q_history else 0
                
                log_file.write(f"{frame_idx},{avg_reward:.2f},{epsilon:.3f},{avg_loss:.4f},{avg_q_value:.3f}\n")
                log_file.flush()
                
                progress = (frame_idx / config.TOTAL_FRAMES) * 100
                print(f"Progress: {progress:.1f}% | Frame: {frame_idx} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f} | Avg Q: {avg_q_value:.3f}")
        
        episode_rewards.append(episode_reward)
        if len(episode_rewards) > 100:
            episode_rewards.pop(0)
    
    torch.save(policy_net.state_dict(), 'trained_model.pth')
    log_file.close()
    env.close()
    
    print("Training completed!")

if __name__ == "__main__":
    train()