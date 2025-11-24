import ale_py
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import config
import model
import utils

def init_state(env):
    state, _ = env.reset()
    state = utils.preprocess_frame(state)
    state_queue = deque([state] * config.STACK_FRAMES, maxlen=config.STACK_FRAMES)
    return np.stack(state_queue, axis=0), state_queue

def update_state(state_queue, next_frame):
    state_queue.append(utils.preprocess_frame(next_frame))
    return np.stack(state_queue, axis=0)

def play_game(model_path, episodes=10, render=True):
    env = gym.make(
        config.ENV_NAME,
        render_mode='human' if render else None,
        frameskip=config.FRAME_SKIP,
        repeat_action_probability=config.REPEAT_ACTION_PROBABILITY
    )
    action_size = env.action_space.n
    
    policy_net = model.create_dqn(action_size)
    policy_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    policy_net.eval()
    
    print(f"Available actions: {env.unwrapped.get_action_meanings()}")
    
    all_scores = []
    
    for episode in range(episodes):
        state, state_queue = init_state(env)
        total_reward = 0
        steps = 0
        action_counts = {i: 0 for i in range(action_size)}
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.uint8, device=config.DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                q_values = model.dqn_forward(policy_net, state_tensor)
                action = q_values.max(1)[1].item()
            
            action_counts[action] += 1
            next_frame, reward, done, truncated, _ = env.step(action)
            next_state = update_state(state_queue, next_frame)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or truncated:
                break
        
        all_scores.append(total_reward)
        print(f'Episode {episode + 1}: Score = {total_reward}, Steps = {steps}')
        print(f'Action distribution: {action_counts}')
    
    env.close()
    
    print(f"\nSummary - Average Score: {np.mean(all_scores):.1f} ± {np.std(all_scores):.1f}")
    print(f"Min Score: {min(all_scores)}, Max Score: {max(all_scores)}")

if __name__ == "__main__":
    play_game('trained_model.pth', episodes=10)