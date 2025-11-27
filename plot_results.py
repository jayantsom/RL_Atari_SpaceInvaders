"""
Visualization script for DQN training results.
Creates smoothed plots of rewards, loss, Q-values, and exploration rate over time.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reading training log file and extracting metrics for plotting.
def read_logs(log_file):
    frames, rewards, epsilons, losses, q_values = [], [], [], [], []
    # Opening the training log file
    with open(log_file, 'r') as f:
        # Skipping header line
        lines = f.readlines()[1:]  
        # Processing each line in the log file
        for line in lines:
            if line.strip():
                # Splitting comma-separated values
                parts = line.strip().split(',')
                frames.append(int(parts[0]))
                rewards.append(float(parts[1]))
                epsilons.append(float(parts[2]))
                losses.append(float(parts[3]))
                q_values.append(float(parts[4]))
    
    return frames, rewards, epsilons, losses, q_values

# Smoothing data using moving average for better visualization.
def smooth_data(y, window_size=20):
    # Using pandas rolling window to calculate moving average
    return pd.Series(y).rolling(window=window_size, center=True, min_periods=1).mean()

# Plotting training metrics with smoothing for clarity.
def plot_training(log_file):
    # Reading data from training log file
    frames, rewards, epsilons, losses, q_values = read_logs(log_file)
    
    # Applying smoothing to all metrics with different window sizes
    smooth_rewards = smooth_data(rewards, window_size=20)
    smooth_losses = smooth_data(losses, window_size=30)
    smooth_q_values = smooth_data(q_values, window_size=20)
    smooth_epsilons = smooth_data(epsilons, window_size=10)
    
    # Creating a 2x2 grid of subplots for all metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Increasing spacing between subplots for better readability
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Plot 1: Rewards over training frames
    ax1.plot(frames, smooth_rewards, linewidth=2.5, color='#1f77b4', label='Smoothed Rewards')
    ax1.plot(frames, rewards, alpha=0.2, color='#1f77b4', label='Raw Rewards')
    ax1.set_title('Training Rewards (Moving Average)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Frames', fontsize=12, labelpad=10)
    ax1.set_ylabel('Average Reward', fontsize=12, labelpad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Exploration rate (epsilon) decay over time
    ax2.plot(frames, smooth_epsilons, linewidth=2.5, color='#2ca02c', label='Smoothed Epsilon')
    ax2.plot(frames, epsilons, alpha=0.2, color='#2ca02c', label='Raw Epsilon')
    ax2.set_title('Exploration Rate (Epsilon)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Frames', fontsize=12, labelpad=10)
    ax2.set_ylabel('Epsilon', fontsize=12, labelpad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training loss showing learning stability
    ax3.plot(frames, smooth_losses, linewidth=2.5, color='#d62728', label='Smoothed Loss')
    ax3.plot(frames, losses, alpha=0.2, color='#d62728', label='Raw Loss')
    ax3.set_title('Training Loss (Moving Average)', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Frames', fontsize=12, labelpad=10)
    ax3.set_ylabel('Loss', fontsize=12, labelpad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Q-values showing how the agent learns state values
    ax4.plot(frames, smooth_q_values, linewidth=2.5, color='#9467bd', label='Smoothed Q-Values')
    ax4.plot(frames, q_values, alpha=0.2, color='#9467bd', label='Raw Q-Values')
    ax4.set_title('Q-Values (Moving Average)', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Frames', fontsize=12, labelpad=10)
    ax4.set_ylabel('Average Q-Value', fontsize=12, labelpad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjusting layout and saving high-quality plot
    plt.tight_layout(pad=3.0)
    plt.savefig('training_plots_smooth.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    # Generating plots from the training log file
    plot_training('training_logs.txt')