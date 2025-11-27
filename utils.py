"""
Utility functions for the DQN Space Invaders project.
Contains image preprocessing, loss functions, and exploration scheduling.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Converting game frame to grayscale and resize for neural network input.
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

# Calculating Huber loss for stable Q-learning updates.
def huber_loss(pred, target):
    # Using PyTorch's smooth L1 loss which behaves like Huber loss
    return F.smooth_l1_loss(pred, target)

# Calculating current exploration rate using exponential decay.
def get_epsilon(frame_idx, config):
    # Computing exponential decay of exploration rate
    epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * np.exp(-1. * frame_idx / config.EPSILON_DECAY)
    # Ensuring epsilon doesn't go below minimum value
    return max(epsilon, config.EPSILON_END)