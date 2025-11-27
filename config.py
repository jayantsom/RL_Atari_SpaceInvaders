"""
Configuration file for DQN Space Invaders training.
Contains all hyperparameters and environment settings in one place.
"""
import torch

# Environment settings for Space Invaders game
ENV_NAME = "ALE/SpaceInvaders-v5"
IMG_WIDTH = 84
IMG_HEIGHT = 84
STACK_FRAMES = 4

# Frame skipping to speed up training and reduce computation
FRAME_SKIP = 4
REPEAT_ACTION_PROBABILITY = 0.25

# Hardware configuration - using GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Learning hyperparameters for the DQN algorithm
LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMMA = 0.99

# Target network update frequency for stable learning
TARGET_UPDATE = 10000

# Experience replay memory settings
MEMORY_SIZE = 100000
REPLAY_START_SIZE = 10000

# Total training duration in frames
TOTAL_FRAMES = 500000

# Exploration schedule - epsilon starts high and decays over time
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 400000