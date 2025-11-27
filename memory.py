"""
Experience replay memory for DQN training.
Stores and samples past experiences to break correlation in training data.
"""
import numpy as np
import random

# Global variables for the experience replay buffer
memory_buffer = []
memory_position = 0
memory_capacity = 0

# Initializing the experience replay memory with given capacity.
def init_memory(capacity):
    global memory_buffer, memory_position, memory_capacity
    # Creating a fixed-size buffer for storing experiences
    memory_buffer = [None] * capacity
    memory_position = 0
    memory_capacity = capacity

# Adding a new experience to the replay memory.
def push_memory(state, action, reward, next_state, done):
    global memory_buffer, memory_position
    # Storing the experience tuple in the buffer
    memory_buffer[memory_position] = (state, action, reward, next_state, done)
    # Moving to the next position with circular buffer behavior
    memory_position = (memory_position + 1) % memory_capacity

# Randomly sampling a batch of experiences from memory.
def sample_memory(batch_size):
    # Collecting all valid (non-None) experiences from buffer
    valid_samples = [x for x in memory_buffer if x is not None]
    # Returning None if we don't have enough samples for a batch
    if len(valid_samples) < batch_size:
        return None
    # Randomly selecting a batch of experiences for training
    return random.sample(valid_samples, batch_size)

# Getting the current number of experiences stored in memory.
def memory_size():
    # Counting how many valid experiences are currently in the buffer
    return len([x for x in memory_buffer if x is not None])