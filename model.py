"""
Deep Q-Network model definition for Space Invaders.
Contains the neural network architecture that learns to play the game.
"""
import torch
import torch.nn as nn
import numpy as np
import config

# Creating and initializing the Deep Q-Network model.
def create_dqn(action_size):
    class DQN(nn.Module):
        def __init__(self, action_size):
            super().__init__()

            # Defining convolutional layers for processing game frames
            self.conv1 = nn.Conv2d(config.STACK_FRAMES, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            
            # Calculate conv output size
            with torch.no_grad():
                # Creating a dummy input to determine output size after convolutions
                test_input = torch.zeros(1, config.STACK_FRAMES, config.IMG_HEIGHT, config.IMG_WIDTH)
                conv_out = self.conv3(self.conv2(self.conv1(test_input)))
                self.conv_out_size = int(np.prod(conv_out.size()))
            
            # Defining fully connected layers for Q-value prediction
            self.fc1 = nn.Linear(self.conv_out_size, 512)
            self.fc2 = nn.Linear(512, action_size)

            # Initializing weights for stable training
            self._initialize_weights()
        
        # Initialize network weights.
        def _initialize_weights(self):
            # Applying Kaiming initialization to all convolutional and linear layers
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.0)
            
            # Using smaller initialization for final layer to prevent large Q-values
            nn.init.uniform_(self.fc2.weight, -0.003, 0.003)
            nn.init.constant_(self.fc2.bias, 0.0)
        
        # Forward pass through the network.
        def forward(self, x):
            # Normalizing pixel values from 0-255 to 0-1
            x = x.float() / 255.0
            # Passing through convolutional layers with ReLU activation
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            # Flattening convolutional output for fully connected layers
            x = x.view(x.size(0), -1)
            # Passing through fully connected layers
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Creating the model and moving it to the appropriate device (GPU/CPU)
    return DQN(action_size).to(config.DEVICE)

# Helper function for forward pass with proper input handling
def dqn_forward(model, x):
    return model(x)