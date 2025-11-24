import torch
import torch.nn as nn
import numpy as np
import config

def create_dqn(action_size):
    class DQN(nn.Module):
        def __init__(self, action_size):
            super().__init__()
            self.conv1 = nn.Conv2d(config.STACK_FRAMES, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            
            # Calculate conv output size
            with torch.no_grad():
                test_input = torch.zeros(1, config.STACK_FRAMES, config.IMG_HEIGHT, config.IMG_WIDTH)
                conv_out = self.conv3(self.conv2(self.conv1(test_input)))
                self.conv_out_size = int(np.prod(conv_out.size()))
            
            self.fc1 = nn.Linear(self.conv_out_size, 512)
            self.fc2 = nn.Linear(512, action_size)
            
            self._initialize_weights()
        
        def _initialize_weights(self):
            # Proper DQN initialization from DeepMind
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.0)
            
            # Final layer smaller initialization
            nn.init.uniform_(self.fc2.weight, -0.003, 0.003)
            nn.init.constant_(self.fc2.bias, 0.0)
        
        def forward(self, x):
            x = x.float() / 255.0
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    return DQN(action_size).to(config.DEVICE)

def dqn_forward(model, x):
    return model(x)