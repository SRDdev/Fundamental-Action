"""
Author: Shreyas Dixit

This file contains the implementation of a Deeper Neural Network with Layer Normalization that takes in configurable 
input size, hidden sizes, and outputs an 80 Point 4D Trajectory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryNet(nn.Module):
    """
    A Deeper Neural Network with Layer Normalization that takes in configurable input size, hidden sizes,
    and outputs an 80 Point 4D Trajectory.
    """
    def __init__(self, config):
        """
        Initializes the neural network model with parameters from the config file.

        Args:
            config (dict): Configuration dictionary with model parameters like input size, hidden sizes, and output size.
        """
        super(TrajectoryNet, self).__init__()
        input_size = config.get("TrajectoryNet.model.input_size")
        hidden_size = config.get("TrajectoryNet.model.hidden_size")
        output_size = config.get("TrajectoryNet.model.output_size")
        dropout = config.get("TrajectoryNet.model.dropout")

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc5 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.fc6 = nn.Linear(hidden_size * 4, hidden_size * 8)
        self.fc7 = nn.Linear(hidden_size * 8, hidden_size * 8)
        self.fc8 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.fc9 = nn.Linear(hidden_size * 4, output_size)  # Output layer (80 points * 4 dimensions)

        self.dropout = nn.Dropout(dropout)

        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size * 2)
        self.ln4 = nn.LayerNorm(hidden_size * 4)
        self.ln5 = nn.LayerNorm(hidden_size * 4)
        self.ln6 = nn.LayerNorm(hidden_size * 8)
        self.ln7 = nn.LayerNorm(hidden_size * 8)
        self.ln8 = nn.LayerNorm(hidden_size * 4)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, 80, 4) representing the trajectory.
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(F.relu(self.ln4(self.fc4(x))))
        x = F.relu(self.ln5(self.fc5(x)))
        x = self.dropout(F.relu(self.ln6(self.fc6(x))))
        x = F.relu(self.ln7(self.fc7(x)))
        x = F.relu(self.ln8(self.fc8(x)))
        x = self.fc9(x)
        return x