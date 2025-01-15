"""
Author: Shreyas Dixit
A Neural Network which takes in 6 inputs (Locations of 2 Objects) and outputs a 80 Point 4D Trajectory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryNet(nn.Module):
    """
    A Neural Network which takes in configurable input size, hidden sizes, and outputs an 80 Point 4D Trajectory.
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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc5 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.fc6 = nn.Linear(hidden_size * 4, output_size)  # Output layer (80 points * 4 dimensions)

        self.dropout = nn.Dropout(dropout)
        # Create separate batch norm layers for different dimensions
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.bn4 = nn.BatchNorm1d(hidden_size * 4)
        self.bn5 = nn.BatchNorm1d(hidden_size * 4)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, 80, 4) representing the trajectory.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(F.relu(self.bn4(self.fc4(x))))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        # x = x.view(-1, 80, 4)
        return x
