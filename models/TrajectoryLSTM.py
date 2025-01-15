"""
Author : Shreyas Dixit
This file contains the TrajectoryLSTM model to predict the trajectory of an object.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryLSTM(nn.Module):
    """
    A LSTM model that takes in configurable input size, hidden sizes, and outputs an 80 Point 4D Trajectory.
    
    Args:
        config (dict): Configuration dictionary with model parameters like input size, hidden sizes, and output size.
    """
    def __init__(self, config):
        super(TrajectoryLSTM, self).__init__()

        self.input_size = config.get("input_size", 10)
        self.hidden_size = config.get("hidden_size", 512)
        self.num_layers = config.get("num_layers", 3)
        self.output_size = config.get("output_size", 4)
        self.dropout = config.get("dropout", 0.5)
        
        # Create a single LSTM with multiple layers instead of ModuleList
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden=None):
        """
        Forward pass of the model
        """
        lstm_out, (hidden_state, cell_state) = self.lstm(x, hidden)
        
        last_hidden_state = hidden_state[-1]
        normalized = self.batch_norm(last_hidden_state)
        output = self.fc(normalized)
        
        return output, (hidden_state, cell_state)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config_loader import LoadConfig
    
    config_path = '../config/config.yaml'
    config = LoadConfig(config_path)

    model = TrajectoryLSTM(config)
    
    batch_size = config.get("TrajectoryLSTM.training.batch_size", 8)
    seq_len = config.get("TrajectoryLSTM.training.seq_len", 80)

    example_input = torch.randn(batch_size, seq_len, 10)
    
    output, hidden = model(example_input)
    
    print(f"Input shape: {example_input.shape}")
    print(f"Output shape: {output.shape}")
