"""
Author: Shreyas Dixit
A PyTorch Dataset class for the Trajectory Prediction Problem.  
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset for trajectory prediction.
    """
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def preprocess_data(file_path):
    """
    Preprocess the data and return PyTorch Datasets.
    """
    data = pd.read_csv(file_path)
    
    input_columns = ['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z']
    target_columns = [f'p{i}_{dim}' for i in range(1, 81) for dim in ['x', 'y', 'z', 'c']]
    
    inputs = data[input_columns].values
    targets = data[target_columns].values
    
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    inputs_normalized = input_scaler.fit_transform(inputs)
    targets_normalized = target_scaler.fit_transform(targets)
    
    X_train, X_val, y_train, y_val = train_test_split(
        inputs_normalized, targets_normalized, test_size=0.1,shuffle=False
    )
    
    train_dataset = TrajectoryDataset(X_train, y_train)
    val_dataset = TrajectoryDataset(X_val, y_val)
    
    return train_dataset, val_dataset, input_scaler, target_scaler
