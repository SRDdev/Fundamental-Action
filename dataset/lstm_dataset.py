"""
Author: Shreyas Dixit

A PyTorch LSTM dataset class for loading and preprocessing data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import wandb

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len, input_size=10, target_size=4):
        """
        Args:
            csv_file (string): Path to the CSV file with data.
            seq_len (int): Length of the input sequence for LSTM.
            input_size (int): Number of input features.
            target_size (int): Number of target features to predict.
        """
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.input_size = input_size
        self.target_size = target_size
        self.use_wandb = False
        # Check the dataset's shape
        print(f"Columns in dataset: {self.data.columns}")
        print(f"Shape of dataset: {self.data.shape}")

        # Create sequences from the data
        self.samples = self.create_sequences()

    def create_sequences(self):
        """
        Create sequences of input and target data from the dataset.
        """
        sequences = []
        for i in range(len(self.data) - self.seq_len):
            input_seq = self.data.iloc[i:i + self.seq_len, :].values  # Use all 10 columns
            target = self.data.iloc[i + self.seq_len - 1, -4:].values  # Last 4 columns
            sequences.append((input_seq, target))
        return sequences
            

    def evaluate(self, epoch):
        """
        Evaluate the model on the test data.
        """
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_data, batch_labels in self.test_data_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                output, _ = self.model(batch_data)
                predictions.append(output.cpu().numpy())
                targets.append(batch_labels.view(-1, 4).cpu().numpy())  # Ensure targets match output shape
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        mse = mean_squared_error(targets, predictions)
        print(f"Test MSE after epoch {epoch+1}: {mse:.4f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        # Convert input and target to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return input_tensor, target_tensor


def create_dataloader(csv_file, seq_len, batch_size, input_size=10, target_size=4, test_split=0.15, shuffle=False):
    """
    Create DataLoaders for training and testing the model.
    
    Args:
        csv_file (str): Path to the CSV file with data.
        seq_len (int): Length of the input sequence for LSTM.
        batch_size (int): Number of samples per batch.
        input_size (int): Number of input features.
        target_size (int): Number of target features to predict.
        test_split (float): Fraction of the data to be used for testing.
        shuffle (bool): Whether to shuffle the data.
    
    Returns:
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the testing data.
    """
    dataset = TrajectoryDataset(csv_file, seq_len, input_size, target_size)
    
    # Split indices for train and test sets
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
