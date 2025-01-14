# """
# Author : Shreyas Dixit
# This file converts the normal dataset into a dataset for LSTM model in PyTorch.
# """
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# class LSTMDataLoader:
#     def __init__(self, csv_file, timesteps=80, features=10, target_col='p_c', batch_size=32, test_size=0.2):
#         """
#         Initialize the data loader.

#         :param csv_file: Path to the CSV file containing the dataset.
#         :param timesteps: The number of timesteps in each sequence (default 80).
#         :param features: The number of features (default 10).
#         :param target_col: The column name for the target variable (default 'p_c').
#         :param batch_size: The batch size for training (default 32).
#         :param test_size: Proportion of the data to use for testing (default 0.2).
#         """
#         self.csv_file = csv_file
#         self.timesteps = timesteps
#         self.features = features
#         self.target_col = target_col
#         self.batch_size = batch_size
#         self.test_size = test_size
        
#         # Load and preprocess the dataset
#         self.load_data()

#     def load_data(self):
#         """
#         Load the dataset, normalize the features, and split into train/test.
#         """
#         # Load the dataset from the CSV file
#         df = pd.read_csv(self.csv_file)
        
#         # Select features and target columns
#         features = ['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z', 'p_x', 'p_y', 'p_z', 'p_c']
#         target = self.target_col
        
#         # Extract feature values (X) and target values (y)
#         X = df[features].values
#         y = df[target].values
        
#         # Normalize the features using MinMaxScaler
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         X_scaled = scaler.fit_transform(X)

#         # Calculate the number of samples that will be created after reshaping
#         num_sequences = len(X_scaled) // self.timesteps
        
#         # Trim the data to be evenly divisible by timesteps
#         trim_length = num_sequences * self.timesteps
#         X_scaled = X_scaled[:trim_length]
#         y = y[:trim_length]

#         # Reshape X into sequences: [num_sequences, timesteps, features]
#         X_sequences = X_scaled.reshape(num_sequences, self.timesteps, self.features)
        
#         # Modify y to have 4 output values per sequence (using the last 4 values)
#         y_sequences = np.column_stack([y[i:i+self.timesteps][-1] for i in range(0, len(y), self.timesteps)])

#         # Now split the sequences into train and test sets
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X_sequences, y_sequences, 
#             test_size=self.test_size, 
#             shuffle=False
#         )

#         # Convert to PyTorch tensors
#         self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
#         self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
#         self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
#         self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

#         print(f"X_train shape: {self.X_train.shape}")
#         print(f"X_test shape: {self.X_test.shape}")
#         print(f"y_train shape: {self.y_train.shape}")
#         print(f"y_test shape: {self.y_test.shape}")

#     # def load_data(self):
#     #     """
#     #     Load the dataset, normalize the features, and split into train/test.
#     #     """
#     #     # Load the dataset from the CSV file
#     #     df = pd.read_csv(self.csv_file)
        
#     #     # Select features and target columns
#     #     features = ['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z', 'p_x', 'p_y', 'p_z', 'p_c']
#     #     target = self.target_col
        
#     #     # Extract feature values (X) and target values (y)
#     #     X = df[features].values
#     #     y = df[target].values
        
#     #     # Normalize the features using MinMaxScaler
#     #     scaler = MinMaxScaler(feature_range=(0, 1))
#     #     X_scaled = scaler.fit_transform(X)

#     #     # Calculate the number of samples that will be created after reshaping
#     #     num_sequences = len(X_scaled) // self.timesteps
        
#     #     # Trim the data to be evenly divisible by timesteps
#     #     trim_length = num_sequences * self.timesteps
#     #     X_scaled = X_scaled[:trim_length]
#     #     y = y[:trim_length]

#     #     # Reshape X into sequences: [num_sequences, timesteps, features]
#     #     X_sequences = X_scaled.reshape(num_sequences, self.timesteps, self.features)
        
#     #     # For y, we need one target value per sequence
#     #     # We'll take the last value of each sequence as the target
#     #     y_sequences = y.reshape(num_sequences, self.timesteps)[:, -1]

#     #     # Now split the sequences into train and test sets
#     #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#     #         X_sequences, y_sequences, 
#     #         test_size=self.test_size, 
#     #         shuffle=False
#     #     )

#     #     # Convert to PyTorch tensors
#     #     self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
#     #     self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
#     #     self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
#     #     self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

#     #     print(f"X_train shape: {self.X_train.shape}")
#     #     print(f"X_test shape: {self.X_test.shape}")
#     #     print(f"y_train shape: {self.y_train.shape}")
#     #     print(f"y_test shape: {self.y_test.shape}")

#     def get_train_data(self):
#         """
#         Returns the training data.
#         """
#         return self.X_train, self.y_train

#     def get_test_data(self):
#         """
#         Returns the testing data.
#         """
#         return self.X_test, self.y_test

#     def get_train_loader(self):
#         """
#         Returns a PyTorch DataLoader for training data.
#         """
#         train_dataset = TensorDataset(self.X_train, self.y_train)
#         return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

#     def get_test_loader(self):
#         """
#         Returns a PyTorch DataLoader for test data.
#         """
#         test_dataset = TensorDataset(self.X_test, self.y_test)
#         return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

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

        # Check the dataset's shape
        print(f"Columns in dataset: {self.data.columns}")
        print(f"Shape of dataset: {self.data.shape}")

        # Create sequences from the data
        self.samples = self.create_sequences()

    def create_sequences(self):
        """
        Create sequences of inputs and targets from the dataset.
        """
        sequences = []
        
        # For each row, generate a sequence with a fixed seq_len
        for i in range(len(self.data) - self.seq_len):
            # Input: Take 'seq_len' rows as input (all columns, including 'p_c')
            input_seq = self.data.iloc[i:i + self.seq_len, :].values  # Take all columns
            
            # Ensure the input sequence has the correct number of features
            if input_seq.shape[1] != self.input_size:
                print(f"Warning: Expected {self.input_size} features, but got {input_seq.shape[1]} features.")
                print(f"Input sequence (first row): {input_seq[0]}")

            # Target: Take the last row's target value ('p_c')
            target = self.data.iloc[i + self.seq_len - 1, -1]  # The last column is 'p_c'
            
            sequences.append((input_seq, target))
        
        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        # Convert input and target to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return input_tensor, target_tensor


def create_dataloader(csv_file, seq_len, batch_size, input_size=10, target_size=4, shuffle=True):
    """
    Create a DataLoader for training and testing the model.
    """
    dataset = TrajectoryDataset(csv_file, seq_len, input_size, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
