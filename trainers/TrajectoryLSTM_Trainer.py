"""
Author: Shreyas Dixit
This file contains the TrajectoryLSTM_Trainer class which is used to train the TrajectoryLSTM model.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from API import WANDB_API_KEY
# Add this import for wandb
import wandb

class TrajectoryLSTM_Train:
    def __init__(self, model, train_data_loader, test_data_loader, config):
        """
        Initialize the trainer.

        :param model: The TrajectoryLSTM model.
        :param train_data_loader: A data loader for the training data.
        :param test_data_loader: A data loader for the testing data.
        :param config: Configuration dictionary containing training parameters.
        """
        self.model = model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.config = config
        
        self.batch_size = config.get("TrajectoryLSTM.training.batch_size", 32)
        self.epochs = config.get("TrajectoryLSTM.training.epochs", 10)
        self.lr = config.get("TrajectoryLSTM.training.learning_rate", 0.001)
        self.device = config.get("device", "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Check if we should use wandb for logging
        self.use_wandb = config.get("wandb", False)
        wandb.login(key=WANDB_API_KEY)
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(project="Fundamental_Action", config=config)
            wandb.watch(self.model, log="all")
        
    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            num_batches = len(self.train_data_loader)
            
            for batch_data, batch_labels in self.train_data_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            
                self.optimizer.zero_grad()
                
                output, _ = self.model(batch_data)
                
                # Ensure batch_labels are correctly shaped, if needed (e.g., unsqueeze)
                loss = self.loss_fn(output, batch_labels.unsqueeze(1))
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / num_batches
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            
            # Log the training loss to wandb if enabled
            if self.use_wandb:
                wandb.log({"train_loss": avg_loss, "epoch": epoch+1})
            
            self.evaluate(epoch)
    
    def evaluate(self, epoch):
        """
        Evaluate the model on the test set and print the results.
        
        :param epoch: The current epoch number.
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_data, batch_labels in self.test_data_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                output, _ = self.model(batch_data)
                
                predictions.append(output.cpu().numpy())
                targets.append(batch_labels.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        mse = mean_squared_error(targets, predictions)
        print(f"Test MSE after epoch {epoch+1}: {mse:.4f}")
        
        # Log test MSE to wandb if enabled
        if self.use_wandb:
            wandb.log({"test_mse": mse, "epoch": epoch+1})

    def save_model(self, save_path):
        """
        Save the trained model to a file.
        
        :param save_path: The path where the model will be saved.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Log model checkpoint to wandb if enabled
        if self.use_wandb:
            wandb.save(save_path)

    def load_model(self, load_path):
        """
        Load a pre-trained model from a file.
        
        :param load_path: The path where the model is saved.
        """
        self.model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
        
if __name__ == "__main__":
    import sys
    import os
    config_path = 'config/config.yaml'

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Importing necessary components
    from config.config_loader import LoadConfig
    from models.TrajectoryLSTM import TrajectoryLSTM
    from dataset.lstm_dataset import TrajectoryDataset, create_dataloader  # Updated import

    # Loading the configuration file
    config = LoadConfig(config_path)

    csv_file = config.get('TrajectoryLSTM.data.file_path', '../data/trajectory_data.csv')

    seq_len = config.get("TrajectoryLSTM.training.seq_len", 80)
    input_size = config.get("TrajectoryLSTM.training.input_size", 10) 
    target_size = config.get("TrajectoryLSTM.training.target_size", 4)
    batch_size = config.get("TrajectoryLSTM.training.batch_size", 8)

    train_loader = create_dataloader(csv_file, seq_len, batch_size, input_size, target_size)

    model = TrajectoryLSTM(config)

    trainer = TrajectoryLSTM_Train(model, train_loader, None, config)

    trainer.train()

    model_dir = config.get("TrajectoryLSTM.data.save_dir")
    model_save_path = f"{model_dir}/trajectory_lstm_model.pth"
    trainer.save_model(model_save_path)

    print(f"Model saved to {model_save_path}")

