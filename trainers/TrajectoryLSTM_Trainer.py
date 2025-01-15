"""
Author: Shreyas Dixit

This file contains the TrajectoryLSTM_Train class which is used to train the TrajectoryLSTM model.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from API import WANDB_API_KEY
import wandb
from tqdm import tqdm

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
        
        self.batch_size = config.get("TrajectoryLSTM.training.batch_size", 80)
        self.epochs = config.get("TrajectoryLSTM.training.epochs", 10)
        self.lr = config.get("TrajectoryLSTM.training.learning_rate", 0.001)
        self.device = config.get("device", "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.use_wandb = config.get("wandb", False)
        wandb.login(key=WANDB_API_KEY)
        if self.use_wandb:
            wandb.init(project="Fundamental_Actions", config=config)
            wandb.watch(self.model, log="all")
        
    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            num_batches = len(self.train_data_loader)
            num_rows_processed = 0

            progress_bar = tqdm(self.train_data_loader, total=num_batches, desc=f"Epoch {epoch+1}/{self.epochs}", ncols=100, unit='batch')

            for batch_data, batch_labels in progress_bar:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                output, _ = self.model(batch_data)
                
                loss = self.loss_fn(output, batch_labels.squeeze(1))
                
                running_loss += loss.item()
                
                num_rows_processed += batch_data.size(0)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            avg_train_loss = running_loss / num_batches
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({"train_loss": avg_train_loss, "epoch": epoch+1})
            
            self.evaluate(epoch, avg_train_loss)

    
    def evaluate(self, epoch, train_loss):
        """
        Evaluate the model on the test set and print the results.
        
        :param epoch: The current epoch number.
        :param train_loss: The training loss for the current epoch.
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.test_data_loader)
        
        with torch.no_grad():
            for batch_data, batch_labels in self.test_data_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                output, _ = self.model(batch_data)
                
                loss = self.loss_fn(output, batch_labels.squeeze(1))
                val_loss += loss.item()

        avg_val_loss = val_loss / num_batches
        print(f"Epoch [{epoch+1}/{self.epochs}], Validation Loss: {avg_val_loss:.4f}")

        if self.use_wandb:
            wandb.log({"val_loss": avg_val_loss, "train_loss": train_loss, "epoch": epoch+1})

    def save_model(self, save_path):
        """
        Save the trained model to a file.
        
        :param save_path: The path where the model will be saved.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

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
    
    from config.config_loader import LoadConfig
    from models.TrajectoryLSTM import TrajectoryLSTM
    from dataset.lstm_dataset import create_dataloader
    config = LoadConfig(config_path)

    csv_file = config.get('TrajectoryLSTM.data.file_path', '../data/trajectory_data.csv')

    seq_len = config.get("TrajectoryLSTM.training.seq_len", 80)
    input_size = config.get("TrajectoryLSTM.training.input_size", 10) 
    target_size = config.get("TrajectoryLSTM.training.target_size", 4)
    batch_size = config.get("TrajectoryLSTM.training.batch_size", 8)

    train_loader, test_loader = create_dataloader(csv_file, seq_len, batch_size, input_size, target_size)

    model = TrajectoryLSTM(config)

    trainer = TrajectoryLSTM_Train(model, train_loader, test_loader, config)

    trainer.train()

    model_dir = "checkpoints"
    model_save_path = f"{model_dir}/trajectory_lstm_model.pth"
    trainer.save_model(model_save_path)

    print(f"Model saved to {model_save_path}")

