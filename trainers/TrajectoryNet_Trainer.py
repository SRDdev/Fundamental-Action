import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import wandb
from .API import WANDB_API_KEY

class TrajectoryNet_Train:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, device='cpu', save_dir='./checkpoints'):
        """
        Initialize the Trainer class.

        Args:
            model (nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            config (dict): Configuration dictionary that includes hyperparameters and wandb settings.
            device (str): Device to train on ('cpu' or 'cuda').
            save_dir (str): Directory to save model checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_loss = float('inf')
        self.config = config
        self.name = config.get("TrajectoryNet.training.name")

        self.num_epochs = config.get("TrajectoryNet.training.num_epochs")
        self.early_stopping_patience = config.get("TrajectoryNet.training.early_stopping_patience")
        self.grad_clip_value = config.get("TrajectoryNet.training.grad_clip_value", None)
        self.warmup_epochs = config.get("TrajectoryNet.training.warmup_epochs", 0)

        if self.config.get("wandb", False):
            wandb.login(key=WANDB_API_KEY)

            run = wandb.init(project="Fundamental_Actions", config=self.config, name=self.name)
            self.wandb_enabled = True
        else:
            self.wandb_enabled = False

    def adjust_learning_rate(self, epoch):
        """
        Adjust the learning rate during warmup epochs.
        """
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.get("TrajectoryNet.training.base_lr", 0.001) * warmup_factor

    def train(self):
        """
        Train the model and evaluate on validation data.

        Args:
            num_epochs (int): Number of epochs to train.
            early_stopping_patience (int): Number of epochs to wait before early stopping.
        """
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")

            self.adjust_learning_rate(epoch)

            self.model.train()
            train_loss = 0
            for inputs, targets in tqdm(self.train_loader, desc="Training"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            print(f"Train Loss: {train_loss:.4f}")

            if self.wandb_enabled:
                wandb.log({"train_loss": train_loss, "epoch": epoch})

            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")

            if self.wandb_enabled:
                wandb.log({"val_loss": val_loss, "epoch": epoch})

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                print("Saved best model.")
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        print("Training completed.")

    def validate(self):
        """
        Validate the model on validation data.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()

        return val_loss / len(self.val_loader)
