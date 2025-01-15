import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import wandb
from .API import WANDB_API_KEY

class TrajectoryNet_Train:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, device='cpu', save_dir='./checkpoints', use_wandb=False):
        """
        Initialize the Trainer class.
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

        # Handle wandb initialization
        self.wandb_enabled = use_wandb or self.config.get("wandb", False)
        if self.wandb_enabled and not wandb.run:
            wandb.login(key=WANDB_API_KEY)
            wandb.init(project="Fundamental_Actions", config=self.config.config['TrajectoryNet'], name=self.name)

    def adjust_learning_rate(self, epoch):
        """
        Adjust the learning rate during warmup epochs.
        """
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            base_lr = self.config.get("TrajectoryNet.training.base_lr", 0.001)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor

    def save_checkpoint(self, epoch, val_loss):
        """
        Save model checkpoint and handle wandb logging.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save local checkpoint
        model_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(checkpoint, model_path)
        
        if self.wandb_enabled:
            # Log model state as a wandb artifact
            artifact = wandb.Artifact(
                name=f'model-{wandb.run.id}',
                type='model',
                description=f'Model checkpoint from epoch {epoch}'
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    def train(self):
        """
        Train the model and evaluate on validation data.
        """
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            if not self.wandb_enabled:
                print(f"Epoch {epoch}/{self.num_epochs}")

            self.adjust_learning_rate(epoch)

            # Training phase
            self.model.train()
            train_loss = 0
            train_iterator = tqdm(self.train_loader, desc="Training") if not self.wandb_enabled else self.train_loader
            
            for inputs, targets in train_iterator:
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
            if not self.wandb_enabled:
                print(f"Train Loss: {train_loss:.4f}")

            # Validation phase
            val_loss = self.validate()
            if not self.wandb_enabled:
                print(f"Validation Loss: {val_loss:.4f}")

            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            
            if self.wandb_enabled:
                wandb.log(metrics)

            # Model checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                if not self.wandb_enabled:
                    print("Saved best model.")
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                if not self.wandb_enabled:
                    print("Early stopping triggered.")
                break

        if not self.wandb_enabled:
            print("Training completed.")
        return self.best_loss

    def validate(self):
        """
        Validate the model on validation data.
        """
        self.model.eval()
        val_loss = 0
        val_iterator = tqdm(self.val_loader, desc="Validation") if not self.wandb_enabled else self.val_loader
        
        with torch.no_grad():
            for inputs, targets in val_iterator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)