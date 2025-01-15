import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from models.TrajectoryNet import TrajectoryNet
from trainers.TrajectoryNet_Trainer import TrajectoryNet_Train
from dataset.dataset import preprocess_data
from config.config_loader import LoadConfig

def train_with_sweep():
    """
    Main function to train the TrajectoryNet model with wandb sweep.
    """
    # Initialize wandb
    with wandb.init() as run:
        # Load base configuration
        config = LoadConfig("config/config.yaml")
        
        # Update config with wandb sweep parameters
        sweep_config = {
            'TrajectoryNet.model.hidden_size': wandb.config.hidden_size,
            'TrajectoryNet.training.learning_rate': wandb.config.learning_rate,
            'TrajectoryNet.training.weight_decay': wandb.config.weight_decay,
            'TrajectoryNet.model.dropout': wandb.config.dropout
        }
        config.update(sweep_config)

        # Preprocess data
        file_path = config.get("TrajectoryNet.data.file_path")
        train_dataset, val_dataset, input_scaler, target_scaler = preprocess_data(file_path)

        # Create DataLoaders
        batch_size = config.get("TrajectoryNet.training.batch_size")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TrajectoryNet(config).to(device)

        # Define optimizer and loss
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("TrajectoryNet.training.learning_rate"),
            weight_decay=config.get("TrajectoryNet.training.weight_decay")
        )
        criterion = nn.MSELoss()

        # Initialize trainer
        trainer = TrajectoryNet_Train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            use_wandb=True
        )

        # Train the model and get the best validation loss
        best_val_loss = trainer.train()

def main():
    """
    Setup and run the wandb sweep
    """
    # Define sweep configuration
    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'hidden_size': {
                'values': [8, 64, 128, 256]
            },
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': -9.21,  # log(0.0001)
                'max': -4.61    # log(0.01)
            },
            'weight_decay': {
                'distribution': 'log_uniform',
                'min': -9.21,  # log(0.0001)
                'max': -4.61    # log(0.01)
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            }
        }
    }

    # Initialize wandb
    wandb.login()
    
    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="TrajectoryNet_Optimization"
    )

    # Start the sweep
    wandb.agent(sweep_id, function=train_with_sweep, count=20)  # Run 20 experiments

if __name__ == "__main__":
    main()