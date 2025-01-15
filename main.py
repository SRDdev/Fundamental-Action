"""
Author: Shreyas Dixit

This script trains the TrajectoryNet model on the given dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TrajectoryNet import TrajectoryNet
from trainers.TrajectoryNet_Trainer import TrajectoryNet_Train
from dataset.dataset import preprocess_data
from config.config_loader import LoadConfig


def main():
    """
    Main function to train the TrajectoryNet model.
    """
    #-------------------------------------------------------------------------------------------------------------------#
    # Step 1: Load configuration
    #-------------------------------------------------------------------------------------------------------------------#
    config = LoadConfig("config/config.yaml")
    
    # Print config to verify it
    print(f"Configuration loaded: {config}")

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 2: Preprocess the data
    #-------------------------------------------------------------------------------------------------------------------#
    file_path = config.get("TrajectoryNet.data.file_path")
    print("Preprocessing data...")
    
    # Corrected this part by calling preprocess_data
    train_dataset, val_dataset, input_scaler, target_scaler = preprocess_data(file_path)

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 3: Create DataLoaders
    #-------------------------------------------------------------------------------------------------------------------#
    batch_size = config.get("TrajectoryNet.training.batch_size")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 4: Initialize the model
    #-------------------------------------------------------------------------------------------------------------------#
    print("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryNet(config).to(device)

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 5: Define the loss function and optimizer
    #-------------------------------------------------------------------------------------------------------------------#
    learning_rate = float(config.get("TrajectoryNet.training.learning_rate"))
    weight_decay = float(config.get("TrajectoryNet.training.weight_decay"))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 6: Initialize the Trainer
    #-------------------------------------------------------------------------------------------------------------------#
    print("Starting training...")
    trainer = TrajectoryNet_Train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,  # Pass config to the trainer to handle dynamic wandb logging
        device=device
    )

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 7: Train the model
    #-------------------------------------------------------------------------------------------------------------------#
    num_epochs = config.get("TrajectoryNet.training.num_epochs")
    early_stopping_patience = config.get("TrajectoryNet.training.early_stopping_patience")
    trainer.train()

    #-------------------------------------------------------------------------------------------------------------------#
    # Step 8: Save the scalers for inference
    #-------------------------------------------------------------------------------------------------------------------#
    save_dir = config.get("TrajectoryNet.training.save_dir")
    torch.save(input_scaler, f'{save_dir}/input_scaler.pth')
    torch.save(target_scaler, f'{save_dir}/target_scaler.pth')
    print("Training complete. Model and scalers saved.")




if __name__ == "__main__":
    main()

