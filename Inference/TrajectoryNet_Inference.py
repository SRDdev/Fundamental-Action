import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import yaml

# Ensure proper path loading for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.TrajectoryNet import TrajectoryNet

USE_SCALERS = True 

def load_model(model_path, config, device='cpu'):
    """
    Load the pre-trained model from the specified path.

    Args:
        model_path (str): Path to the saved model checkpoint.
        config (dict): Configuration dictionary containing model parameters.
        device (str): The device to load the model on ('cpu' or 'cuda').

    Returns:
        model (nn.Module): The loaded model with pre-trained weights.
    """
    try:
        model = TrajectoryNet(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_scalers(save_dir):
    """
    Load the input and output scalers from the saved directory.

    Args:
        save_dir (str): Directory where the scalers are saved.

    Returns:
        input_scaler (sklearn.preprocessing.StandardScaler): The input scaler.
        target_scaler (sklearn.preprocessing.StandardScaler): The target scaler.
    """
    input_scaler = torch.load(f".{save_dir}/input_scaler.pth")
    target_scaler = torch.load(f".{save_dir}/target_scaler.pth")
    return input_scaler, target_scaler


def predict(model, input_data, input_scaler, target_scaler, device='cpu', use_scalers=True):
    """
    Run inference on the model with the given input data.

    Args:
        model (nn.Module): The trained model.
        input_data (numpy.ndarray or torch.Tensor): The input data to make predictions.
        input_scaler (sklearn.preprocessing.StandardScaler): The input scaler to normalize input data.
        target_scaler (sklearn.preprocessing.StandardScaler): The target scaler to reverse the output scaling.
        device (str): The device to run the model on ('cpu' or 'cuda').
        use_scalers (bool): Whether to use scalers or not.

    Returns:
        torch.Tensor: The predicted trajectory.
    """
    model.to(device)

    # If using scalers, scale the input data
    if use_scalers:
        input_data = input_scaler.transform(input_data)

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Reverse the scaling on the output if scalers are used
    if use_scalers:
        predictions = target_scaler.inverse_transform(output.cpu().numpy())
    else:
        predictions = output.cpu().numpy()

    return predictions


def save_predictions(predictions, output_file='predictions.json'):
    """
    Save the predictions to a file.

    Args:
        predictions (numpy.ndarray): The predicted trajectory.
        output_file (str): The output file path to save the predictions.

    Returns:
        None
    """
    # Save the predictions as a JSON file
    with open(output_file, 'w') as f:
        json.dump(predictions.tolist(), f)

def main():
    """
    Main function to load the model, scalers, and make predictions.
    """
    from config.config_loader import LoadConfig
    config_path = "../config/config.yaml"
    config = LoadConfig(config_path)
    device = config.get('device', 'cpu')

    model_dir = config.get('TrajectoryNet.training.save_dir')
    model_path = f'.{model_dir}/best_model.pth'
    
    # Load model
    model = load_model(model_path, config, device)

    # Load input and output scalers if using them
    if USE_SCALERS:
        input_scaler, target_scaler = load_scalers(model_dir)
    else:
        input_scaler = target_scaler = None

    # Example input: Locations of 2 objects (6 features)
    example_input = np.random.rand(1, 6)  # Replace with actual input data

    # Make prediction
    predictions = predict(model, example_input, input_scaler, target_scaler, device, use_scalers=USE_SCALERS)

    # Print the predicted trajectory
    print(f"Predicted Trajectory: {predictions}")

    # Save the predictions
    save_predictions(predictions)


if __name__ == '__main__':
    main()
