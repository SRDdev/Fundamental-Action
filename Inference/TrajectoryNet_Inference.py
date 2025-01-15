# """
# Author: Shreyas Dixit

# This script loads a pre-trained TrajectoryNet model and makes predictions on a sample input.
# """

# import sys
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import json
# import yaml
# import csv

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models.TrajectoryNet import TrajectoryNet

# USE_SCALERS = True 

# def load_model(model_path, config, device='cpu'):
#     """
#     Load the pre-trained model from the specified path.

#     Args:
#         model_path (str): Path to the saved model checkpoint.
#         config (dict): Configuration dictionary containing model parameters.
#         device (str): The device to load the model on ('cpu' or 'cuda').

#     Returns:
#         model (nn.Module): The loaded model with pre-trained weights.
#     """
#     try:
#         model = TrajectoryNet(config)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise


# def load_scalers(save_dir):
#     """
#     Load the input and output scalers from the saved directory.

#     Args:
#         save_dir (str): Directory where the scalers are saved.

#     Returns:
#         input_scaler (sklearn.preprocessing.StandardScaler): The input scaler.
#         target_scaler (sklearn.preprocessing.StandardScaler): The target scaler.
#     """
#     input_scaler = torch.load(f".{save_dir}/input_scaler.pth")
#     target_scaler = torch.load(f".{save_dir}/target_scaler.pth")
#     return input_scaler, target_scaler


# def predict(model, input_data, input_scaler, target_scaler, device='cpu', use_scalers=True):
#     """
#     Run inference on the model with the given input data.

#     Args:
#         model (nn.Module): The trained model.
#         input_data (numpy.ndarray or torch.Tensor): The input data to make predictions.
#         input_scaler (sklearn.preprocessing.StandardScaler): The input scaler to normalize input data.
#         target_scaler (sklearn.preprocessing.StandardScaler): The target scaler to reverse the output scaling.
#         device (str): The device to run the model on ('cpu' or 'cuda').
#         use_scalers (bool): Whether to use scalers or not.

#     Returns:
#         torch.Tensor: The predicted trajectory.
#     """
#     model.to(device)

#     if use_scalers:
#         input_data = input_scaler.transform(input_data)

#     input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

#     with torch.no_grad():
#         output = model(input_tensor)

#     if use_scalers:
#         predictions = target_scaler.inverse_transform(output.cpu().numpy())
#     else:
#         predictions = output.cpu().numpy()

#     return predictions


# def save_predictions(predictions, output_file='predictions.csv'):
#     """
#     Save the predictions to a CSV file.

#     Args:
#         predictions (numpy.ndarray): The predicted trajectory.
#         output_file (str): The output file path to save the predictions.

#     Returns:
#         None
#     """
#     reshaped_predictions = predictions.reshape(80, 4)
#     with open(output_file, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['x', 'y', 'z', 'c'])
#         writer.writerows(reshaped_predictions)

# def main():
#     """
#     Main function to load the model, scalers, and make predictions.
#     """
#     from config.config_loader import LoadConfig
    
#     config_path = "../config/config.yaml"
#     config = LoadConfig(config_path)
#     device = config.get('device', 'cpu')

#     model_dir = config.get('TrajectoryNet.training.save_dir')
#     model_path = f'.{model_dir}/best_model.pth'
    
#     model = load_model(model_path, config, device)

#     if USE_SCALERS:
#         input_scaler, target_scaler = load_scalers(model_dir)
#     else:
#         input_scaler = target_scaler = None


#     input_data = np.array([[-202.9524806,-828.6285971,41.35059464,-370.808101,-735.661648,104.0947533]])
#     print(input_data)

#     predictions = predict(model, input_data, input_scaler, target_scaler, device, use_scalers=USE_SCALERS)


#     print("-"*100)
#     print(predictions.shape)
#     print("-"*100)
#     reshaped_predictions = predictions.reshape(80, 4)

#     for i, pred in enumerate(reshaped_predictions):
#         print(f"Prediction {i+1}: {pred}")

#     save_predictions(predictions)
    
# if __name__ == '__main__':
#     main()


"""
Author: Shreyas Dixit
Modified version: Takes 6 inputs and returns CSV predictions
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import yaml
import csv
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.TrajectoryNet import TrajectoryNet

USE_SCALERS = True 

def load_model(model_path, config, device='cpu'):
    """
    Load the pre-trained model from the specified path.
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
    """
    input_scaler = torch.load(f".{save_dir}/input_scaler.pth")
    target_scaler = torch.load(f".{save_dir}/target_scaler.pth")
    return input_scaler, target_scaler

def predict(model, input_data, input_scaler, target_scaler, device='cpu', use_scalers=True):
    """
    Run inference on the model with the given input data.
    """
    model.to(device)

    if use_scalers:
        input_data = input_scaler.transform(input_data)

    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    if use_scalers:
        predictions = target_scaler.inverse_transform(output.cpu().numpy())
    else:
        predictions = output.cpu().numpy()

    return predictions

def get_predictions_csv(x1, y1, z1, x2, y2, z2):
    """
    Main function that takes 6 inputs and returns predictions as a CSV string.
    
    Args:
        x1, y1, z1: First point coordinates
        x2, y2, z2: Second point coordinates
        
    Returns:
        str: CSV string containing the predictions
    """
    from config.config_loader import LoadConfig
    
    config_path = "../config/config.yaml"
    config = LoadConfig(config_path)
    device = config.get('device', 'cpu')

    model_dir = config.get('TrajectoryNet.training.save_dir')
    model_path = f'.{model_dir}/trajectorynet_7.pth'
    
    model = load_model(model_path, config, device)

    if USE_SCALERS:
        input_scaler, target_scaler = load_scalers(model_dir)
    else:
        input_scaler = target_scaler = None

    # Format input data using the 6 parameters
    input_data = np.array([[x1, y1, z1, x2, y2, z2]])
    
    predictions = predict(model, input_data, input_scaler, target_scaler, device, use_scalers=USE_SCALERS)
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['x', 'y', 'z', 'c'])
    
    reshaped_predictions = predictions.reshape(80, 4)
    writer.writerows(reshaped_predictions)
    
    return output.getvalue()

if __name__ == '__main__':
    # Example usage
    x1, y1, z1 = -202.9524806, -828.6285971, 41.35059464
    x2, y2, z2 = -370.808101, -735.661648, 104.0947533
    
    csv_output = get_predictions_csv(x1, y1, z1, x2, y2, z2)
    print(csv_output)