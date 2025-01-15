"""
Author: Shreyas Dixit

This file converters normal TrajectoryNet dataset into LSTM dataset
"""
import pandas as pd

def reshape_data(input_file, output_file):
    """
    Reshape the input data from the original CSV file to a new CSV file suitable for LSTM training.
    
    Args:
        input_file (str): Path to the original CSV file.
        output_file (str): Path to the new CSV file to save the reshaped data.
    """
    try:
        df = pd.read_csv(input_file, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        required_columns = ['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z'] + \
                           [f'p{i}_{dim}' for i in range(1, 81) for dim in ['x', 'y', 'z', 'c']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in input file: {missing_columns}")

        reshaped_data = []
        for _, row in df.iterrows():
            input1_x, input1_y, input1_z = row['input1_x'], row['input1_y'], row['input1_z']
            input2_x, input2_y, input2_z = row['input2_x'], row['input2_y'], row['input2_z']
            
            for i in range(1, 81):
                p_x = row[f'p{i}_x']
                p_y = row[f'p{i}_y']
                p_z = row[f'p{i}_z']
                p_c = row[f'p{i}_c']
                
                reshaped_data.append([input1_x, input1_y, input1_z, input2_x, input2_y, input2_z, p_x, p_y, p_z, p_c])
    
        reshaped_df = pd.DataFrame(reshaped_data, columns=['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z', 'p_x', 'p_y', 'p_z', 'p_c'])
        
        reshaped_df.to_csv(output_file, index=False)
        print(f"Data reshaped and saved successfully to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

input_file = 'dataset.csv'
output_file = 'lstm_dataset.csv'
reshape_data(input_file, output_file)
