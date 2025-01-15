import pandas as pd

def reshape_data(input_file, output_file):
    try:
        # Read the original CSV file with robust settings
        df = pd.read_csv(input_file, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        
        # Validate column names
        required_columns = ['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z'] + \
                           [f'p{i}_{dim}' for i in range(1, 81) for dim in ['x', 'y', 'z', 'c']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in input file: {missing_columns}")
        
        # Create an empty list to store the reshaped data
        reshaped_data = []

        # Loop over each row in the original DataFrame
        for _, row in df.iterrows():
            # Extract inputs
            input1_x, input1_y, input1_z = row['input1_x'], row['input1_y'], row['input1_z']
            input2_x, input2_y, input2_z = row['input2_x'], row['input2_y'], row['input2_z']
            
            # Loop through p1_x to p80_x and corresponding p_y, p_z, p_c columns
            for i in range(1, 81):
                p_x = row[f'p{i}_x']
                p_y = row[f'p{i}_y']
                p_z = row[f'p{i}_z']
                p_c = row[f'p{i}_c']
                
                # Append the reshaped row to the list
                reshaped_data.append([input1_x, input1_y, input1_z, input2_x, input2_y, input2_z, p_x, p_y, p_z, p_c])
        
        # Create a DataFrame from the reshaped data
        reshaped_df = pd.DataFrame(reshaped_data, columns=['input1_x', 'input1_y', 'input1_z', 'input2_x', 'input2_y', 'input2_z', 'p_x', 'p_y', 'p_z', 'p_c'])
        
        # Write the reshaped DataFrame to a new CSV file
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

# Example usage
input_file = 'dataset.csv'  # Ensure this file is a valid CSV file
output_file = 'lstm_dataset.csv'  # Specify the output file name
reshape_data(input_file, output_file)
