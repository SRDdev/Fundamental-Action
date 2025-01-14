import pandas as pd

def reshape_data(input_file, output_file):
    # Read the original CSV file
    df = pd.read_csv(input_file)
    
    # Create an empty list to store the reshaped data
    reshaped_data = []
    
    # Loop over each row in the original DataFrame
    for _, row in df.iterrows():
        # Extract input1_x, input1_y, input1_z, input2_x, input2_y, input2_z
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

# Example usage
input_file = 'sample_data.csv'  # Input CSV file name
output_file = 'reshaped_dataset.csv'  # Output CSV file name
reshape_data(input_file, output_file)
