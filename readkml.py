import pandas as pd

# Specify the file path of the CSV file
csv_file_path = 'kmlTable.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
print(df.head())
# Print the 'point' column
print(df['name'])