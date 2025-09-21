

import pandas as pd
import os

# Path to raw data
data_path = "../data/raw/"

# List CSV files
files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
print("Found files:", files)

# Empty list to store dataframes
dfs = []

for file in files:
    df = pd.read_csv(os.path.join(data_path, file))
    
    # Drop unnamed extra columns (if any)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Append to list
    dfs.append(df)

# Merge all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Sort by city and date
merged_df = merged_df.sort_values(['City', 'Date'])

# Fill missing values (forward fill)
merged_df.fillna(method='ffill', inplace=True)

# Optional: reset index
merged_df.reset_index(drop=True, inplace=True)

# Check final merged data
print(merged_df.head())
print(merged_df.info())
print(merged_df.isnull().sum())

# Save cleaned CSV
os.makedirs("../data/processed", exist_ok=True)
merged_df.to_csv("../data/processed/cleaned_AQI_dataset.csv", index=False)
print("Merged and cleaned dataset saved to data/processed/cleaned_AQI_dataset.csv")
