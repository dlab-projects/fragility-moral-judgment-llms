import pandas as pd
from pathlib import Path
from llm_evaluations_everyday_dilemmas.submissions import clean_submissions_df

# Define input and output paths
input_file = 'data/2025.csv'
output_file = 'data/2025_cleaned.csv'

print(f"Reading data from {input_file}")
df = pd.read_csv(input_file)

print("Cleaning submission texts...")
cleaned_df = clean_submissions_df(df, text_column='selftext')

print(f"Saving cleaned data to {output_file}")
cleaned_df.to_csv(output_file, index=False)
print("Done!") 