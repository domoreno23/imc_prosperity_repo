"""
This script Merges all CSV files within a given directory into a single CSV file.
NOTE: ensure that the directory ONLY contains the csv files that you would like to merge

Args:
directory (str): The path to the directory containing the CSV files.
output_file (str): The name of the output CSV file.
"""

import pandas as pd
import glob
import os

def merge_csv_files(directory, output_file):
    all_filenames = glob.glob(os.path.join(directory, "*.csv"))
    all_df = []
    for f in all_filenames:
        df = pd.read_csv(f)
        all_df.append(df)
    merged_df = pd.concat(all_df, ignore_index=True)
    merged_df.to_csv(output_file, index=False)

# Example usage:
directory_path = r"/Users/damianmoreno/imc_prosperity/data/prices_day1"  # Replace with the actual path
output_filename = "merged_price_day_1_data.csv"
merge_csv_files(directory_path, output_filename)