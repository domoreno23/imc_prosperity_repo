"""
This script allows you to input a MERGED CSV for a given round and get a pandas dataframe of only one inputted asset
EX:
get_asset(merged_price_day_1_data.csv, "KELP")
The command above would provide you with a pandas dataframe for only kelp

NOTE: this currently does not give you a fully clean csv file just yet

Args:
merged_file (str): merged csv file for a given round
asset (str): desired asset
"""
import pandas as pd

def get_asset(merged_file, asset):
  df = pd.read_csv(merged_file, delimiter=';')
  filter_asset = df['product'] == asset
  asset_df = df[filter_asset]
  return asset_df

print(get_asset(merged_file="/Users/damianmoreno/imc_prosperity/data/prices_day1/merged_price_day_1_data.csv", asset="KELP"))