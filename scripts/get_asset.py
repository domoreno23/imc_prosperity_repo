"""
This script allows you to input a CSV for a given round and get a CSV file back of only one inputted asset
EX:
get_asset(merged_price_day_1_data.csv, "KELP")
The command above would provide you with a CSV file for only kelp

NOTE: this currently does not give you a fully clean csv file just yet

Args:
merged_file (str): merged csv file for a given round
asset (str): desired asset
"""
import pandas as pd

def get_asset(file, asset):
  df = pd.read_csv(file, delimiter=';')
  filter_asset = df['product'] == asset
  asset_df = df[filter_asset]
  asset_df.to_csv("macarons.csv")

get_asset(file="", asset="MAGNIFICENT_MACARONS")