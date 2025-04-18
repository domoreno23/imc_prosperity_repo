import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def show_data(file, product_name):
    # Step 1: Load the CSV using the semicolon delimiter.
    # Replace 'your_file.csv' with the path to your CSV file.
    data = pd.read_csv(file, sep=';')

    # Print the raw columns to see what pandas interpreted.
    print("Columns before cleaning:", data.columns.tolist())

    # Step 2: Drop columns that are completely empty (if they exist).
    data = data.dropna(axis=1, how='all')
    print("Columns after dropping fully empty columns:", data.columns.tolist())

    # At this point, you should have the primary columns along with auto-generated ones
    # for the additional fields. For example, if the CSV has extra fields, pandas may call them:
    # ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2',
    #  'Unnamed: 7', 'Unnamed: 8', ...]
    #
    # Step 3: Rename auto-generated columns if you know their intended meaning.
    # In this example, we assume the following:
    # - Unnamed: 7  => ask_price_1
    # - Unnamed: 8  => ask_volume_1
    # - Unnamed: 9  => ask_price_2
    # - Unnamed: 10 => ask_volume_2
    rename_dict = {}
    for col in data.columns:
        if col.startswith("Unnamed"):
            col_index = data.columns.get_loc(col)
            if col_index == 7:
                rename_dict[col] = "ask_price_1"
            elif col_index == 8:
                rename_dict[col] = "ask_volume_1"
            elif col_index == 9:
                rename_dict[col] = "ask_price_2"
            elif col_index == 10:
                rename_dict[col] = "ask_volume_2"
            # Add further renames if you have more extra columns.
    data.rename(columns=rename_dict, inplace=True)
    print("Columns after renaming:", data.columns.tolist())

    # Step 4: Convert the price and volume columns to numeric types.
    # This ensures that plotting works correctly.
    numeric_cols = ["bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
                    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # OPTIONAL: If your timestamps are not in a standard datetime format,
    # and your 'day' column represents a day offset (with 'timestamp' as seconds),
    # you might want to combine them. For example, assuming day 0 corresponds to a known start:
    # (Adjust start_date as needed for your data.)
    start_date = datetime(2025, 1, 1)
    data['datetime'] = data.apply(lambda row: start_date + timedelta(days=row['day'], 
                                                                    seconds=row['timestamp']), axis=1)

    # Step 5: Filter the data for a specific product if desired (e.g., "RAINFOREST_RESIN")
    subset = data[data["product"] == product_name]

    # ---------------------------
    # Step 5. Plot the Data
    # ---------------------------
    plt.figure(figsize=(12, 6))

    # Plot Bid Price (using bid_price_1),
    # Ask Price (using ask_price_1), 
    # and Mid Price computed above.
    plt.plot(subset["datetime"], subset["bid_price_1"], label="Bid Price",  color="b")
    plt.plot(subset["datetime"], subset["ask_price_1"], label="Ask Price", color="r")
    plt.plot(subset["datetime"], subset["mid_price"], label="Mid Price", color="g")


    plt.plot(subset["datetime"], subset["bid_price_2"], label="Bid Price2", color="c")
    plt.plot(subset["datetime"], subset["ask_price_2"], label="Ask Price3", color="m")



    # Labeling the chart
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Bid, Ask, and Mid Prices Over Time for {product_name}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


show_data("/Users/damianmoreno/imc_prosperity/data/prices_round4/prices_round_4_day_1.csv", "MAGNIFICENT_MACARONS")