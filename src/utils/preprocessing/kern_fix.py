import pandas as pd 
from pathlib import Path 

original_data_file_path = "data/kern_agg_drought_baseline.csv"
df = pd.read_csv(original_data_file_path)
print(df.head())
df = df.drop(columns=["FIRE_Acres_Burned"])
print(" ---- Dropping Fire ----")
print(df.head())
df.to_csv("data/kern_agg_drought_baseline_fixed.csv", index=False)