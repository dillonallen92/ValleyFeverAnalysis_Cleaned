import pandas as pd 
from pathlib import Path 

rodent_dir: Path = Path("data/Kern_Rodent_Data")

frames:dict = []

for item in rodent_dir.iterdir():
  df: pd.DataFrame = pd.read_csv(item, 
                   delimiter='\t',
                   usecols=["YEAR", "DATE", "COUNTY_NAME", "POUNDS_PRODUCT_APPLIED"])
  df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b-%y")
  df["YEAR_MONTH"] = df["DATE"].dt.strftime("%Y-%m")

  monthly: pd.DataFrame = (
    df.groupby("YEAR_MONTH")["POUNDS_PRODUCT_APPLIED"]
      .sum()
      .reset_index()
    )
  frames.append(monthly)

main_rodent_df: pd.DataFrame = pd.concat(frames, ignore_index=True)
main_rodent_df = main_rodent_df.sort_values("YEAR_MONTH").reset_index(drop=True)

# get fresno data, rename the Year-Month index to make merging easier
main_df: pd.DataFrame = pd.read_csv("data/Kern_Aggregate.csv")
cpy_main: pd.DataFrame = main_df.copy()
cpy_main = cpy_main.rename(columns={"Year-Month":"YEAR_MONTH"})

# merge the two
merged: pd.DataFrame = pd.merge(
  cpy_main,
  main_rodent_df,
  on="YEAR_MONTH",
  how="inner"
)
print(merged)
merged.to_csv('data/merged_rodent_kern_agg.csv', index=False)



