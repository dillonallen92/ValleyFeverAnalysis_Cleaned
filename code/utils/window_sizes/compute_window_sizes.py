# Python Package Imports
import numpy as np 
import torch 
import pandas as pd 
import torch.optim as optim 
from pathlib import Path 

# My own imports
from code.models.lstm import LSTM 
from code.models.trainer import Trainer
from code.utils.loss_functions import RMSELoss
from code.utils.config_file_parser import config_file_parser
from code.utils.window_sizes.sliding_window_pipeline import SlidingWindowPipeline, SlidingWindowBatch
  
def main():
  config_path : Path = Path("config/masked_lstm_config.ini")
  data_path : Path = Path("data/merged_rodent_fresno_agg.csv")
  
  # load config data
  lstm_params, _ = config_file_parser(config_path=config_path)
  hidden_size    = int(lstm_params["hidden_size"])
  num_layers     = int(lstm_params["num_layers"])
  dropout        = float(lstm_params["dropout"])
  learning_rate  = float(lstm_params["learning_rate"])
  epochs         = int(lstm_params["epochs"])
  weight_decay   = float(lstm_params["weight_decay"])
  train_frac     = float(lstm_params["train_frac"])
  test_frac      = 1 - train_frac
  
  # import and clean the dataframe (remove date)
  df = pd.read_csv(data_path)
  df = df.drop(columns=["YEAR_MONTH"])
  
  # Isolate the features I want
  feature_columns = [col for col in df.columns if col != "VFRate"]
  
  # Create the feature and target vectors
  X = df[feature_columns].values
  y = df["VFRate"].values 
  
  # Create the datapipeline for sliding window calculations
  pipeline = SlidingWindowPipeline(X, y, test_frac=test_frac)
  
  # create a list of sliding window sizes from 1 to 12
  sliding_window_sizes = [x for x in range(1,13)]
  criterion = RMSELoss()
  
  results = [] 
  
  for feat_idx, feat_name in enumerate(feature_columns):
    for win_size in sliding_window_sizes:
      batch : SlidingWindowBatch = pipeline.scale(*pipeline.create_single_feature_sequences(feat_idx, win_size))
      
      model = LSTM(
        input_size = 1,
        hidden_size= hidden_size, 
        num_layers= num_layers,
        dropout= dropout
      )
      
      optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
      trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scaler_y = batch.scaler_y)
      
      trainer.train(X_train = batch.X_train, y_train = batch.y_train, epochs=epochs)
      preds, true = trainer.evaluate(batch.X_test, batch.y_test)
      
      rmse = np.sqrt(np.mean((preds - true)**2))
      results.append({
        "feature": feat_name, 
        "window_size": win_size,
        "rmse": rmse
      })
      print(f"{feat_name:20s} | w={win_size:2d} | RMSE = {rmse:.4f}")
  
  print(" ---- Saving Files ----")
  results_df = pd.DataFrame(results)
  timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
  results_df.to_csv(f"data/sliding_window_calc/fresno_{timestamp}.csv", index=False)
  best_vals = results_df.loc[results_df.groupby("feature")["rmse"].idxmin()]
  best_vals.to_csv(f"data/best_sliding_windows/fresno_best_{timestamp}.csv", index=False)
  print(" ---- Files Saved ----")

if __name__ == "__main__":
  main()