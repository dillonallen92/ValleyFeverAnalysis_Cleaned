# This file is the main analysis file
# masked_analysis.py is focused on testing the masking for a pre-computed window
# this script will compute the window, pass into mask, and generate the actual data 
# this unifies both ideas into one main script
# -> this script will output a plot and data package that contains the sliding window values, test RMSE for each,
#    prediction vs true values, and all of the parameters used for that run

import numpy as np 
import pandas as pd 
import torch 
import torch.optim as optim 
from pathlib import Path 
import json
import matplotlib.pyplot as plt

# My imports
from src.models.lstm import LSTM 
from src.models.masked_lstm import MaskedLSTM
from src.models.trainer import Trainer
from src.models.masked_trainer import MaskedTrainer
from src.utils.loss_functions import RMSELoss
from src.utils.config_file_parser import config_file_parser
from src.utils.plot_predictions import plot_predictions
from src.utils.plot_losses import plot_loss_curves
from src.utils.datapipeline import DataPipeline
from src.utils.window_sizes.sliding_window_pipeline import SlidingWindowPipeline, SlidingWindowBatch
from src.interpretability.pfi import permutation_feature_importance
from src.interpretability.pfi_plots import plot_pfi_radar, plot_pfi_bar
from src.utils.metric_functions import rmse

def main():
  # Change if needed
  county_name = "Fresno"
  rodent_flag = True
  # Timestamp to track creation of run data
  timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
  config_path = Path("config/masked_lstm_config.ini")
  if rodent_flag:
    data_path   = Path(f"data/merged_rodent_{county_name.lower()}_agg.csv")
    run_dir     = Path(f"data/runs/{county_name.lower()}_Rat_{timestamp}")
  else:
     data_path  = Path(f"data/{county_name.lower()}_Aggregate.csv")
     run_dir    = Path(f"data/runs/{county_name.lower()}_noRat_{timestamp}")

  run_dir.mkdir(parents=True, exist_ok=True)
  
  # ------------- DO NOT TOUCH BELOW HERE --------------
  
  # load config data
  lstm_params, pipeline_params = config_file_parser(config_path=config_path)
  hidden_size                  = int(lstm_params["hidden_size"])
  num_layers                   = int(lstm_params["num_layers"])
  dropout                      = float(lstm_params["dropout"])
  learning_rate                = float(lstm_params["learning_rate"])
  epochs                       = int(lstm_params["epochs"])
  weight_decay                 = float(lstm_params["weight_decay"])
  train_frac                   = float(lstm_params["train_frac"])
  test_frac                    = 1 - train_frac
  
  # import and clean the dataframe (remove date)
  df = pd.read_csv(data_path)
  for col in ["YEAR_MONTH", "Year-Month", "DATE"]:
      if col in df.columns:
        df = df.drop(columns = [col])
  
  # Isolate the features I want
  feature_columns = [col for col in df.columns if col != "VFRate"]
  
  # Create the feature and target vectors
  X = df[feature_columns].values
  y = df["VFRate"].values 
  
  ########################################
  #       Sliding Window Pipeline        #
  ########################################
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
      
      rmse_sw = np.sqrt(np.mean((preds - true)**2))
      results.append({
        "feature": feat_name, 
        "window_size": win_size,
        "rmse": rmse_sw
      })
      print(f"{feat_name:20s} | w={win_size:2d} | RMSE = {rmse_sw:.4f}")
  
  results_df = pd.DataFrame(results)
  best_vals = results_df.loc[results_df.groupby("feature")["rmse"].idxmin()]
  
  ########################################
  #         Masked LSTM Analysis         #
  ########################################
  window_sizes = best_vals
  
  pipeline = DataPipeline(
    data_file_path=data_path,
    config_data = lstm_params,
    window_sizes = window_sizes
  )
  
  pipeline.load_data()
  pipeline.build_windows()
  dataset = pipeline.scale(*pipeline.split())
  
  mask_batch = dataset.mask.unsqueeze(-1)
  mask_batch = mask_batch.permute(2,0,1)
  
  model = MaskedLSTM(
    input_size = dataset.X_train.shape[1],
    hidden_size= hidden_size,
    num_layers= num_layers,
    dropout= dropout
  )
  
  optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
  criterion = RMSELoss()
  
  trainer = MaskedTrainer(
    model = model, 
    criterion= criterion,
    optimizer= optimizer,
    scaler_y= dataset.scaler_y
  )
  
  history = trainer.train(
    X_train = dataset.X_train.permute(2,0,1),
    y_train = dataset.y_train,
    X_test= dataset.X_test.permute(2,0,1),
    y_test=dataset.y_test,
    mask_train = mask_batch.repeat(dataset.X_train.shape[2], 1, 1),
    mask_test = mask_batch.repeat(dataset.X_test.shape[2], 1, 1),
    epochs= epochs
  )
  
  preds, true = trainer.evaluate(
    X_test = dataset.X_test.permute(2,0,1),
    y_test= dataset.y_test,
    mask= mask_batch.repeat(dataset.X_test.shape[2], 1, 1)
  )
  
  rmse_final = np.sqrt(np.mean((preds - true)**2))
  print(f"Final test RMSE: {rmse_final:.4f}")
  
  # ----------------------------------------------------
  # Compute training predictions
  # ----------------------------------------------------
  with torch.no_grad():
      train_preds = model(
          dataset.X_train.permute(2,0,1),
          mask_batch.repeat(dataset.X_train.shape[2], 1, 1)
      ).cpu().numpy().reshape(-1,1)

  train_preds_inv = dataset.scaler_y.inverse_transform(train_preds).flatten()
  true_train_inv  = dataset.scaler_y.inverse_transform(dataset.y_train.cpu().numpy().reshape(-1,1)).flatten()

  # test preds already computed
  pred_test_inv = preds
  true_test_inv = true

  # ----------------------------------------------------
  # Plot everything and start saving data
  # ----------------------------------------------------
  plot_predictions(
      true_train=true_train_inv,
      pred_train=train_preds_inv,
      true_test=true_test_inv,
      pred_test=pred_test_inv,
      title=f"({county_name}) Masked LSTM — True vs Predicted",
      save_path=run_dir/"prediction_curve.png"
  )
  
  # saving data
  print("---- Saving Data ----")
  params_to_save = {
    "hidden_size" : hidden_size,
    "num_layers" : num_layers, 
    "dropout" : dropout, 
    "learning_rate": learning_rate,
    "epochs" : epochs, 
    "weight_decay" : weight_decay, 
    "train_frac" : train_frac, 
    "test_frac" : test_frac
  }
  
  with open(run_dir / "parameters.json", "w") as f:
    json.dump(params_to_save, f, indent = 4)
  
  results_df.to_csv(run_dir/"sliding_window_results.csv", index = False)
  best_vals.to_csv(run_dir/"best_window_sizes.csv", index = False)
  
  pred_data = pd.DataFrame({
    "true_test" : true_test_inv,
    "pred_test" : pred_test_inv
  })
  pred_data.to_csv(run_dir/"true_vs_pred_test.csv", index=False)
  
  # plot and save history curve
  plot_loss_curves(history, save_path = run_dir/"loss_curves.png")
  
  if bool(pipeline_params["run_pfi"]):
     importances, baseline_error = permutation_feature_importance(model = model,
                                                                  X_test = dataset.X_test.permute(2,0,1),
                                                                  y_test = dataset.y_test,
                                                                  mask = mask_batch.repeat(dataset.X_test.shape[2], 1, 1),
                                                                  scaler_y = dataset.scaler_y,
                                                                  metric_fn = rmse,
                                                                  n_repeats = 40
                                                                  )
  
  pfi_df = pd.DataFrame({
     "Feature": feature_columns,
     "Importance": importances
  })
  
  pfi_df["key"] = pfi_df["Feature"].str.lower()
  window_sizes["key"] = window_sizes["feature"].str.lower()
  
  pfi_df_merged = pfi_df.merge(
     window_sizes[["key", "feature", "window_size"]],
     on="key",
     how="inner"
  )
  pfi_df_merged.drop(columns=["key"])
  
  pfi_df = pfi_df_merged[["Feature", "Importance", "window_size"]]
  print(pfi_df)
  pfi_df.to_csv(run_dir/"pfi_importance.csv", index = False)
  
  plot_pfi_radar(pfi_df, save_path=run_dir/"pfi_radar_plot.png", title=f"{county_name} Permutation Feature Importance (Radar)")
  
  plot_pfi_bar(pfi_df, save_path=run_dir/"pfi_bar.png", title=f"{county_name} Permutation Feature Importance (Bar)")
  
  plt.show()
  
if __name__ == "__main__":
  main()