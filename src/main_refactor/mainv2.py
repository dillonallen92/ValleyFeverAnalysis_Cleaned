# STDLIB and other library imports
import numpy as np 
import pandas as pd 
import torch 
import torch.optim as optim 
from pathlib import Path 
import json 
import matplotlib.pyplot as plt 
import random 
import gc 
from sklearn.model_selection import ParameterGrid

# My own function imports
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

# Seed for anything random (like PFI and weight generation)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

def load_and_prep_data(data_path: Path, tgt_variable:str) -> tuple[np.ndarray, np.ndarray, list[str]]:
  df: pd.DataFrame = pd.read_csv(data_path)
  for col in ["YEAR_MONTH", "Year-Month", "DATE"]:
      if col in df.columns:
        df = df.drop(columns = [col])
  
  # Isolate the features I want
  feature_columns: list[str] = [col for col in df.columns if col != tgt_variable]
  print(feature_columns)
  # Create the feature and target vectors
  X: np.ndarray = df[feature_columns].values
  y: np.ndarray = df[tgt_variable].values 
  
  return X, y, feature_columns
  
def get_optimal_windows(
      X: np.ndarray,
      y: np.ndarray,
      feature_columns: list[str],
      params: dict,
      device: torch.device
) -> pd.DataFrame:
   """
   This function determines the optimal window size for each feature in the dataset
   after the dataset has been pre-processed
   """
   pipeline: SlidingWindowPipeline = SlidingWindowPipeline(X, y, test_frac = params['test_frac'])
   sliding_window_sizes = range(1,13)
   criterion = RMSELoss()
   results = []
   
   print(f"\n ---- Starting Window Search (Hidden: {params["hidden_size"]}, LR: {params["learning_rate"]}) ----\n")
   
   for feat_idx, feat_name in enumerate(feature_columns):
      print(f"Optimizing Feature [{feat_idx+1}/{len(feature_columns)}]: {feat_name}\n")
      for win_size in sliding_window_sizes:
         batch: SlidingWindowBatch = pipeline.scale(
            *pipeline.create_single_feature_sequences(feat_idx, win_size)
         )
         
         # create a 'validation set' of data within the training set to make sure
         # the model isn't trying to hide within 12 month windows to minimize RMSE
         train_size:float = .80
         val_size:float = 1 - train_size
         
         num_samples_train:int = int(np.floor(batch.X_train.shape[0] * train_size))
         
         X_train = batch.X_train[:num_samples_train].to(device)
         y_train = batch.y_train[:num_samples_train].to(device)
         
         X_val = batch.X_train[num_samples_train:].to(device)
         y_val = batch.y_train[num_samples_train:].to(device)
                  
         model = LSTM(
            input_size = 1,
            hidden_size=params["hidden_size"],
            num_layers = params["num_layers"],
            dropout= params["dropout"]
         ).to(device)
         
         optimizer = optim.Adam(
            model.parameters(),
            lr = params["learning_rate"],
            weight_decay = params["weight_decay"]
         )
         
         trainer = Trainer(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            scaler_y=batch.scaler_y
            )
         
         trainer.train(
            X_train = X_train,
            y_train= y_train,
            epochs = params["epochs"] 
         )
         
         preds, true = trainer.evaluate(X_test = X_val, y_test = y_val)
         
         rmse_sw = np.sqrt(np.mean((preds - true)**2))
         results.append({"feature": feat_name, "window_size": win_size, "rmse": rmse_sw})
         print(f" > Win: {win_size:2d} | Val RMSE: {rmse_sw:.6f}\n")
         
         del model, optimizer, trainer, batch 
         torch.mps.empty_cache()
   
   results_df: pd.DataFrame = pd.DataFrame(results)
   best_Vals = results_df.loc[results_df.groupby("feature")["rmse"].idxmin()]
   
   print("\nSearch Complete. Selected Windows:")
   for _, row in best_Vals.iterrows():
      print(f"   {row['feature']:20s} : Window = {int(row["window_size"])}")
  
   return best_Vals









if __name__ == "__main__":
  county_name = "Fresno"
  tgt_variable = "VFRate"
  datafile_version = "baseline"
  
  device = torch.device("mps" if torch.backend.mps.is_available() else "cpu")
  print(f"Using device: {device}")
  timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
  base_run_dir = Path(f"data/runs/{county_name}_{timestamp}")
  
  param_grid = {
    "hidden_size": [32, 64, 128],
    "lr": [8e-3, 5e-3, 1e-3, 8e-4, 5e-4, 1e-4],
    "num_layers": [1, 2],
    "epochs": [50, 100, 120, 150, 200],
    "dropout": [.20, .40],
    "weight_decay": [1e-4, 1e-5],
    "tgt_variable": [tgt_variable],
    "train_frac": 0.75
  }
  
  
  