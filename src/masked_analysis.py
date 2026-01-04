import torch 
import torch.optim as optim 
import pandas as pd 
import numpy as np 
from pathlib import Path 

# My Imports
from src.utils.config_file_parser import config_file_parser
from src.utils.datapipeline import DataPipeline
from src.models.masked_lstm import MaskedLSTM
from src.models.masked_trainer import MaskedTrainer
from src.utils.loss_functions import RMSELoss
from src.utils.plot_predictions import plot_predictions

def main():
  # Load Config
  config_path = Path("config/masked_lstm_config.ini")
  config_data, _ = config_file_parser(config_path)
  
  hidden_size   = int(config_data["hidden_size"])
  num_layers    = int(config_data["num_layers"])
  dropout       = float(config_data["dropout"])
  learning_rate = float(config_data["learning_rate"])
  epochs        = int(config_data["epochs"])
  weight_decay  = float(config_data["weight_decay"])
  train_frac    = float(config_data["train_frac"])
  
  # Load window sizes
  window_data = Path("data/best_sliding_windows/fresno_best_2025-11-28_22-25-37.csv")
  df_window_sizes = pd.read_csv(window_data)
  window_sizes = df_window_sizes
  
  # Load the data
  data_file = Path("data/merged_rodent_fresno_agg.csv")
  
  pipeline = DataPipeline(
    data_file_path=data_file,
    config_data = config_data,
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
  
  trainer.train(
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
  
  rmse = np.sqrt(np.mean((preds - true)**2))
  print(f"Final test RMSE: {rmse:.4f}")
  
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
  # Plot everything
  # ----------------------------------------------------
  plot_predictions(
      true_train=true_train_inv,
      pred_train=train_preds_inv,
      true_test=true_test_inv,
      pred_test=pred_test_inv,
      title="Masked LSTM — True vs Predicted"
  )
  

if __name__ == "__main__":
  main()