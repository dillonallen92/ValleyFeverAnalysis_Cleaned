import numpy as np 
import pandas as pd 
from models import LSTM
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import torch
from datetime import date 
from loss_functions import RMSELoss
import torch.optim as optim 
from datetime import datetime
import sys 
from config_file_parser import config_file_parser

def preprocess_data(X, y, feature_index, window_size, test_size=0.2):
  # Select the feature column
  if feature_index is not None:
    X_feature = X[:, feature_index].reshape(-1, 1)
  else:
    X_feature = X # its the whole feautre set

  # Create sequences for LSTM
  X_sequences, y_sequences = [], []
  for i in range(len(X_feature) - window_size):
    X_sequences.append(X_feature[i:i+window_size])
    y_sequences.append(y[i+window_size])

  X_sequences = np.array(X_sequences)
  y_sequences = np.array(y_sequences)

  # Train-test split
  split_index = int(len(X_sequences) * (1 - test_size))
  X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
  y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]

  return X_train, X_test, y_train, y_test 

def scale_data(X_train, X_test, y_train, y_test):
  # Reshape for scaling
  num_samples, window_size, num_features = X_train.shape
  X_train_reshaped = X_train.reshape(-1, num_features)
  X_test_reshaped = X_test.reshape(-1, num_features)

  # Scale features
  X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(num_samples, window_size, num_features)
  X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape[0], window_size, num_features)

  # Scale target
  y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
  y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

  return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

class TrainerNewNew:
  """
  Trainer Class: A class that contains the training, testing, and visualization functions.
  """
  def __init__(self, model, criterion, optimizer, scaler):
    """
    Initialize the class. Takes in a model, crtierion for loss, optimizer, scaler.
    
    Inputs:
      - Model: Neural Network model
      - Criterion: Loss function (Typically MSELoss for time series, may look into more)
      - Optimizer: Optimizer with learning rate added. Typically using Adam
      - Scaler: MinMaxScaler scaler value, used for inverse transform to get actual data back
    """
    self.model     = model 
    self.criterion = criterion 
    self.optimizer = optimizer
    self.scaler    = scaler
  
  def train(self, X_train, y_train, X_test, y_test, epochs):
    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(epochs):
        self.model.train()
        output = self.model(X_train)
        loss = self.criterion(output, y_train)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        if epoch % 10 == 0:
            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(X_test), y_test)
                history['test_loss'].append(val_loss.item())
                # print(f"Epoch {epoch+1}/{epochs} - Training Loss {loss.item():.4f}, Testing Loss {val_loss.item():.4f}")
    
    # Capture the final training predictions
    self.model.eval()
    with torch.no_grad():
        final_train_preds = self.model(X_train).detach().cpu().numpy()
        
    # Also inverse transform the training data for later plotting
    y_train_true = self.scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
    final_train_preds_inv = self.scaler.inverse_transform(final_train_preds).flatten()
    
    return history, final_train_preds_inv, y_train_true
    
  def evaluate(self, X_test, y_test):
    """
    Evaluation Loop. Evaluates the model and generates predictions.

    Inputs:
      - X_test: Test matrix X
      - y_test: Test target vector y
    
    Outputs:
      - y_pred: predicted target vector from the model using X_test
      - y_true: True target vector (y_test)
    """
    self.model.eval()
    with torch.no_grad():
        preds = self.model(X_test).detach().cpu().numpy()

        # Inverse transform the predictions using the y_scaler
        # The y_scaler was fit on a 1-D array, so the predictions should be reshaped
        vf_pred = self.scaler.inverse_transform(preds)

        # Inverse transform the true values using the y_scaler
        # The y_scaler was fit on a 1-D array, so the true values should be reshaped
        vf_true = self.scaler.inverse_transform(y_test.cpu().numpy())
        
    return vf_pred.flatten(), vf_true.flatten()
  
  def visualize_results(self, true, pred, county_name="", model_type = "LSTM", title_text = "", show_plot = True, save_fig = False):
    """
    Function to visualize the prediction vs true (test) vector

    Inputs:
      - True: True data (test or validation target vector)
      - Pred: Prediction data from the model evaluation function
    """
    if show_plot:
      plt.figure(figsize=(12, 6))
      plt.plot(true, label="True Values")
      plt.plot(pred[1:], label = "Predicted Values", linestyle="--")
      plt.title(f"{county_name} {model_type} {title_text} True vs Predicted Valley Fever Case Rates")
      plt.xlabel("Months")
      plt.ylabel("Case Rates")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()
    
    if save_fig:
      plt.figure(figsize=(12, 6))
      plt.plot(true, label="True Values")
      plt.plot(pred[1:], label = "Predicted Values", linestyle="--")
      plt.title(f"{county_name} {model_type} {title_text} LSTM True vs Predicted Valley Fever Case Rates")
      plt.xlabel("Months")
      plt.ylabel("Case Rates")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      img_str = f"Project/plots/{model_type}/{county_name}_{title_text}_plot_{date.today()}.png"
      plt.savefig(img_str)

if __name__ == "__main__":
  ####################
  #    Change Here   #
  ####################
  county_name = "Fresno" # change this between Fresno and Kern
  config_file_path = "Project/configs/masked_lstm_config.ini"
  
  ####################
  #  DO NOT CHANGE   #
  ####################
  
  # Functionality for command line args
  all_args = sys.argv 
  if len(all_args) == 2 and all_args[1] in ["Kern", "Fresno"]:
     county_name = all_args[1] 
     print(f" ---- Input Arg Detected - County set to {county_name} ----")
  else:
     print(f" ---- No Input Args- Default Set To {county_name} ----")
  
  today_val = datetime.now()
  timestamp_str = today_val.strftime("%Y-%m-%d_%H-%M-%S")
  save_file_str = f"{county_name}_sliding_window_calc_{timestamp_str}.csv"
  save_best_file_str = f"{county_name}_best_sliding_vals_{timestamp_str}.csv"
  data_path = f"Project/data/{county_name}_Aggregate.csv"
  df_data = pd.read_csv(data_path)
  
  # Prepare features for LSTM
  # Exclude 'Year-Month' and 'VFRate' (target variable)
  feature_columns = ['FIRE_Acres_Burned', 'PRECIP', 'WIND_EventCount', 'WIND_AvgMPH', 
      'WIND_RunMiles', 'AQI_PM25', 'AQI_PM10', 'EARTHQUAKE_Total', 'PESTICIDE_Total']

  # Create X (features) and y (target)
  X = df_data[feature_columns].values
  y = df_data['VFRate'].values

  # now what we want to do is loop through each feature and run the LSTM on each feature individually with a list of
  # sliding window sizes and see which feature and sliding window size gives us the best performance
  sliding_window_sizes: list[int] = [x for x in range(1, 13)]
  results = []
  scaler_X = MinMaxScaler()
  scaler_y = MinMaxScaler()
  
  # below are model parameters 
  # model parameters
  # lookback has been removed because we are varying the sliding window size
  # hidden_size          = 32
  # num_layers           = 2
  # dropout              = 0.2
  # learning_rate        = 0.001
  # epochs               = 300
  # weight_decay         = 1e-5
  
  lstm_params, _  = config_file_parser(config_path=config_file_path)
  hidden_size     = int(lstm_params["hidden_size"])
  num_layers      = int(lstm_params["num_layers"])
  dropout         = float(lstm_params["dropout"])
  learning_rate   = float(lstm_params["learning_rate"])
  epochs          = int(lstm_params["epochs"])
  weight_decay    = float(lstm_params["weight_decay"])
  train_frac      = float(lstm_params["train_frac"])
  test_frac       = 1 - train_frac
  
  

  criterion = RMSELoss()
  print("---- Computing Sliding Window Values ----")
  for feature_index, feature in enumerate(feature_columns):
    # generate the feature vector and target vector
    for window_size in sliding_window_sizes:
      X_train, X_test, y_train, y_test = preprocess_data(X, y, feature_index, window_size, test_size = test_frac)
      
      # scale the data
      X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = scale_data(X_train, X_test, y_train, y_test)
      
      # Convert to PyTorch tensors and reshape for LSTM input
      X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
      X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)
      y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
      y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)

      # Initialize the model
      model = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
      
      # Define optimizer
      optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
      
      # Create Trainer instance
      trainer = TrainerNewNew(model, criterion, optimizer, scaler_y)
      
      # Train the model
      history, train_preds_inv, y_train_true = trainer.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs)
      
      # Evaluate the model
      y_pred_inv, y_true_inv = trainer.evaluate(X_test_tensor, y_test_tensor)
      
      # Calculate RMSE for test set
      rmse = np.sqrt(np.mean((y_pred_inv - y_true_inv) ** 2))
      
      # Store results
      results.append({
          'feature': feature,
          'window_size': window_size,
          'rmse': rmse
      })
      
      print(f"Feature: {feature}, Window Size: {window_size}, Test RMSE: {rmse:.4f}")
      
      # Optionally visualize results for each run
      #trainer.visualize_results(y_true_inv, y_pred_inv, county_name="Fresno", model_type="LSTM", title_text=f"{feature} Window {window_size}", show_plot=True, save_fig=False)
  
  print(" ---- Storing all sliding window computations ----")
  results_df: pd.DataFrame = pd.DataFrame(results)
  print(results_df)
  print(" ---- Saving Feature Calculations to CSV ----")
  results_df.to_csv(f"Project/data/{save_file_str}", index=False)
  
  # for each feature, what is the sliding window that has the lowest RMSE?
  print("---- Storing all of the best sliding window values ----")
  best_results = results_df.loc[results_df.groupby('feature')['rmse'].idxmin()]
  print(" ---- Best Results ----")
  print(best_results)
  print(" ---- Saving Best Features to CSV ----")
  best_results.to_csv(f"Project/data/{save_best_file_str}", index=False)
  