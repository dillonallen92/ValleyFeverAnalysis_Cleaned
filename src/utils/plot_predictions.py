import matplotlib.pyplot as plt 
import numpy as np 

def plot_predictions(true_train, pred_train, true_test, pred_test, tgt_variable : str = "VFRate", title="Fresno Masked LSTM Results", save_path=""):
  train_len = len(true_train)
  
  full_true = np.concatenate([true_train, true_test])
  full_pred = np.concatenate([pred_train, pred_test])
  
  plt.figure(figsize=(14,6))
  plt.plot(full_true, label=f"True {tgt_variable} Case Rates", linewidth=2, color="black")
  plt.plot(range(train_len), pred_train, label="Predicted (Train)", linestyle="-.", color="red")
  plt.plot(range(train_len, train_len + len(pred_test)), pred_test, label="Predicted (Test)", linestyle="-.", color="green")
  plt.title(title)
  plt.xlabel("Time (months)")
  plt.ylabel(tgt_variable)
  plt.grid(True, alpha = 0.3)
  plt.legend()
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path)     
 # plt.show()
  