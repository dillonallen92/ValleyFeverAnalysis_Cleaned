import matplotlib.pyplot as plt 

def plot_loss_curves(history : dict[str, list[float]], save_path = None):
  fig, axes = plt.subplots(1,2, figsize=(14,6))
  
  axes[0].plot(history["train"], label="Train Loss", color = "blue")
  axes[0].set_title("Training Loss")
  axes[0].set_xlabel("Epoch")
  axes[0].set_ylabel("Loss")
  axes[0].grid(True)
  
  axes[1].plot(history["test"], label="Test Loss", color="red")
  axes[1].set_title("Test Loss")
  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("Loss")
  axes[1].grid(True)
  
  plt.tight_layout()
  
  if save_path:
    plt.savefig(save_path)
  
  plt.show()