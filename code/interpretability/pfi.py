import torch 
import numpy as np 

def permutation_feature_importance(model, X_test, y_test, mask, scaler_y, metric_fn, n_repeats = 5):
  model.eval()
  
  with torch.no_grad():
    base_preds = model(X_test, mask).cpu().numpy().reshape(-1,1)
    base_preds = scaler_y.inverse_transform(base_preds).flatten()
    base_true  = scaler_y.inverse_transform(y_test.cpu().numpy()).flatten()
  
  baseline_error = metric_fn(base_true, base_preds)
  
  num_features = X_test.shape[-1]
  importances = []
  
  for feat in range(num_features):
    errors = [] 
    
    for _ in range(n_repeats):
      X_perm = X_test.clone()
      idx = torch.randperm(X_test.shape[0])
      X_perm[:,:,feat] = X_test[idx, :, feat]
      
      with torch.no_grad():
        preds = model(X_perm, mask).cpu().numpy().reshape(-1, 1)
        preds = scaler_y.inverse_transform(preds).flatten()
      
      err = metric_fn(base_true, preds)
      errors.append(err)
      
    importance = np.mean(errors) - baseline_error
    importances.append(importance)

  return np.array(importances), baseline_error