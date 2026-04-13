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
from sklearn.model_selection import ParameterSampler

# Custom function imports
from src.models.lstm import LSTM 
from src.models.masked_lstm import MaskedLSTM
from src.models.trainer import Trainer
from src.main_refactor.masked_trainer2 import MaskedTrainer
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

# ----------------------------------------------------
# Modular Functions
# ----------------------------------------------------

def load_and_prep_data(data_path: Path, tgt_variable: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df: pd.DataFrame = pd.read_csv(data_path)
    for col in ["YEAR_MONTH", "Year-Month", "DATE"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Isolate the features
    feature_columns: list[str] = [col for col in df.columns if col != tgt_variable]
    print(f"Features: {feature_columns}")
    
    # Create the feature and target vectors
    X: np.ndarray = df[feature_columns].values
    y: np.ndarray = df[tgt_variable].values 
    
    return X, y, feature_columns

def get_optimal_windows(X: np.ndarray, y: np.ndarray, feature_columns: list[str], params: dict, device: torch.device) -> pd.DataFrame:
    """
    Determines the optimal window size for each feature in the dataset.
    """
    pipeline = SlidingWindowPipeline(X, y, test_frac=params['test_frac'])
    sliding_window_sizes = range(1, 13)
    criterion = RMSELoss()
    results = []
    
    print(f"\n ---- Starting Window Search (Hidden: {params['hidden_size']}, LR: {params['learning_rate']}) ----\n")
    
    for feat_idx, feat_name in enumerate(feature_columns):
        print(f"Optimizing Feature [{feat_idx+1}/{len(feature_columns)}]: {feat_name}")
        for win_size in sliding_window_sizes:
            batch: SlidingWindowBatch = pipeline.scale(
                *pipeline.create_single_feature_sequences(feat_idx, win_size)
            )
            
            # Validation split
            train_size: float = 0.80
            num_samples_train: int = int(np.floor(batch.X_train.shape[0] * train_size))
            
            X_train = batch.X_train[:num_samples_train].to(device)
            y_train = batch.y_train[:num_samples_train].to(device)
            
            X_val = batch.X_train[num_samples_train:].to(device)
            y_val = batch.y_train[num_samples_train:].to(device)
                    
            model = LSTM(
                input_size=1,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            ).to(device)
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"]
            )
            
            trainer = Trainer(
                model=model, 
                criterion=criterion, 
                optimizer=optimizer, 
                scaler_y=batch.scaler_y
            )
            
            trainer.train(
                X_train=X_train,
                y_train=y_train,
                epochs=params["epochs"] 
            )
            
            preds, true = trainer.evaluate(X_test=X_val, y_test=y_val)
            
            rmse_sw = float(np.sqrt(np.mean((preds - true)**2)))
            results.append({"feature": feat_name, "window_size": win_size, "rmse": rmse_sw})
            print(f"  > Win: {win_size:2d} | Val RMSE: {rmse_sw:.6f}")
            
            del model, optimizer, trainer, batch 
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
    results_df = pd.DataFrame(results)
    best_vals = results_df.loc[results_df.groupby("feature")["rmse"].idxmin()]
    
    print("\nSearch Complete. Selected Windows:")
    for _, row in best_vals.iterrows():
        print(f"   {row['feature']:20s} : Window = {int(row['window_size'])}")
   
    return best_vals

def run_trial(data_path: Path, best_vals: dict[str, int], params: dict, device: torch.device):
    """Executes a single trial of the Masked LSTM pipeline."""
    
    # 1. OVERWRITE BASELINE: Merge the trial-specific params into the config
    # This allows DataPipeline to see the swept train_frac, epochs, etc.
    current_config = params['lstm_params'].copy()
    for k, v in params.items():
        if k != 'lstm_params':
            current_config[k] = v

    # 2. Pipeline Setup
    pipeline = DataPipeline(
        data_file_path=data_path,
        config_data=current_config, 
        window_sizes=best_vals,
        tgt_variable=params['tgt_variable']
    )
    
    pipeline.load_data()
    pipeline.build_windows()
    dataset = pipeline.scale(*pipeline.split())
    
    # 3. Mask and Data Formatting
    mask_batch = dataset.mask.unsqueeze(-1).permute(2, 0, 1)
    
    X_train_t = dataset.X_train.permute(2, 0, 1).to(device)
    y_train_t = dataset.y_train.to(device)
    X_test_t = dataset.X_test.permute(2, 0, 1).to(device)
    y_test_t = dataset.y_test.to(device)
    
    mask_train_t = mask_batch.repeat(dataset.X_train.shape[2], 1, 1).to(device)
    mask_test_t = mask_batch.repeat(dataset.X_test.shape[2], 1, 1).to(device)
    
    # 4. Model & Trainer Instantiation
    model = MaskedLSTM(
        input_size=dataset.X_train.shape[1],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
    )
    
    criterion = RMSELoss()
    
    trainer = MaskedTrainer(
        model=model, 
        criterion=criterion,
        optimizer=optimizer,
        scaler_y=dataset.scaler_y
    )
    
    # 5. Execution
    history = trainer.train(
        X_train=X_train_t, y_train=y_train_t,
        X_test=X_test_t, y_test=y_test_t,
        mask_train=mask_train_t, mask_test=mask_test_t,
        epochs=params['epochs']
    )
    
    if history is None:
        return {'rmse': float('inf'), 'state_dict': None}
    
    # 6. Evaluation
    pred_test_inv, true_test_inv = trainer.evaluate(X_test_t, y_test_t, mask_test_t)
    rmse_final = float(np.sqrt(np.mean((pred_test_inv - true_test_inv)**2)))
    
    train_preds_inv, true_train_inv = trainer.evaluate(X_train_t, y_train_t, mask_train_t)
    
    # 7. Memory Cleanup
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    
    return {
        'rmse': rmse_final,
        'state_dict': state_dict_cpu,
        'train_preds': train_preds_inv,
        'true_train': true_train_inv,
        'test_preds': pred_test_inv,
        'true_test': true_test_inv,
        'history': history
    }

def search_manager(data_path: Path, frozen_windows: dict[str, int], grid: ParameterGrid, device: torch.device):
    """Iterates through hyperparameter grid using frozen optimal window sizes."""
    best_rmse = float('inf')
    best_state = None
    best_params = None
    best_metrics = {}

    print(f"\nStarting Grid Search over {len(grid)} combinations...")

    for i, trial_params in enumerate(grid):
        print(f"\nTrial {i+1}/{len(grid)}: {trial_params}")
        
        results = run_trial(data_path, frozen_windows, trial_params, device)
        
        if results['rmse'] < best_rmse:
            print(f"*** New Best RMSE: {results['rmse']:.4f} ***")
            best_rmse = results['rmse']
            best_state = results['state_dict']
            best_params = trial_params
            best_metrics = {
                'train_preds': results['train_preds'],
                'true_train': results['true_train'],
                'test_preds': results['test_preds'],
                'true_test': results['true_test'],
                'history': results['history']
            }

        # Force VRAM cleanup after every trial
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    return best_rmse, best_state, best_params, best_metrics


# ----------------------------------------------------
# Main Execution Block
# ----------------------------------------------------

if __name__ == "__main__":
    # --- 1. Global Setup & Flags ---
    county_name = "Fresno"
    datafile_version = "baseline" 
    tgt_variable = "VFRate"
    rodent_flag = True
    drought_flag = True
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Path & Directory Routing ---
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if rodent_flag and drought_flag:
        data_path = Path(f"data/{county_name.lower()}_agg_drought_{datafile_version}.csv")
        base_run_dir = Path(f"data/runs/{county_name.lower()}_Rat_Drought_{datafile_version}_{timestamp}")
    elif rodent_flag:
        data_path = Path(f"data/merged_rodent_{county_name.lower()}_agg.csv")
        base_run_dir = Path(f"data/runs/{county_name.lower()}_Rat_{timestamp}")
    else:
        data_path = Path(f"data/{county_name.lower()}_Aggregate.csv")
        base_run_dir = Path(f"data/runs/{county_name.lower()}_noRat_{timestamp}")

    base_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading data from: {data_path}")
    print(f"Saving run to: {base_run_dir}")
    
    # --- 3. Load Configs (USING YOUR EXPLICIT CASTING) ---
    config_path = Path(f"config/masked_lstm_config_{county_name.lower()}.ini")
    raw_lstm_params, pipeline_params = config_file_parser(config_path=config_path)
    
    train_frac = float(raw_lstm_params["train_frac"])
    
    # Pack the explicitly cast variables into a dictionary for the pipeline
    typed_lstm_params = {
        "hidden_size": int(raw_lstm_params["hidden_size"]),
        "num_layers": int(raw_lstm_params["num_layers"]),
        "dropout": float(raw_lstm_params["dropout"]),
        "learning_rate": float(raw_lstm_params["learning_rate"]),
        "epochs": int(raw_lstm_params["epochs"]),
        "weight_decay": float(raw_lstm_params["weight_decay"]),
        "train_frac": train_frac,
        "test_frac": 1.0 - train_frac
    }

    # --- 4. Phase 1: Window Search Phase ---
    X, y, feature_cols = load_and_prep_data(data_path, tgt_variable)
    
    best_vals_df = get_optimal_windows(X, y, feature_cols, typed_lstm_params, device)
    frozen_windows = best_vals_df
    
    # --- 5. Phase 2: Hyperparameter Grid Search ---
    param_grid = {
        "hidden_size": [32, 48, 64, 96, 128],
        "learning_rate": [5e-3, 1e-3, 3e-3, 8e-4, 5e-4, 1e-4],
        "num_layers": [2],
        "epochs": [100, 120, 150, 200, 250],
        "dropout": [.20, .30, .40, .50],
        "weight_decay": [1e-4, 1e-5, 1e-6, 1e-7],
        "tgt_variable": [tgt_variable],
        "train_frac": [0.70, 0.75, 0.80], 
        "lstm_params": [typed_lstm_params] # Pass the safe, typed dictionary here
    }
    
    # grid = ParameterGrid(param_grid)
    grid = ParameterSampler(param_grid, n_iter = 500, random_state=SEED)
    
    best_rmse, best_state, best_params, best_metrics = search_manager(
        data_path=data_path, 
        frozen_windows=frozen_windows, 
        grid=grid, 
        device=device
    )
    
    print("\n" + "="*50)
    print("SEARCH COMPLETE")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("="*50)
    
    # --- 6. Victory Lap: Reconstruct the Best Environment ---
    current_config = best_params['lstm_params'].copy()
    for k, v in best_params.items():
        if k != 'lstm_params':
            current_config[k] = v
            
    best_pipeline = DataPipeline(
        data_file_path=data_path,
        config_data=current_config,
        window_sizes=frozen_windows,
        tgt_variable=best_params['tgt_variable']
    )
    best_pipeline.load_data()
    best_pipeline.build_windows()
    best_dataset = best_pipeline.scale(*best_pipeline.split())
    
    best_mask_batch = best_dataset.mask.unsqueeze(-1).permute(2,0,1)
    
    best_model = MaskedLSTM(
        input_size=best_dataset.X_train.shape[1],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    best_model.load_state_dict(best_state)
    best_model.to(device)
    best_model.eval()

    # --- 7. Save Artifacts & Configurations ---
    print("---- Saving Best Model Data ----")
    
    params_to_save = {k: v for k, v in best_params.items() if k != 'lstm_params'}
    with open(base_run_dir / "best_parameters.json", "w") as f:
        json.dump(params_to_save, f, indent=4)
        
    best_vals_df.to_csv(base_run_dir / "best_window_sizes.csv", index=False)
    torch.save(best_state, base_run_dir / "best_model_weights.pt")

    pd.DataFrame({
        "sample_index": np.arange(len(best_metrics['true_test'])),
        "true_test": best_metrics['true_test'],
        "pred_test": best_metrics['test_preds']
    }).to_csv(base_run_dir / "true_vs_pred_test.csv", index=False)

    pd.DataFrame({
        "sample_index": np.arange(len(best_metrics['true_train'])),
        "true_train": best_metrics['true_train'],
        "pred_train": best_metrics['train_preds']
    }).to_csv(base_run_dir / "true_vs_pred_train.csv", index=False)

    pd.DataFrame({
        "epoch": np.arange(1, len(best_metrics['history']["train"]) + 1),
        "train_loss": best_metrics['history']["train"],
        "test_loss": best_metrics['history']["test"]
    }).to_csv(base_run_dir / "training_history.csv", index=False)

    # --- 8. Plotting ---
    plot_predictions(
        true_train=best_metrics['true_train'],
        pred_train=best_metrics['train_preds'],
        true_test=best_metrics['true_test'],
        pred_test=best_metrics['test_preds'],
        tgt_variable=tgt_variable,
        title=f"({county_name}) Best Masked LSTM — True vs Predicted",
        save_path=base_run_dir / "prediction_curve.png"
    )
    
    plot_loss_curves(best_metrics['history'], save_path=base_run_dir / "loss_curves.png")

    # --- 9. Interpretability (PFI) ---
    # Need to check pipeline_params as a string since we didn't explicitly cast it
    if str(pipeline_params.get("run_pfi", "False")).lower() == "true":
        print("\n---- Running Permutation Feature Importance ----")
        
        X_test_pfi = best_dataset.X_test.permute(2,0,1).to(device)
        y_test_pfi = best_dataset.y_test.to(device)
        mask_test_pfi = best_mask_batch.repeat(best_dataset.X_test.shape[2], 1, 1).to(device)

        importances, baseline_error, all_results = permutation_feature_importance(
            model=best_model,
            X_test=X_test_pfi,
            y_test=y_test_pfi,
            mask=mask_test_pfi,
            scaler_y=best_dataset.scaler_y,
            metric_fn=rmse,
            n_repeats=40
        )
        
        # 1. Clean the strings to prevent whitespace mismatches
        clean_features = [str(f).strip() for f in feature_cols]
        clean_frozen = [str(f).strip() for f in frozen_windows['feature']]
        
        # 2. Build the DataFrame
        pfi_df = pd.DataFrame({
            "Feature": clean_features,
            "Importance": importances
        })
        
        # 3. Safely map the window sizes directly (No pd.merge required)
        win_dict = dict(zip(clean_frozen, frozen_windows['window_size']))
        pfi_df['window_size'] = pfi_df['Feature'].map(win_dict) 
        
        print(pfi_df)
        pfi_df.to_csv(base_run_dir / "pfi_importance.csv", index=False)
        
        plot_pfi_radar(pfi_df, save_path=base_run_dir/"pfi_radar_plot.png", title=f"{county_name} PFI (Radar)")
        plot_pfi_bar(pfi_df, save_path=base_run_dir/"pfi_bar.png", title=f"{county_name} PFI (Bar)")

    plt.show()