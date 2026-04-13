import torch

class MaskedTrainer:
    def __init__(self, model, criterion, optimizer, scaler_y):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y

    def train(self, X_train, y_train, X_test, y_test, mask_train, mask_test, epochs):
        train_losses, test_losses = [], []
        
        # Calculate true variance ONCE before the loop starts
        true_var = torch.var(y_test)
        
        for epoch in range(epochs):
            self.model.train()

            preds = self.model(X_train, mask_train)
            loss = self.criterion(preds, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                preds_test = self.model(X_test, mask_test)
                test_loss = self.criterion(preds_test, y_test).item()
                test_losses.append(test_loss)

                # ---------------------------------------------------------
                # Convergence Monitor (Circuit Breaker)
                # ---------------------------------------------------------
                # Wait 15 epochs for the model to warm up before checking
                if epoch > 15:
                    pred_var = torch.var(preds_test)
                    # Add 1e-8 to prevent division by zero just in case
                    if pred_var / (true_var + 1e-8) < 0.005:
                        print(f"  !!! Early Exit at Epoch {epoch+1:4d}: Mean Prediction Detected !!!")
                        return None # Signals failure to run_trial
                # ---------------------------------------------------------

            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch + 1:4d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}")
                
        history = {"train" : train_losses, "test": test_losses}
        return history

    def evaluate(self, X_test, y_test, mask):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test, mask).cpu().numpy().reshape(-1, 1)
            true  = y_test.cpu().numpy().reshape(-1, 1)

            preds_inv = self.scaler_y.inverse_transform(preds).flatten()
            true_inv  = self.scaler_y.inverse_transform(true).flatten()

        return preds_inv, true_inv