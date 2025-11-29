import torch

class MaskedTrainer:
    def __init__(self, model, criterion, optimizer, scaler_y):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y

    def train(self, X_train, y_train, X_test, y_test, mask_train, mask_test, epochs):
        train_losses, test_losses = [], []
        
        for epoch in range(epochs):
            self.model.train()

            preds = self.model(X_train, mask_train)
            loss = self.criterion(preds, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    preds_test = self.model(X_test, mask_test)
                    test_loss = self.criterion(preds_test, y_test).item()
                    test_losses.append(test_loss)
                print(f"Epoch: {epoch + 1:4d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}")

    def evaluate(self, X_test, y_test, mask):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test, mask).cpu().numpy().reshape(-1, 1)
            true  = y_test.cpu().numpy().reshape(-1, 1)

            preds_inv = self.scaler_y.inverse_transform(preds).flatten()
            true_inv  = self.scaler_y.inverse_transform(true).flatten()

        return preds_inv, true_inv