import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, scaler_y):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler_y = scaler_y

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            self.model.train()

            preds = self.model(X_train)
            loss = self.criterion(preds, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test).cpu().numpy().reshape(-1, 1)
            true  = y_test.cpu().numpy().reshape(-1, 1)
            
            preds_inv = self.scaler_y.inverse_transform(preds).flatten()
            true_inv  = self.scaler_y.inverse_transform(true).flatten()

        return preds_inv, true_inv