import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import wandb
import numpy as np

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = torch.relu(out)
        out = self.fc_out(out)
        return out.squeeze(-1)

class LSTM:
    def __init__(self, input_size, device, hidden_size=1024, num_layers=2, epochs=2000, lr=1e-3, weight_decay=0):
        self.model = LSTMRegressor(input_size, hidden_size, num_layers).to(device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

    def fit(self, X, y):
        
        # Initialize wandb for logging
        wandb.init(
            project="LSTM-pendulum",  # Project name in wandb
            config={
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay
            }
        )

        Xtrain, Xval, Ytrain, Yval = train_test_split(X, y, test_size=0.2, random_state=42)

        X = torch.tensor(Xtrain, dtype=torch.float32).to(self.device)
        y = torch.tensor(Ytrain, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(Xval, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(Yval, dtype=torch.float32).to(self.device)
        
        X = X.view(X.shape[0], 1, X.shape[1])
        X_val = X_val.view(X_val.shape[0], 1, X_val.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}', end='\r')
            
            # train
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()

            Ytrain_pred = pred.cpu().detach().numpy()
            rms_radians_train = np.mean((Ytrain_pred - y.cpu().detach().numpy()) ** 2) ** 0.5
            rms_degrees_train = rms_radians_train / (2 * np.pi) * 360
            nrms_train = rms_radians_train / y.cpu().detach().numpy().std() * 100

            wandb.log({
                "train_loss": loss.item(),
                "train_rms_radians": rms_radians_train,
                "train_rms_degrees": rms_degrees_train,
                "train_nrms": nrms_train
            })

            optimizer.step()
            scheduler.step()

            # validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss_value = loss_fn(val_pred, y_val)
                Yval_pred = val_pred.cpu().detach().numpy()
                rms_radians_val = np.mean((Yval_pred - y_val.cpu().detach().numpy()) ** 2) ** 0.5
                rms_degrees_val = rms_radians_val / (2 * np.pi) * 360
                nrms_val = rms_radians_val / y_val.cpu().detach().numpy().std() * 100
                wandb.log({"val_loss": val_loss_value.item(),
                            "val_rms_radians": rms_radians_val,
                            "val_rms_degrees": rms_degrees_val,
                            "val_nrms": nrms_val})
        

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.view(X.shape[0], 1, X.shape[1])
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        return pred.cpu().numpy()