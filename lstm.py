import torch.nn as nn
import torch


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

class LSTM(nn.Module):
    def __init__(self, input_size, device, hidden_size=1024, num_layers=2, epochs=2000, lr=1e-3):
        super().__init__()
        self.model = LSTMRegressor(input_size, hidden_size, num_layers).to(device)
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        X = X.view(X.shape[0], 1, X.shape[1])
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}', end='\r')
            
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.view(X.shape[0], 1, X.shape[1])
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        return pred.cpu().numpy()