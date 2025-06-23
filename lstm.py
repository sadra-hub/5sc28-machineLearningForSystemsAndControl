import torch.nn as nn
import torch




class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Match NARX's 3-layer MLP after LSTM output
        self.lay1 = nn.Linear(hidden_size, hidden_size)
        self.lay2 = nn.Linear(hidden_size, hidden_size)
        self.lay3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Only use the final output like the NARX (no time-pooling or attention)
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        out = torch.relu(self.lay1(out))
        out = torch.relu(self.lay2(out))
        out = torch.relu(self.lay3(out))
        out = self.output(out)
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