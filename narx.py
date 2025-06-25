import torch.nn as nn
import torch


class NARX_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lay1 = nn.Linear(input_size, hidden_size)
        self.lay2 = nn.Linear(hidden_size, hidden_size)
        self.lay3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
       
    def forward(self, x):
        x = torch.relu(self.lay1(x))
        x = torch.relu(self.lay2(x))
        x = torch.relu(self.lay3(x))
        x = self.output(x)
        return x

class NARX(nn.Module):
    def __init__(self, input_size, device, hidden_size=1024, epochs=2000, lr=1e-3):
        super().__init__()
        self.model = NARX_Regressor(input_size, hidden_size).to(device)
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
            pred = pred.view(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.view(X.shape[0], 1, X.shape[1])
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        pred = pred.view(-1)
        return pred.cpu().numpy()