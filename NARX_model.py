from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Network(nn.Module):
    def __init__(self, n_in, n_hidden_nodes):
        super(Network, self).__init__()
        self.lay1 = nn.Linear(n_in, n_hidden_nodes).double()
        self.lay2 = nn.Linear(n_hidden_nodes, n_hidden_nodes).double()
        self.lay3 = nn.Linear(n_hidden_nodes, n_hidden_nodes).double()
        self.output = nn.Linear(n_hidden_nodes, 1).double()

    def forward(self, x):
        x = torch.relu(self.lay1(x))
        x = torch.relu(self.lay2(x))
        x = torch.relu(self.lay3(x))
        y = self.output(x)[:,0]
        return y
    
class NARX_model:
    def __init__(self, na=5, nb=5, n_hidden_nodes=32, epochs=5000):
        self.na = na
        self.nb = nb
        self.n_hidden_nodes = n_hidden_nodes
        self.epochs = epochs
        self.model = None

    def loading_data(self):
        out = np.load("gym-unbalanced-disk-master\\disc-benchmark-files\\training-val-test-data.npz")
        u_train = out['u']
        th_train = out['th']

        data = np.load("gym-unbalanced-disk-master\\disc-benchmark-files\\hidden-test-prediction-submission-file.npz")
        upast_test = data['upast']
        thpast_test = data['thpast']

        return u_train, th_train, upast_test, thpast_test

    def create_IO_data(self, u, y):
        X = []
        Y = []
        
        for k in range(max(self.na,self.nb), len(y)):
            X.append(np.concatenate([u[k-self.nb:k],y[k-self.na:k]]))
            Y.append(y[k])

        return np.array(X), np.array(Y)
    
    def splitting_data(self, Xdata, Ydata, train_frac=0.7, val_frac=0.15):
        N = len(Ydata)

        n_train = int(N*train_frac)
        n_val = int(N*val_frac)

        #train/validation/test --> 70/15/15
        Xtrain = Xdata[:n_train]
        Ytrain = Ydata[:n_train]
        Xval = Xdata[n_train:n_train+n_val]
        Yval = Ydata[n_train:n_train+n_val]
        Xtest = Xdata[n_train+n_val:]
        Ytest = Ydata[n_train+n_val:]

        return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
    
    def training_model(self, Xtrain, Ytrain, Xval, Yval):
        Xtrain, Xval, Ytrain, Yval = [torch.as_tensor(x) for x in [Xtrain, Xval, Ytrain, Yval]]
        train_losses = []
        validation_losses = []

        self.model = Network(Xtrain.shape[1], self.n_hidden_nodes)
        optimizer = torch.optim.AdamW(self.model.parameters())

        for epoch in range(self.epochs):
            train_loss = torch.mean((self.model(Xtrain) - Ytrain)**2)
            with torch.no_grad():
                validation_loss = torch.mean((self.model(Xval) - Yval)**2)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())
            validation_losses.append(validation_loss.item())

            if epoch%100==0:
                print("Epoch:", epoch, "Training loss:", train_loss.item(), "Validation loss:", validation_loss.item())

        print('Train prediction errors:')
        print('RMS:', torch.mean((self.model(Xtrain) - Ytrain)**2)**0.5, 'radians')
        print('RMS:', torch.mean((self.model(Xtrain) - Ytrain)**2)**0.5/(2*torch.pi)*360, 'degrees')
        print('NRMS:', torch.mean((self.model(Xtrain) - Ytrain)**2)**0.5/Ytrain.std()*100, '%')

        return train_losses, validation_losses
    
    def prediction(self, upast_test, thpast_test, filename_npz):
        Xtest = np.concatenate([upast_test[:,15-self.nb:], thpast_test[:,15-self.na:]], axis=1)
        Xtest_tensor = torch.as_tensor(Xtest).double()

        with torch.no_grad():
            Ypredict = self.model(Xtest_tensor)

        np.savez(filename_npz, upast=upast_test, thpast=thpast_test, thnow=Ypredict)
        print(f"Predictions saved to {filename_npz}.")

def main():
    NARX = NARX_model(na=5, nb=5, n_hidden_nodes=32, epochs=5000)

    #Loading data
    u_train, th_train, upast_test, thpast_test = NARX.loading_data()

    #Constructing training data
    Xdata, Ydata = NARX.create_IO_data(u_train, th_train)

    #Splitting dataset
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = NARX.splitting_data(Xdata, Ydata)

    #Training
    NARX.training_model(Xtrain, Ytrain, Xval, Yval)

    #Prediction
    NARX.prediction(upast_test, thpast_test, filename_npz="NARX_prediction.npz")
        
if __name__=="__main__":
    main()