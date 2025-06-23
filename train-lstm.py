import numpy as np
from sklearn.model_selection import train_test_split
import torch

#### Select device for training ####
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#### Define hyperparameters ####
EPOCHS = 1000
LR = 0.001
HIDDEN_SIZE = 1024
NUM_LAYERS = 1


#### Load training and test data ####
out = np.load('./NPZ/training-val-test-data.npz')
th_train = out['th'] #th[0],th[1],th[2],th[3],...
u_train = out['u'] #u[0],u[1],u[2],u[3],...

data = np.load('./NPZ/hidden-test-simulation-submission-file.npz')
u_hidden_test = data['u']
th_hidden_test = data['th'] #only the first 50 values are filled the rest are zeros



#### Create input-output data for training ####
def create_IO_data(u,y,na,nb):
    X = []
    Y = []
    for k in range(max(na,nb), len(y)):
        X.append(np.concatenate([u[k-nb:k],y[k-na:k]]))
        Y.append(y[k])
    return np.array(X), np.array(Y)

na = 5
nb = 5
X, y = create_IO_data(u_train, th_train, na, nb)

#### Split data into training and validation sets ####
Xtrain, Xval, Ytrain, Yval = train_test_split(X, y, test_size=0.2, random_state=42)


#### Define and train the model ####
from lstm import LSTM
from sklearn.preprocessing import StandardScaler
model = LSTM(input_size=Xtrain.shape[1], device=device, epochs=EPOCHS, lr=LR, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

# Normalize the data

scaler_X = StandardScaler()
scaler_y = StandardScaler()

Xtrain = scaler_X.fit_transform(Xtrain)
Xval = scaler_X.transform(Xval)
Ytrain = scaler_y.fit_transform(Ytrain.reshape(-1, 1)).flatten()
Yval = scaler_y.transform(Yval.reshape(-1, 1)).flatten()

# Update model to use scalers for inverse transform in prediction if needed
model.scaler_X = scaler_X
model.scaler_y = scaler_y

model.fit(Xtrain, Ytrain)

#### Save the trained model ####
torch.save(model.state_dict(), 'lstm.pth')


#### Train prediction results ####
Ytrain_pred = model.predict(Xtrain)
print('train prediction errors:')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5,'radians')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/Ytrain.std()*100,'%')

#### Validation prediction results ####
Yval_pred = model.predict(Xval)
print('validation prediction errors:')
print('RMS:', np.mean((Yval_pred-Yval)**2)**0.5,'radians')
print('RMS:', np.mean((Yval_pred-Yval)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Yval_pred-Yval)**2)**0.5/Yval.std()*100,'%')



def autoregressive_predict(model, x_raw):
    x_scaled = model.scaler_X.transform(x_raw.reshape(1, -1))  # reshape to (1, input_size)
    y_scaled = model.predict(x_scaled)[0]
    y = model.scaler_y.inverse_transform([[y_scaled]])[0, 0]
    return y

def simulation_IO_model(f, ulist, ylist, skip=50):

    upast = ulist[skip-na:skip].tolist() #good initialization
    ypast = ylist[skip-nb:skip].tolist()
    Y = ylist[:skip].tolist()
    for u in ulist[skip:]:
        x = np.concatenate([upast,ypast],axis=0)
        ypred = f(x)
        Y.append(ypred)
        upast.append(u)
        upast.pop(0)
        ypast.append(ypred)
        ypast.pop(0)
    return np.array(Y)

skip = max(na,nb)
th_train_sim = simulation_IO_model(lambda x: autoregressive_predict(model, x[None,:]), u_train, th_train, skip=skip)
print('train simulation errors:')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5,'radians')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/th_train.std()*100,'%')


#### Hidden test simulation ####
skip = 50
th_hidden_test_sim = simulation_IO_model(lambda x: model.predict(x[None,:])[0], u_hidden_test, th_hidden_test, skip=skip)

assert len(th_hidden_test_sim)==len(th_hidden_test)
np.savez('hidden-test-simulation-lstm-submission-file.npz', th=th_hidden_test_sim, u=u_hidden_test)


# train prediction errors:
# RMS: 0.00652481785384419 radians
# RMS: 0.3738445251168797 degrees
# NRMS: 0.652481785384419 %
# validation prediction errors:
# RMS: 0.006802541407328163 radians
# RMS: 0.3897569126028872 degrees
# NRMS: 0.6678799411738088 %
# train simulation errors:
# RMS: 0.02112926229904037 radians
# RMS: 1.2106175539599 degrees
# NRMS: 4.40801527608925 %