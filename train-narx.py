import numpy as np
from sklearn.model_selection import train_test_split
import torch

#### Select device for training ####
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#### Define hyperparameters ####
EPOCHS = 4000
LR = 1e-3
HIDDEN_SIZE = 1024



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
#[TODO]
from narx import NARX
model = NARX(input_size=Xtrain.shape[1], hidden_size=HIDDEN_SIZE, device=device, epochs=EPOCHS).to(device)
model.fit(Xtrain, Ytrain)


#### Save the trained model ####
torch.save(model.state_dict(), 'narx.pth')


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
th_train_sim = simulation_IO_model(lambda x: model.predict(x[None,:])[0], u_train, th_train, skip=skip)
print('train simulation errors:')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5,'radians')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/th_train.std()*100,'%')


#### Hidden test simulation ####
skip = 50
th_hidden_test_sim = simulation_IO_model(lambda x: model.predict(x[None,:])[0], u_hidden_test, th_hidden_test, skip=skip)

assert len(th_hidden_test_sim)==len(th_hidden_test)
np.savez('hidden-test-simulation-narx-submission-file.npz', th=th_hidden_test_sim, u=u_hidden_test)


#train prediction errors:
#RMS: 0.002681949987235875 radians
#RMS: 0.15366441513378065 degrees
#NRMS: 0.5615607081516221 %
#validation prediction errors:
#RMS: 0.0032344589961870785 radians
#RMS: 0.18532084948964042 degrees
#NRMS: 0.664928417702889 %
#train simulation errors:
#RMS: 0.015171805336877184 radians
#RMS: 0.8692804133971208 degrees
#NRMS: 3.165162547764122 %