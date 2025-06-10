import numpy as np
from sklearn.model_selection import train_test_split
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

#### Select device for training ####
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#### Define hyperparameters ####
num_inducing = 200  # Number of inducing points for sparse GP
training_iter = 50  # Number of training iterations
batch_size = 256    # Batch size for training


#### Load training and test data ####
out = np.load('training-val-test-data.npz')
th_train = out['th'] #th[0],th[1],th[2],th[3],...
u_train = out['u'] #u[0],u[1],u[2],u[3],...

data = np.load('hidden-test-simulation-submission-file.npz')
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
Xtrain_torch = torch.tensor(Xtrain, dtype=torch.float32).to(device)
Ytrain_torch = torch.tensor(Ytrain, dtype=torch.float32).to(device)

train_dataset = TensorDataset(Xtrain_torch, Ytrain_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

inducing_points = Xtrain_torch[:num_inducing]
model = SparseGPModel(inducing_points).to(device)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Xtrain_torch.size(0))

for i in range(training_iter):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}')


#### Save the trained model ####
torch.save(model.state_dict(), 'gp.pth')


#### Train prediction results ####
model.eval()
likelihood.eval()

def gp_predict(x):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x)).mean
    return pred.cpu().numpy()

Ytrain_pred = gp_predict(Xtrain)
print('train prediction errors:')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5,'radians')
print('RMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Ytrain_pred-Ytrain)**2)**0.5/Ytrain.std()*100,'%')

#### Validation prediction results ####
Yval_pred = gp_predict(Xval)
print('validation prediction errors:')
print('RMS:', np.mean((Yval_pred-Yval)**2)**0.5,'radians')
print('RMS:', np.mean((Yval_pred-Yval)**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((Yval_pred-Yval)**2)**0.5/Yval.std()*100,'%')


def simulation_IO_model(f, ulist, ylist, skip=50):
    upast = ulist[skip-na:skip].tolist() #good initialization
    ypast = ylist[skip-nb:skip].tolist()
    Y = ylist[:skip].tolist()
    for u in ulist[skip:]:
        x = np.concatenate([upast, ypast], axis=0)
        ypred = f(x)
        Y.append(ypred)
        upast.append(u)
        upast.pop(0)
        ypast.append(ypred)
        ypast.pop(0)
    return np.array(Y)

skip = max(na,nb)
th_train_sim = simulation_IO_model(lambda x: gp_predict(x[None,:])[0], u_train, th_train, skip=skip)
print('train simulation errors:')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5,'radians')
print('RMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/(2*np.pi)*360,'degrees')
print('NRMS:', np.mean((th_train_sim[skip:]-th_train[skip:])**2)**0.5/th_train.std()*100,'%')


#### Hidden test simulation ####
skip = 50
th_hidden_test_sim = simulation_IO_model(lambda x: gp_predict(x[None,:])[0], u_hidden_test, th_hidden_test, skip=skip)

assert len(th_hidden_test_sim)==len(th_hidden_test)
np.savez('hidden-test-simulation-gp-submission-file.npz', th=th_hidden_test_sim, u=u_hidden_test)

# FOR 200 inducing points:
#     train prediction errors:
#     RMS: 0.005745671092858161 radians
#     RMS: 0.329202704091092 degrees
#     NRMS: 1.2030586487696358 %
#     validation prediction errors:
#     RMS: 0.005729208674204482 radians
#     RMS: 0.3282594769816587 degrees
#     NRMS: 1.177790061002249 %
#     train simulation errors:
#     RMS: 0.12031842411052436 radians
#     RMS: 6.893737899198132 degrees
#     NRMS: 25.100992356854036 %
    
# FOR 750 inducing points:
#     train prediction errors:
#     RMS: 0.005170851016641181 radians
#     RMS: 0.2962679397444707 degrees
#     NRMS: 1.082699816354247 %
#     validation prediction errors:
#     RMS: 0.005226653819493596 radians
#     RMS: 0.29946520483291467 degrees
#     NRMS: 1.0744766460708057 %
#     train simulation errors:
#     RMS: 0.07232235425302494 radians
#     RMS: 4.143765663148349 degrees
#     NRMS: 15.087987353186064 %