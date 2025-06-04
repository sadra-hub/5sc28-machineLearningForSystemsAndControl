import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import GPy
import numpy as np
np.random.seed(101)

# === Load dataset (no header) ===
DATA_PATH = "gym-unbalanced-disk-master/disc-benchmark-files/training-val-test-data.csv"
df = pd.read_csv(DATA_PATH, comment="#", header=None, names=["u", "th"])

# === Drop missing data ===
df.dropna(inplace=True)

# === Build NARX features ===
df['th_t'] = df['th']
df['th_tm1'] = df['th'].shift(1)
df['u_t'] = df['u']
df['u_tm1'] = df['u'].shift(1)
df['th_tp1'] = df['th'].shift(-1)  # target

df.dropna(inplace=True)

# === Inputs (X) and target (y) as Arrays NOT Dataframes ===
X = df[['th_t', 'th_tm1', 'u_t', 'u_tm1']].values
y = df['th_tp1'].values.reshape(-1,1)

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# === Split dataset into Train dataset and Test+Inducing (TI) Dataset ===
#X_train, X_TI, y_train, y_TI = train_test_split(X, y, test_size=0.3, random_state=20)
#print(y_train.shape)

# === Split Test+Inducing Dataset into Test dataset and Inducing points dataset ===
#X_test, X_indu, y_test, y_indu = train_test_split(X_TI, y_TI, test_size=0.3, random_state=20)
#print(X_test.shape)

# Define inducing points (subset of X_test)
#Z = X_train[np.random.choice(X_train.shape[0], 190 , replace=False)]  # Select 20 inducing points

# === Select 30 inducing points uniformly from the feature space X ===
mins = X_train.min(axis=0)
maxs = X_train.max(axis=0)
Z = np.random.uniform(low=mins,high=maxs,size=(30,X.shape[1]))    # Select 30 inducing points

# Define a kernel and create a Sparse GP model
kernel = GPy.kern.RBF(input_dim=4)
model = GPy.models.SparseGPRegression(X_train , y_train , kernel , Z=Z)

# Set the noise in datase variance
model.likelihood.variance = 0.01  # Lower values assume less noise

# Optimize the model
model.optimize('bfgs')

# Plot the results
#model.plot()

print (model)
#print(y_train)

# fixing the random chosen inducing points
model.inducing_inputs.fix()

# Predicting on Sparse GP model
y_test_pred , y_test_var = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Test RMSE:", rmse)

# Plot predictions vs true values and error bars (first 100 points)
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="True")
plt.plot(y_test_pred[:100], label="Predicted")
plt.fill_between(range(100),
                 (y_test_pred[:100] - 2 * np.sqrt(y_test_var[:100])).flatten(),
                 (y_test_pred[:100] + 2 * np.sqrt(y_test_var[:100])).flatten(),
                 color="gray", alpha=0.2, label="±2 std")
plt.errorbar(range(100), (y_test_pred[:100].flatten()), yerr=2*y_test_var[:100].flatten(),fmt='.r')
plt.legend()
plt.title("Sparse GP Prediction on Test Set")
plt.xlabel("Sample")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print kernel info ===
print("Trained Sparse GP kernel:", model.rbf)
