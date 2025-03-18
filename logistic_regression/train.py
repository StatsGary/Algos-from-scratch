import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from models.logistic_reg import LogisticRegressionNumpy, LogisticRegressionTorch
from sklearn.linear_model import LogisticRegression

SAVE_PATH = 'logistic_regression/saved_models'

data = load_breast_cancer()
X = data.data
y = data.target
print(y)

# Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, 
    y, 
    test_size=0.2, 
    random_state=42)

# For PyTorch, convert data to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_torch = torch.tensor(X_train_np, dtype=torch.float32, device=device)
y_train_torch = torch.tensor(y_train_np, dtype=torch.float32, device=device)

# Create a directory to save models
os.makedirs(SAVE_PATH, exist_ok=True)


# -------------------------------------------------------------------------
# Train LogisticRegressionNumpy model
# -------------------------------------------------------------------------

model_numpy = LogisticRegressionNumpy(
    penalty='l2',
    C=1.0, 
    learning_rate=0.1,
    max_iter=1000, 
    tol=1e-4, 
    random_state=42
)

# Fit the nump model 
model_numpy.fit(X_train_np, y_train_np)
print('Trained Logistic Regression Numpy model')

# Save the NumPy model parameters
numpy_model_params = {
    'theta': model_numpy.theta,
    'bias': model_numpy.bias,
    'classes_': model_numpy.classes_,
    'penalty': model_numpy.penalty,
    'C': model_numpy.C}
np.savez(os.path.join(SAVE_PATH,'logistic_regression_numpy.npz'), **numpy_model_params)
print("Saved LogisticRegressionNumpy model parameters.")

# Save the scaler used for preprocessing
joblib.dump(scaler, os.path.join(SAVE_PATH, 'scaler.joblib'))

# -------------------------------------------------------------------------
# Train LogisticRegressionTorch model
# -------------------------------------------------------------------------

model_torch = LogisticRegressionTorch(
    penalty='l2',
    C=1.0, 
    learning_rate=0.1,
    max_iter=1000, 
    tol=1e-4, 
    random_state=42
)

model_torch.fit(X_train_torch, y_train_torch)
print('Trained LogisticRegressionTorch model')

# Save the models

torch.save({
    'theta': model_torch.theta,
    'bias': model_torch.bias,
    'classes_': model_torch.classes_,
    'penalty': model_torch.penalty,
    'C': model_torch.C,
},
os.path.join(SAVE_PATH, 'logistic_regression_torch.pth'))
print('Saved Logistic Regression Torch model parameters')

# -------------------------------------------------------------------------
# Train Sci-kit learn base Logistic Regression model
# -------------------------------------------------------------------------

model_sklearn = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000, 
    random_state=42
)

model_sklearn.fit(X_train_np, y_train_np)
print('Trained scikit-learn Logistic Regression model')

joblib.dump(model_sklearn, filename=os.path.join(SAVE_PATH, 'log_regression_sklearn.joblib'))
print('Saved Sci-kit learn Logistic regression model')