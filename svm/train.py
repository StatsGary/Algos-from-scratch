# train_models.py

import numpy as np
import torch
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier  # Use SGDClassifier
from models.svm_c import SVM_C_Numpy
from models.svm_c import SVM_C_PyTorch

# Create a directory to save models
SAVE_DIR = 'svm/saved_models'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -------------------------
# Load the breast cancer dataset
# -------------------------
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 or 1)

# Convert labels from {0, 1} to {-1, 1}
y = np.where(y == 0, -1, 1)

# -------------------------
# Split into training and test sets
# -------------------------
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Standardize features (mean=0, std=1)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler for later use
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.joblib'))

# -------------------------
# Set hyperparameters
# -------------------------
C = 1.0  # Regularization parameter
n_iters = 5000
learning_rate = 0.001

# For SGDClassifier, we need to set alpha (regularization strength)
# alpha = 1 / (n_samples * C)
n_samples = X_train_scaled.shape[0]
alpha = 1 / (n_samples * C)

# -------------------------
# Train SVM (NumPy Implementation)
# -------------------------
print("Training SVM using NumPy implementation...")
model_numpy = SVM_C_Numpy(C=C, n_iters=n_iters, learning_rate=learning_rate)
model_numpy.fit(X_train_scaled, y_train)

# Save the trained model parameters
np.savez(os.path.join(SAVE_DIR, 'svm_numpy.npz'),
         w=model_numpy.w,
         b=model_numpy.b)

# -------------------------
# Train SVM (PyTorch Implementation)
# -------------------------
print("Training SVM using PyTorch implementation...")
model_torch = SVM_C_PyTorch(C=C, n_iters=n_iters, learning_rate=learning_rate)
model_torch.fit(X_train_scaled, y_train)

# Save the trained model parameters
torch.save({
    'w': model_torch.w,
    'b': model_torch.b
}, os.path.join(SAVE_DIR, 'svm_torch.pth'))

# -------------------------
# Train SVM (scikit-learn Implementation with SGDClassifier)
# -------------------------
print("Training SVM using scikit-learn SGDClassifier...")
model_sklearn = SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=alpha,
    max_iter=n_iters,
    learning_rate='constant',
    eta0=learning_rate,
    tol=None,
    shuffle=False,  # To mimic batch gradient descent
    random_state=42
)
model_sklearn.fit(X_train_scaled, y_train)

# Save the scikit-learn model using joblib
joblib.dump(model_sklearn, os.path.join(SAVE_DIR, 'svm_sklearn.joblib'))
