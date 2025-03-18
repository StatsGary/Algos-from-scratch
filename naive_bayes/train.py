# train_models.py

import numpy as np
import torch
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from models.gaussian_naive_bayes import GaussianNaiveBayesNumpy, GaussianNaiveBayesTorch

# Create a directory to save models
SAVE_DIR = 'naive_bayes/saved_models'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

# Split into training and test sets
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler for later use
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.joblib'))

# -------------------------
# Train Gaussian Naive Bayes (NumPy Implementation)
# -------------------------
model_numpy = GaussianNaiveBayesNumpy()
model_numpy.fit(X_train_scaled, y_train)

# Save the trained model parameters
np.savez(os.path.join(SAVE_DIR, 'gaussian_nb_numpy.npz'),
         classes_=model_numpy.classes_,
         class_prior_=model_numpy.class_prior_,
         theta_=model_numpy.theta_,
         sigma_=model_numpy.sigma_)

# -------------------------
# Train Gaussian Naive Bayes (PyTorch Implementation)
# -------------------------
model_torch = GaussianNaiveBayesTorch()
model_torch.fit(X_train_scaled, y_train)

# Save the trained model parameters
torch.save({
    'classes_': model_torch.classes_,
    'class_prior_': model_torch.class_prior_,
    'theta_': model_torch.theta_,
    'sigma_': model_torch.sigma_
}, os.path.join(SAVE_DIR, 'gaussian_nb_torch.pth'))

# -------------------------
# Train Gaussian Naive Bayes (scikit-learn Implementation)
# -------------------------
model_sklearn = GaussianNB()
model_sklearn.fit(X_train_scaled, y_train)

# Save the scikit-learn model using joblib
joblib.dump(model_sklearn, os.path.join(SAVE_DIR, 'gaussian_nb_sklearn.joblib'))
