# train_decision_tree.py

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from models.decision_tree import DecisionTreeClassifier, DecisionTreeClassifierPyTorch
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

# Define the path where models will be saved
SAVE_PATH = 'decision_tree/saved_models'

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Feature matrix
y = data.target  # Target vector (binary labels)
print("Loaded breast cancer dataset.")

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized.")

# Split into training and test sets
print("Splitting data into training and test sets...")
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
print(f"Training samples: {X_train_np.shape[0]}, Test samples: {X_test_np.shape[0]}.")

# For PyTorch, convert data to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
X_train_torch = torch.tensor(X_train_np, dtype=torch.float32, device=device)
y_train_torch = torch.tensor(y_train_np, dtype=torch.long, device=device)  # Use long for classification
print("Converted training data to PyTorch tensors.")

# Create a directory to save models
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Models will be saved to '{SAVE_PATH}'.")

# -------------------------------------------------------------------------
# Train DecisionTreeClassifier (NumPy Implementation)
# -------------------------------------------------------------------------
print("\nTraining NumPy Decision Tree Classifier...")
model_numpy = DecisionTreeClassifier(
    criterion='gini',  # You can switch to 'entropy' if desired
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the NumPy model
model_numpy.fit(X_train_np, y_train_np)
print('Trained NumPy Decision Tree Classifier.')

# Save the NumPy model parameters using joblib
numpy_model_params = {
    'tree': model_numpy.tree,
    'criterion': model_numpy.criterion,
    'max_depth': model_numpy.max_depth,
    'min_samples_split': model_numpy.min_samples_split,
    'min_samples_leaf': model_numpy.min_samples_leaf,
    'random_state': model_numpy.random_state
}
joblib.dump(numpy_model_params, os.path.join(SAVE_PATH, 'decision_tree_numpy.joblib'))
print("Saved NumPy Decision Tree model parameters to 'decision_tree_numpy.joblib'.")

# -------------------------------------------------------------------------
# Train DecisionTreeClassifierPyTorch (PyTorch Implementation)
# -------------------------------------------------------------------------
print("\nTraining PyTorch Decision Tree Classifier...")
model_torch = DecisionTreeClassifierPyTorch(
    criterion='gini',  # You can switch to 'entropy' if desired
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the PyTorch model
model_torch.fit(X_train_torch, y_train_torch)
print('Trained PyTorch Decision Tree Classifier.')

# Save the PyTorch model parameters using torch.save
torch_model_params = {
    'tree': model_torch.tree,
    'criterion': model_torch.criterion,
    'max_depth': model_torch.max_depth,
    'min_samples_split': model_torch.min_samples_split,
    'min_samples_leaf': model_torch.min_samples_leaf,
    'random_state': model_torch.random_state
}
torch.save(torch_model_params, os.path.join(SAVE_PATH, 'decision_tree_torch.pth'))
print("Saved PyTorch Decision Tree model parameters to 'decision_tree_torch.pth'.")

# -------------------------------------------------------------------------
# Train Scikit-learn DecisionTreeClassifier (Scikit-learn Implementation)
# -------------------------------------------------------------------------
print("\nTraining Scikit-learn Decision Tree Classifier...")
model_sklearn = SklearnDecisionTreeClassifier(
    criterion='gini',  # You can switch to 'entropy' if desired
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the Scikit-learn model
model_sklearn.fit(X_train_np, y_train_np)
print('Trained Scikit-learn Decision Tree Classifier.')

# Save the Scikit-learn model using joblib
joblib.dump(model_sklearn, filename=os.path.join(SAVE_PATH, 'decision_tree_sklearn.joblib'))
print("Saved Scikit-learn Decision Tree model to 'decision_tree_sklearn.joblib'.")

# -------------------------------------------------------------------------
# Save the scaler used for preprocessing
# -------------------------------------------------------------------------
print("\nSaving the scaler used for preprocessing...")
joblib.dump(scaler, os.path.join(SAVE_PATH, 'scaler.joblib'))
print("Saved scaler to 'scaler.joblib'.")

print("\nAll models trained and saved successfully.")
