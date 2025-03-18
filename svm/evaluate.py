import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.svm_c import SVM_C_Numpy
from models.svm_c import SVM_C_PyTorch
from sklearn.linear_model import SGDClassifier  # Updated to match training script
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Directory where models are saved
SAVE_DIR = 'svm/saved_models'

# Load the scaler used during training
scaler = joblib.load(os.path.join(SAVE_DIR, 'scaler.joblib'))

# -------------------------
# Load the breast cancer dataset
# -------------------------
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 or 1)

# Convert labels from {0, 1} to {-1, 1}
y = np.where(y == 0, -1, 1)

# -------------------------
# Standardize features using the loaded scaler
# -------------------------
X_scaled = scaler.transform(X)

# -------------------------
# Split into training and test sets (ensure same random_state)
# -------------------------
_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Load and Evaluate SVM (NumPy Implementation)
# -------------------------
# Load the saved parameters
numpy_params = np.load(os.path.join(SAVE_DIR, 'svm_numpy.npz'), allow_pickle=True)
model_numpy = SVM_C_Numpy()
model_numpy.w = numpy_params['w']
model_numpy.b = numpy_params['b']

# Evaluate the model
y_pred_numpy = model_numpy.predict(X_test)
# Since SVM does not output probabilities, we use decision function values for ROC curve
decision_values_numpy = np.dot(X_test, model_numpy.w) + model_numpy.b

# -------------------------
# Load and Evaluate SVM (PyTorch Implementation)
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_params = torch.load(os.path.join(SAVE_DIR, 'svm_torch.pth'), map_location=device)
model_torch = SVM_C_PyTorch()
model_torch.w = torch_params['w']
model_torch.b = torch_params['b']

# Evaluate the model
y_pred_torch = model_torch.predict(X_test)
# Use decision function values for ROC curve
with torch.no_grad():
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    decision_values_torch = X_test_torch.matmul(model_torch.w) + model_torch.b
    decision_values_torch = decision_values_torch.cpu().numpy()

# -------------------------
# Load and Evaluate SVM (scikit-learn Implementation)
# -------------------------
model_sklearn = joblib.load(os.path.join(SAVE_DIR, 'svm_sklearn.joblib'))

# Evaluate the model
y_pred_sklearn = model_sklearn.predict(X_test)
# Get decision function values for ROC curve
decision_values_sklearn = model_sklearn.decision_function(X_test)

# -------------------------
# Prepare a Results Dictionary
# -------------------------
results = {}
# Evaluate and store metrics for each model

# SVM (NumPy Implementation)
accuracy_numpy = accuracy_score(y_test, y_pred_numpy)
conf_matrix_numpy = confusion_matrix(y_test, y_pred_numpy)
fpr_numpy, tpr_numpy, _ = roc_curve(y_test, decision_values_numpy)
roc_auc_numpy = auc(fpr_numpy, tpr_numpy)

results['NumPy'] = {
    'accuracy': accuracy_numpy,
    'confusion_matrix': conf_matrix_numpy,
    'fpr': fpr_numpy,
    'tpr': tpr_numpy,
    'roc_auc': roc_auc_numpy
}

# SVM (PyTorch Implementation)
accuracy_torch = accuracy_score(y_test, y_pred_torch)
conf_matrix_torch = confusion_matrix(y_test, y_pred_torch)
fpr_torch, tpr_torch, _ = roc_curve(y_test, decision_values_torch)
roc_auc_torch = auc(fpr_torch, tpr_torch)

results['PyTorch'] = {
    'accuracy': accuracy_torch,
    'confusion_matrix': conf_matrix_torch,
    'fpr': fpr_torch,
    'tpr': tpr_torch,
    'roc_auc': roc_auc_torch
}

# SVM (scikit-learn Implementation)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, decision_values_sklearn)
roc_auc_sklearn = auc(fpr_sklearn, tpr_sklearn)

results['scikit-learn'] = {
    'accuracy': accuracy_sklearn,
    'confusion_matrix': conf_matrix_sklearn,
    'fpr': fpr_sklearn,
    'tpr': tpr_sklearn,
    'roc_auc': roc_auc_sklearn
}

# -------------------------
# Print Accuracies
# -------------------------
print("Model Accuracies:")
for key in results:
    print(f"{key}: {results[key]['accuracy'] * 100:.2f}%")

# -------------------------
# Print Weights and Biases
# -------------------------
print("\nModel Weights and Biases:")

# NumPy Model
print("\nNumPy Model:")
print(f"Weights (w):\n{model_numpy.w}")
print(f"Bias (b): {model_numpy.b}")

# PyTorch Model
print("\nPyTorch Model:")
print(f"Weights (w):\n{model_torch.w.cpu().numpy()}")
print(f"Bias (b): {model_torch.b.item()}")

# scikit-learn Model
print("\nscikit-learn Model:")
print(f"Weights (w):\n{model_sklearn.coef_.flatten()}")
print(f"Bias (b): {model_sklearn.intercept_[0]}")

# Create directory for evaluation artifacts
CHART_PATH = "svm/eval_artifacts"
os.makedirs(CHART_PATH, exist_ok=True)

# -------------------------
# Save ROC Curves as Image
# -------------------------
plt.figure(figsize=(8, 6))
plt.plot(results['NumPy']['fpr'], results['NumPy']['tpr'],
         label=f"NumPy (AUC = {results['NumPy']['roc_auc']:.2f})")
plt.plot(results['PyTorch']['fpr'], results['PyTorch']['tpr'],
         label=f"PyTorch (AUC = {results['PyTorch']['roc_auc']:.2f})")
plt.plot(results['scikit-learn']['fpr'], results['scikit-learn']['tpr'],
         label=f"scikit-learn (AUC = {results['scikit-learn']['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for SVM Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(CHART_PATH, 'roc_curves.png'))

# -------------------------
# Save Confusion Matrices as Images
# -------------------------
for key in results:
    cm = results[key]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {key}")
    plt.savefig(os.path.join(CHART_PATH, f"confusion_matrix_{key}.png"))
    plt.close()
