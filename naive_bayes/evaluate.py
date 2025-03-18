# evaluate_models.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.gaussian_naive_bayes import GaussianNaiveBayesNumpy, GaussianNaiveBayesTorch
from sklearn.naive_bayes import GaussianNB
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Directory where models are saved
SAVE_DIR = 'naive_bayes/saved_models'

# Load the scaler used during training
scaler = joblib.load(os.path.join(SAVE_DIR, 'scaler.joblib'))

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

# Standardize features using the loaded scaler
X_scaled = scaler.transform(X)

# Split into training and test sets (ensure same random_state)
_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Load and Evaluate Gaussian Naive Bayes (NumPy Implementation)
# -------------------------
# Load the saved parameters
numpy_params = np.load(os.path.join(SAVE_DIR, 'gaussian_nb_numpy.npz'), allow_pickle=True)
model_numpy = GaussianNaiveBayesNumpy()
model_numpy.classes_ = numpy_params['classes_']
model_numpy.class_prior_ = numpy_params['class_prior_']
model_numpy.theta_ = numpy_params['theta_']
model_numpy.sigma_ = numpy_params['sigma_']

# Evaluate the model
y_pred_numpy = model_numpy.predict(X_test)
y_prob_numpy = model_numpy.predict_proba(X_test)[:, 1]  # Probability of positive class

# -------------------------
# Load and Evaluate Gaussian Naive Bayes (PyTorch Implementation)
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_params = torch.load(os.path.join(SAVE_DIR, 'gaussian_nb_torch.pth'), map_location=device)
model_torch = GaussianNaiveBayesTorch()
model_torch.classes_ = torch_params['classes_']
model_torch.class_prior_ = torch_params['class_prior_']
model_torch.theta_ = torch_params['theta_']
model_torch.sigma_ = torch_params['sigma_']

# Evaluate the model
y_pred_torch = model_torch.predict(X_test)
y_prob_torch = model_torch.predict_proba(X_test)[:, 1]

# -------------------------
# Load and Evaluate Gaussian Naive Bayes (scikit-learn Implementation)
# -------------------------
model_sklearn = joblib.load(os.path.join(SAVE_DIR, 'gaussian_nb_sklearn.joblib'))

# Evaluate the model
y_pred_sklearn = model_sklearn.predict(X_test)
y_prob_sklearn = model_sklearn.predict_proba(X_test)[:, 1]

# -------------------------
# Prepare a Results Dictionary
# -------------------------
results = {}
# Evaluate and store metrics for each model

# Gaussian Naive Bayes (NumPy Implementation)
accuracy_numpy = accuracy_score(y_test, y_pred_numpy)
conf_matrix_numpy = confusion_matrix(y_test, y_pred_numpy)
fpr_numpy, tpr_numpy, _ = roc_curve(y_test, y_prob_numpy)
roc_auc_numpy = auc(fpr_numpy, tpr_numpy)

results['NumPy'] = {
    'accuracy': accuracy_numpy,
    'confusion_matrix': conf_matrix_numpy,
    'fpr': fpr_numpy,
    'tpr': tpr_numpy,
    'roc_auc': roc_auc_numpy
}

# Gaussian Naive Bayes (PyTorch Implementation)
accuracy_torch = accuracy_score(y_test, y_pred_torch)
conf_matrix_torch = confusion_matrix(y_test, y_pred_torch)
fpr_torch, tpr_torch, _ = roc_curve(y_test, y_prob_torch)
roc_auc_torch = auc(fpr_torch, tpr_torch)

results['PyTorch'] = {
    'accuracy': accuracy_torch,
    'confusion_matrix': conf_matrix_torch,
    'fpr': fpr_torch,
    'tpr': tpr_torch,
    'roc_auc': roc_auc_torch
}

# Gaussian Naive Bayes (scikit-learn Implementation)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_prob_sklearn)
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
plt.title('ROC Curves for Gaussian Naive Bayes Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'roc_curves.png'))

# -------------------------
# Save Confusion Matrices as Images
# -------------------------
for key in results:
    cm = results[key]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {key}")
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_{key}.png"))
    plt.close()
