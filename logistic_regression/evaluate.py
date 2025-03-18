import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.logistic_reg import LogisticRegressionNumpy
from sklearn.linear_model import LogisticRegression
import joblib  # For loading scikit-learn model
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

LOAD_PATH = 'logistic_regression/saved_models'

# Load the scaler used during training
scaler = joblib.load(os.path.join(LOAD_PATH, 'scaler.joblib'))
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

# Standardise features using the loaded scaler
X_scaled = scaler.transform(X)

# Split into training and test sets (ensure same random_state)
_, X_test_np, _, y_test_np = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# Load and Evaluate LogisticRegressionNumpy model
# -------------------------
# Load the saved parameters
numpy_params = np.load(os.path.join(LOAD_PATH, 'logistic_regression_numpy.npz'), allow_pickle=True)
model_numpy = LogisticRegressionNumpy(
    penalty=numpy_params['penalty'].item(),
    C=numpy_params['C'].item()
)
# Assign the loaded parameters
model_numpy.theta = numpy_params['theta']
model_numpy.bias = numpy_params['bias'].item()
model_numpy.classes_ = numpy_params['classes_']

# Evaluate the model
y_pred_numpy = model_numpy.predict(X_test_np)
y_prob_numpy = model_numpy.predict_proba(X_test_np)[:, 1]  # Probability of positive class

# -------------------------
# Load and Evaluate scikit-learn LogisticRegression model
# -------------------------
model_sklearn = joblib.load(os.path.join(LOAD_PATH, 'log_regression_sklearn.joblib'))
# Evaluate the model
y_pred_sklearn = model_sklearn.predict(X_test_np)
y_prob_sklearn = model_sklearn.predict_proba(X_test_np)[:, 1]

# -------------------------
# Prepare a Results Dictionary
# -------------------------
results = {}
# Evaluate and store metrics for each model

# LogisticRegressionNumpy
accuracy_numpy = accuracy_score(y_test_np, y_pred_numpy)
conf_matrix_numpy = confusion_matrix(y_test_np, y_pred_numpy)
fpr_numpy, tpr_numpy, _ = roc_curve(y_test_np, y_prob_numpy)
roc_auc_numpy = auc(fpr_numpy, tpr_numpy)

results['NumPy'] = {
    'accuracy': accuracy_numpy,
    'confusion_matrix': conf_matrix_numpy,
    'fpr': fpr_numpy,
    'tpr': tpr_numpy,
    'roc_auc': roc_auc_numpy
}

# scikit-learn LogisticRegression
accuracy_sklearn = accuracy_score(y_test_np, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test_np, y_pred_sklearn)
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test_np, y_prob_sklearn)
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
GRAPHIC_PATH = 'logistic_regression/charts'
os.makedirs(GRAPHIC_PATH, exist_ok=True)


plt.figure(figsize=(8, 6))
plt.plot(results['NumPy']['fpr'], results['NumPy']['tpr'],
         label=f"NumPy (AUC = {results['NumPy']['roc_auc']:.2f})")
plt.plot(results['scikit-learn']['fpr'], results['scikit-learn']['tpr'],
         label=f"scikit-learn (AUC = {results['scikit-learn']['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(GRAPHIC_PATH, 'roc_curves.png'))  # Save ROC curve as image

# -------------------------
# Save Confusion Matrices as Images
# -------------------------
for key in results:
    cm = results[key]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {key}")
    plt.savefig(os.path.join(GRAPHIC_PATH, f"confusion_matrix_{key}.png"))  # Save confusion matrix as image
    plt.close()  # Close the plot to avoid overlap in subsequent iterations
