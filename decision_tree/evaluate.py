# evaluate_decision_tree.py

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from models.decision_tree import DecisionTreeClassifier, DecisionTreeClassifierPyTorch
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

def main():
    # Define the path where models are saved
    SAVE_PATH = 'decision_tree/saved_models'

    # Check if SAVE_PATH exists
    if not os.path.exists(SAVE_PATH):
        print(f"Save path '{SAVE_PATH}' does not exist. Please run the training script first.")
        return

    # Load the breast cancer dataset
    print("Loading the breast cancer dataset...")
    data = load_breast_cancer()
    X = data.data  # Feature matrix
    y = data.target  # Target vector (binary labels)
    feature_names = data.feature_names
    class_names = data.target_names.tolist()
    print("Dataset loaded successfully.\n")

    # Load the scaler used for preprocessing
    print("Loading the scaler used for preprocessing...")
    scaler_path = os.path.join(SAVE_PATH, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        print(f"Scaler file '{scaler_path}' not found. Please ensure the scaler was saved during training.")
        return
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.\n")

    # Split into training and test sets (ensure same split as training)
    print("Splitting data into training and test sets...")
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Ensures the split maintains the class distribution
    )
    print(f"Training samples: {X_train_np.shape[0]}, Test samples: {X_test_np.shape[0]}.\n")

    # Standardize the test data using the loaded scaler
    print("Standardizing the test data...")
    X_test_scaled = scaler.transform(X_test_np)
    print("Test data standardized successfully.\n")

    # Convert test data to PyTorch tensors for the PyTorch model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for PyTorch model: {device}")
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    print("Converted test data to PyTorch tensors.\n")

    # -------------------------------------------------------------------------
    # Load and Evaluate DecisionTreeClassifier (NumPy Implementation)
    # -------------------------------------------------------------------------
    print("Loading NumPy Decision Tree Classifier...")
    numpy_model_path = os.path.join(SAVE_PATH, 'decision_tree_numpy.joblib')
    if not os.path.exists(numpy_model_path):
        print(f"NumPy model file '{numpy_model_path}' not found. Please ensure the model was saved during training.")
        return

    # Load the NumPy model parameters
    numpy_model_params = joblib.load(numpy_model_path)

    # Initialize the NumPy Decision Tree Classifier with loaded parameters
    model_numpy = DecisionTreeClassifier(
        criterion=numpy_model_params['criterion'],
        max_depth=numpy_model_params['max_depth'],
        min_samples_split=numpy_model_params['min_samples_split'],
        min_samples_leaf=numpy_model_params['min_samples_leaf'],
        random_state=numpy_model_params['random_state']
    )
    model_numpy.tree = numpy_model_params['tree']
    print("NumPy Decision Tree Classifier loaded successfully.")

    # Make predictions with the NumPy model
    print("Making predictions with the NumPy model...")
    y_pred_numpy = model_numpy.predict(X_test_scaled)
    print("Predictions made with the NumPy model.\n")

    # -------------------------------------------------------------------------
    # Load and Evaluate DecisionTreeClassifierPyTorch (PyTorch Implementation)
    # -------------------------------------------------------------------------
    print("Loading PyTorch Decision Tree Classifier...")
    torch_model_path = os.path.join(SAVE_PATH, 'decision_tree_torch.pth')
    if not os.path.exists(torch_model_path):
        print(f"PyTorch model file '{torch_model_path}' not found. Please ensure the model was saved during training.")
        return

    # Load the PyTorch model parameters
    torch_model_params = torch.load(torch_model_path, map_location=device)

    # Initialize the PyTorch Decision Tree Classifier with loaded parameters
    model_torch = DecisionTreeClassifierPyTorch(
        criterion=torch_model_params['criterion'],
        max_depth=torch_model_params['max_depth'],
        min_samples_split=torch_model_params['min_samples_split'],
        min_samples_leaf=torch_model_params['min_samples_leaf'],
        random_state=torch_model_params['random_state']
    )
    model_torch.tree = torch_model_params['tree']
    print("PyTorch Decision Tree Classifier loaded successfully.")

    # Make predictions with the PyTorch model
    print("Making predictions with the PyTorch model...")
    y_pred_torch = model_torch.predict(X_test_tensor)
    y_pred_torch = y_pred_torch.cpu().numpy()  # Move to CPU and convert to NumPy array
    print("Predictions made with the PyTorch model.\n")

    # -------------------------------------------------------------------------
    # Load and Evaluate Scikit-learn DecisionTreeClassifier (Scikit-learn Implementation)
    # -------------------------------------------------------------------------
    print("Loading Scikit-learn Decision Tree Classifier...")
    sklearn_model_path = os.path.join(SAVE_PATH, 'decision_tree_sklearn.joblib')
    if not os.path.exists(sklearn_model_path):
        print(f"Scikit-learn model file '{sklearn_model_path}' not found. Please ensure the model was saved during training.")
        return

    # Load the Scikit-learn model
    model_sklearn = joblib.load(sklearn_model_path)
    print("Scikit-learn Decision Tree Classifier loaded successfully.")

    # Make predictions with the Scikit-learn model
    print("Making predictions with the Scikit-learn model...")
    y_pred_sklearn = model_sklearn.predict(X_test_scaled)
    print("Predictions made with the Scikit-learn model.\n")

    # -------------------------------------------------------------------------
    # Calculate and Display Metrics
    # -------------------------------------------------------------------------
    print("Calculating evaluation metrics...\n")
    results = {}

    # NumPy Classifier Metrics
    accuracy_numpy = accuracy_score(y_test_np, y_pred_numpy)
    cm_numpy = confusion_matrix(y_test_np, y_pred_numpy)
    results['NumPy'] = {
        'accuracy': accuracy_numpy,
        'confusion_matrix': cm_numpy
    }

    # PyTorch Classifier Metrics
    accuracy_torch = accuracy_score(y_test_np, y_pred_torch)
    cm_torch = confusion_matrix(y_test_np, y_pred_torch)
    results['PyTorch'] = {
        'accuracy': accuracy_torch,
        'confusion_matrix': cm_torch
    }

    # Scikit-learn Classifier Metrics
    accuracy_sklearn = accuracy_score(y_test_np, y_pred_sklearn)
    cm_sklearn = confusion_matrix(y_test_np, y_pred_sklearn)
    results['Scikit-learn'] = {
        'accuracy': accuracy_sklearn,
        'confusion_matrix': cm_sklearn
    }

    # Print Accuracies
    print("Model Accuracies:")
    for key, metrics in results.items():
        print(f"{key}: {metrics['accuracy'] * 100:.2f}%")
    print()

    # Plot Confusion Matrices
    print("Plotting confusion matrices...\n")
    for key, metrics in results.items():
        cm = metrics['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {key}")
        plt.show()

    # -------------------------------------------------------------------------
    # Optional: Visualize the Decision Trees
    # -------------------------------------------------------------------------
    visualize = input("Would you like to visualize the decision trees? (yes/no): ").strip().lower()
    if visualize in ['yes', 'y']:
        print("\nvisualising NumPy Decision Tree...")
        try:
            model_numpy.visualize(
                feature_names=feature_names,
                class_names=class_names,
                filename=os.path.join(SAVE_PATH, "decision_tree_numpy")
            )
            print("NumPy Decision Tree visualization saved successfully.")
        except Exception as e:
            print(f"An error occurred while visualising the NumPy Decision Tree: {e}")

        print("\nvisualising PyTorch Decision Tree...")
        try:
            model_torch.visualize(
                feature_names=feature_names,
                class_names=class_names,
                filename=os.path.join(SAVE_PATH, "decision_tree_pytorch")
            )
            print("PyTorch Decision Tree visualization saved successfully.")
        except Exception as e:
            print(f"An error occurred while visualising the PyTorch Decision Tree: {e}")

        print("\nvisualising Scikit-learn Decision Tree...")
        try:
            # Scikit-learn has built-in plotting; alternatively, use export_graphviz
            from sklearn import tree
            dot_data = tree.export_graphviz(
                model_sklearn,
                out_file=None,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True
            )
            import graphviz
            graph = graphviz.Source(dot_data)
            graph.render(os.path.join(SAVE_PATH, "decision_tree_sklearn"), format='png', cleanup=True)
            print("Scikit-learn Decision Tree visualization saved successfully.")
        except Exception as e:
            print(f"An error occurred while visualising the Scikit-learn Decision Tree: {e}")
    else:
        print("Skipping tree visualization.")

    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()
