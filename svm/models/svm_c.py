import numpy as np
import torch

class SVM_C_Numpy:
    """
    A Support Vector Machine (SVM) classifier using NumPy.
    This implementation uses the hinge loss and batch gradient descent for optimization.
    """

    def __init__(self, C=1.0, n_iters=1000, learning_rate=0.001):
        """
        Initializes the SVM classifier.

        Parameters:
        - C (float): Regularization parameter.
        - n_iters (int): Number of iterations for training.
        - learning_rate (float): The step size for gradient descent.
        """
        self.C = C
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVM classifier on the provided data.

        Parameters:
        - X (numpy.ndarray): Training data of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels corresponding to X, should be -1 or 1.
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for _ in range(self.n_iters):
            # Compute the decision function
            linear_output = np.dot(X, self.w) + self.b
            # Identify misclassified samples
            condition = y * linear_output < 1
            # Compute gradients
            dw = self.w - self.C * np.dot(X.T, y * condition) / n_samples
            db = -self.C * np.sum(y * condition) / n_samples

            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts the class labels for the given samples.

        Parameters:
        - X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted class labels (-1 or 1).
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

class SVM_C_PyTorch:
    """
    A Support Vector Machine (SVM) classifier using PyTorch.
    This implementation uses the hinge loss and batch gradient descent for optimization.
    """

    def __init__(self, C=1.0, n_iters=1000, learning_rate=0.001):
        """
        Initializes the SVM classifier.

        Parameters:
        - C (float): Regularization parameter.
        - n_iters (int): Number of iterations for training.
        - learning_rate (float): The step size for gradient descent.
        """
        self.C = C
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVM classifier on the provided data.

        Parameters:
        - X (numpy.ndarray): Training data of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels corresponding to X, should be -1 or 1.
        """
        n_samples, n_features = X.shape

        # Convert data to tensors
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Initialize weights and bias
        self.w = torch.zeros(n_features, requires_grad=True, device=self.device)
        self.b = torch.zeros(1, requires_grad=True, device=self.device)

        # Optimizer
        optimizer = torch.optim.SGD([self.w, self.b], lr=self.learning_rate)

        # Training loop
        for _ in range(self.n_iters):
            optimizer.zero_grad()

            # Compute the decision function
            linear_output = X.matmul(self.w) + self.b
            # Compute hinge loss
            hinge_loss = torch.clamp(1 - y * linear_output, min=0)
            # Compute total loss
            loss = 0.5 * torch.dot(self.w, self.w) + self.C * torch.mean(hinge_loss)

            # Backward pass
            loss.backward()
            optimizer.step()

    def predict(self, X):
        """
        Predicts the class labels for the given samples.

        Parameters:
        - X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted class labels (-1 or 1).
        """
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            linear_output = X.matmul(self.w) + self.b
            return torch.sign(linear_output).cpu().numpy()
