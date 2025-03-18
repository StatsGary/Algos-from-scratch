import numpy as np

class SVM_Numpy:
    """
    A Support Vector Machine (SVM) classifier using NumPy.

    This implementation uses the hinge loss and batch gradient descent for optimization.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initializes the SVM classifier.

        Parameters:
        - learning_rate (float): The step size for gradient descent.
        - lambda_param (float): Regularization parameter to prevent overfitting.
        - n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weights vector
        self.b = None  # Bias term

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

        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Training loop
        for _ in range(self.n_iters):
            # Compute the decision function
            linear_output = np.dot(X, self.w) + self.b
            # Compute the gradient components
            condition = y_ * linear_output < 1
            # Compute gradients
            dw = 2 * self.lambda_param * self.w - np.dot(X.T, y_ * condition) / n_samples
            db = -np.sum(y_ * condition) / n_samples

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


import torch

class SVM_PyTorch:
    """
    A Support Vector Machine (SVM) classifier using PyTorch.

    This implementation uses the hinge loss and batch gradient descent for optimization.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initializes the SVM classifier.

        Parameters:
        - learning_rate (float): The step size for gradient descent.
        - lambda_param (float): Regularization parameter to prevent overfitting.
        - n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.w = None  # Weights vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """
        Trains the SVM classifier on the provided data.

        Parameters:
        - X (numpy.ndarray): Training data of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels corresponding to X, should be -1 or 1.
        """
        n_samples, n_features = X.shape

        # Convert input features to PyTorch tensors and move to the appropriate device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # Features tensor
        y = torch.tensor(y, dtype=torch.float32).to(self.device)  # Labels tensor

        # Initialize weights and bias with requires_grad=True to enable gradient computation
        self.w = torch.zeros(n_features, requires_grad=True, device=self.device)  # Weight vector
        self.b = torch.zeros(1, requires_grad=True, device=self.device)  # Bias term

        # Optimizer (using SGD with no momentum)
        optimizer = torch.optim.SGD([self.w, self.b], lr=self.learning_rate)

        # Training loop
        for _ in range(self.n_iters):
            optimizer.zero_grad()  # Zero the gradients

            # Compute the decision function
            linear_output = X.matmul(self.w) + self.b  # Shape: (n_samples,)
            # Compute the hinge loss
            condition = y * linear_output
            hinge_loss = torch.mean(torch.clamp(1 - condition, min=0))
            # Compute the regularization term
            l2_term = self.lambda_param * torch.dot(self.w, self.w)
            # Compute the total loss
            loss = l2_term + hinge_loss

            # Backward pass
            loss.backward()
            # Update weights and bias
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
