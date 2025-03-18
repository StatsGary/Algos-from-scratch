import numpy as np
import torch
import torch.nn as nn

class SVM_Numpy:
    """
    A Support Vector Machine (SVM) classifier using NumPy.

    This implementation uses the hinge loss and gradient descent for optimization.
    """

    def __init__(self, 
                 learning_rate=0.001, 
                 lambda_param=0.01, 
                 n_iters=1000):
        """
        Initializes the SVM classifier.

        Parameters:
        - learning_rate (float): The step size for gradient descent.
        - lambda_param (float): Regularization parameter to prevent overfitting.
        - n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate  # Learning rate for weight updates
        self.lambda_param = lambda_param    # Regularisation parameter
        self.n_iters = n_iters              # Number of training iterations
        self.w = None  # Weights vector (to be initialised during training)
        self.b = None  # Bias term (to be initialised during training)

    def fit(self, X, y):
        """
        Trains the SVM classifier on the provided data.

        Parameters:
        - X (numpy.ndarray): Training data of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels corresponding to X, should be -1 or 1.
        """
        n_samples, n_features = X.shape  # Number of samples and features

        # Initialize weights and bias to zeros
        self.w = np.zeros(n_features)  # Initialize weight vector with zeros
        self.b = 0  # Initialize bias to zero

        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1

        # Training loop for the specified number of iterations
        for _ in range(self.n_iters):
            # Iterate over each sample in the dataset
            for idx, x_i in enumerate(X):
                # Calculate the condition y_i * (w^T x_i + b) >= 1 this ensures that each training example if not only
                # correctly classified but also lies outside the margies boundaries established by the SVM separation hyperplane
                # Using the dot product between each x_i and the weights matrix
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # If the condition is met, only update the weights with regularization
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # If the condition is not met, update both weights and bias
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predicts the class labels for the given samples.

        Parameters:
        - X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted class labels (-1 or 1).
        """
        # Compute the decision function (w^T x + b) for each sample
        approx = np.dot(X, self.w) + self.b
        # Assign class labels based on the sign of the decision function
        return np.sign(approx)
    


class SVM_PyTorch:
    """
    A Support Vector Machine (SVM) classifier using PyTorch.

    This implementation uses the hinge loss and stochastic gradient descent for optimization.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initializes the SVM classifier.

        Parameters:
        - learning_rate (float): The step size for gradient descent.
        - lambda_param (float): Regularization parameter to prevent overfitting.
        - n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate  # Learning rate for weight updates
        self.lambda_param = lambda_param    # Regularisation parameter
        self.n_iters = n_iters              # Number of training iterations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.w = None  # Weights vector (to be initialised during training)
        self.b = None  # Bias term (to be initialised during training)

    def fit(self, X, y):
        """
        Trains the SVM classifier on the provided data.

        Parameters:
        - X (numpy.ndarray): Training data of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels corresponding to X, should be -1 or 1.
        """
        # Convert input features to PyTorch tensors and move to the appropriate device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # Features tensor
        y = torch.tensor(y, dtype=torch.float32).to(self.device)  # Labels tensor

        # Initialize weights and bias with requires_grad=True to enable gradient computation
        self.w = torch.zeros(X.shape[1], requires_grad=True, device=self.device)  # Weight vector
        self.b = torch.zeros(1, requires_grad=True, device=self.device)  # Bias term

        # Training loop for the specified number of iterations
        for _ in range(self.n_iters):
            # Calculate the decision function y_i * (w^T x_i + b) for all samples
            condition = y * (torch.mv(X, self.w) + self.b) >= 1  # Boolean tensor

            # Compute hinge loss: max(0, 1 - y_i * (w^T x_i + b))
            hinge_loss = torch.where(condition, 
                                     torch.tensor(0.0, device=self.device),  # No loss if condition is True
                                     1 - y * (torch.mv(X, self.w) + self.b))  # Hinge loss otherwise

            # Compute the total loss: regularization term + average hinge loss 
            # Hinge loss is a function to train classifiersthat optimise to increase the margin 
            # between data points and the hyperplane decision boundary
            loss = torch.mean(self.lambda_param * self.w.pow(2)) + torch.mean(hinge_loss)

            # Perform backpropagation to compute gradients
            loss.backward()

            with torch.no_grad():
                # Update weights by subtracting the gradient scaled by the learning rate
                self.w -= self.learning_rate * self.w.grad
                # Update bias by subtracting the gradient scaled by the learning rate
                self.b -= self.learning_rate * self.b.grad

                # Reset gradients to zero for the next iteration
                self.w.grad.zero_()
                self.b.grad.zero_()

    def predict(self, X):
        """
        Predicts the class labels for the given samples.

        Parameters:
        - X (numpy.ndarray): Samples to predict, shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted class labels (-1 or 1).
        """
            # Ensure no gradients are computed during prediction
        with torch.no_grad():
            # Convert input features to PyTorch tensor and move to the appropriate device
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Compute the decision function (w^T x + b) for each sample
            approx = torch.mv(X, self.w) + self.b
            # Assign class labels based on the sign of the decision function and move back to CPU
            return torch.sign(approx).cpu().numpy()
        


import numpy as np

class SVM_Numpy_new:
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

        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Training loop
        for i in range(self.n_iters):
            # Compute the decision function
            linear_output = np.dot(X, self.w) + self.b
            # Compute the hinge loss gradient
            condition = y_ * linear_output < 1
            dw = 2 * self.lambda_param * self.w - np.dot(X.T, y_ * condition)
            db = -np.sum(y_ * condition)

            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Optional: Monitor the loss
            if i % 100 == 0:
                distances = 1 - y_ * linear_output
                distances = np.maximum(0, distances)
                hinge_loss = np.mean(distances)
                loss = self.lambda_param * np.dot(self.w, self.w) + hinge_loss
                print(f"Iteration {i}, Loss: {loss}")

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


