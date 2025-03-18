# logistic_regression_scratch.py

import numpy as np  
import torch  

class LogisticRegressionNumpy:
    """
    A Hutsonified version of the Logistic Regression from scratch ML algorithm

    Attributes:
        penalty (str): Regularisation type ('l1', 'l2', or 'none').
        C (float): Inverse of regularisation strength; smaller values specify stronger regularisation.
        solver (str): Optimisation algorithm ('gradient_descent' or 'stochastic_gradient_descent').
        learning_rate (float): Learning rate for gradient descent.
        max_iter (int): Maximum number of iterations for the solver.
        tol (float): Tolerance for stopping criteria.
        multi_class (str): Multi-class strategy ('ovr' for One-vs-Rest).
        class_weight (dict or 'balanced'): Weights associated with classes.
        random_state (int): Seed for random number generator.
        theta (np.ndarray): Weights for the logistic regression model.
        bias (float or np.ndarray): Bias term(s) for the model.
        classes_ (np.ndarray): Unique class labels.
    """

    def __init__(self, penalty='l2', C=1.0, solver='gradient_descent', learning_rate=0.01,
                 max_iter=1000, tol=1e-4, multi_class='ovr', class_weight=None, random_state=None):
        """
        Initialises the LogisticRegressionScratch instance with given hyperparameters.

        Args:
            penalty (str): Regularisation type ('l1', 'l2', or 'none').
            C (float): Inverse of regularisation strength.
            solver (str): Optimisation algorithm.
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for stopping criteria.
            multi_class (str): Multi-class strategy.
            class_weight (dict or 'balanced'): Class weights for handling class imbalance.
            random_state (int): Seed for random number generation.
        """
        self.penalty = penalty              # Type of regularisation to use
        self.C = C                          # Inverse of regularisation strength
        self.solver = solver                # Optimisation algorithm
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.max_iter = max_iter            # Maximum number of iterations
        self.tol = tol                      # Tolerance for convergence criteria
        self.multi_class = multi_class      # Strategy for multi-class classification
        self.class_weight = class_weight    # Class weights for handling imbalance
        self.random_state = random_state    # Seed for random number generation
        self.theta = None                   # Weights of the model (initialised later)
        self.bias = None                    # Bias term(s) of the model (initialised later)
        self.classes_ = None                # Unique class labels (determined during fitting)

    def _initialize_parameters(self, n_features, n_classes):
        """
        Initialises weights and biases.

        Args:
            n_features (int): Number of features.
            n_classes (int): Number of classes.
        """
        # Create a random number generator with the given seed
        rng = np.random.default_rng(self.random_state)
        if n_classes == 2:
            # For binary classification, initialise weights as a 1D array
            self.theta = rng.normal(0, 0.01, size=(n_features,))
            self.bias = 0.0  # Scalar bias term
        else:
            # For multi-class classification, initialise weights as a 2D array using a Normal distribution
            # with a standard deviation of 0.01 and the size is the number of features in the model i.e. x ith features
            self.theta = rng.normal(0, 0.01, size=(n_classes, n_features))
            self.bias = np.zeros(n_classes)  # Bias term for each class

    def _sigmoid(self, z):
        """
        Computes the sigmoid function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of input.
        """
        # Compute the sigmoid activation function
    
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        """
        Computes the softmax function for multi-class classification.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        # Subtract the max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # this keeps the returned dimensions resulting in an array of shape [n_samples, 1]
        # Compute softmax probabilities
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, y, y_hat, weights):
        """
        Computes the loss function with regularisation.

        Args:
            y (np.ndarray): True labels.
            y_hat (np.ndarray): Predicted probabilities.
            weights (np.ndarray): Model weights.

        Returns:
            float: Computed loss.
        """
        m = y.shape[0]  # Number of samples

        # Compute regularisation term based on the penalty - L1 aka Lasso and L2 Ridge
        # L2 shrinks coefficents of thhe model but doesn't eliminate them
        # L1 can be used to get rid of redudant featues with no predictive power
        if self.penalty == 'l2':
            reg = (1 / (2 * self.C)) * np.sum(weights ** 2)  # L2 regularisation
        elif self.penalty == 'l1':
            reg = (1 / self.C) * np.sum(np.abs(weights))     # L1 regularisation
        else:
            reg = 0  # No regularisation

        # Avoid division by zero or log(0) by adding epsilon
        epsilon = 1e-15
        # np.clip removes outliers by capping extreme values
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        if self.classes_.shape[0] == 2:
            # Binary classification loss (log loss)
            loss = (-1 / m) * np.sum(
                y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
            ) + reg
        else:
            # Multi-class classification loss (cross-entropy loss)
            loss = (-1 / m) * np.sum(y * np.log(y_hat)) + reg

        return loss

    def _apply_regularization(self, dw):
        """
        Applies regularisation to the weight gradients.

        Args:
            dw (np.ndarray): Weight gradients.

        Returns:
            np.ndarray: Regularised weight gradients.
        """
        # Apply appropriate regularisation to gradients
        if self.penalty == 'l2':
            #dw is the gradient of the loss function we are utilising
            # theta is the vector of weights (parameters) of the model
            # C is the inverse of the reegularisation strength
            dw += (1 / self.C) * self.theta  # L2 regularisation term
        elif self.penalty == 'l1':
            dw += (1 / self.C) * np.sign(self.theta)  # L1 regularisation term
        # No change if no regularisation
        return dw

    def _encode_labels(self, y):
        """
        Encodes labels into one-hot vectors for multi-class classification.

        Args:
            y (np.ndarray): True labels.

        Returns:
            np.ndarray: One-hot encoded labels.
        """
        m = y.shape[0]  # Number of samples
        n_classes = self.classes_.shape[0]  # Number of classes
        # Initialise a zero matrix for one-hot encoding
        #np.zeros initiliases 
        y_encoded = np.zeros((m, n_classes))

        for idx, val in enumerate(y):
            # Find the index of the class label
            class_idx = np.where(self.classes_ == val)[0][0]
            # Set the corresponding position to 1
            y_encoded[idx, class_idx] = 1
        return y_encoded

    def fit(self, X, y):
        """
        Trains the logistic regression model using the specified solver.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        X = np.array(X)  # Convert to NumPy array if not already
        y = np.array(y)
        m, n_features = X.shape  # Number of samples and features
        self.classes_ = np.unique(y)  # Unique class labels
        n_classes = self.classes_.shape[0]  # Number of unique classes

        # Initialise model parameters (weights and biases)
        self._initialize_parameters(n_features, n_classes)

        # Handle class weights for imbalance
        if self.class_weight == 'balanced':
            # Compute class frequencies
            class_counts = np.bincount(y)
            total_samples = y.shape[0]
            # Compute class weights inversely proportional to class frequencies
            class_weights = total_samples / (len(self.classes_) * class_counts)
            # Assign weights to each sample
            sample_weights = class_weights[y]
        elif isinstance(self.class_weight, dict):
            # Use custom class weights provided in a dictionary
            class_weights = np.ones(len(self.classes_))
            for class_index, class_label in enumerate(self.classes_):
                if class_label in self.class_weight:
                    class_weights[class_index] = self.class_weight[class_label]
            sample_weights = class_weights[y]
        else:
            # No class weighting
            sample_weights = np.ones(m)

        # Training loop for gradient descent
        for _ in range(self.max_iter):
            if n_classes == 2:
                # Binary classification
                z = np.dot(X, self.theta) + self.bias  # Linear combination
                y_hat = self._sigmoid(z)               # Sigmoid activation
                loss = self._compute_loss(y, y_hat, self.theta)  # Compute loss
                error = y_hat - y                      # Compute error term
                dw = (1 / m) * np.dot(X.T, error * sample_weights)  # Gradient w.r.t weights
                db = (1 / m) * np.sum(error * sample_weights)       # Gradient w.r.t bias
            else:
                # Multi-class classification
                z = np.dot(X, self.theta.T) + self.bias    # Linear combination
                y_hat = self._softmax(z)                   # Softmax activation
                y_encoded = self._encode_labels(y)         # One-hot encode labels
                loss = self._compute_loss(y_encoded, y_hat, self.theta)  # Compute loss
                error = y_hat - y_encoded                  # Compute error term
                # Gradient w.r.t weights
                dw = (1 / m) * np.dot(error.T, X * sample_weights[:, np.newaxis])
                # Gradient w.r.t biases
                db = (1 / m) * np.sum(error * sample_weights[:, np.newaxis], axis=0)

            # Apply regularisation to gradients
            dw = self._apply_regularization(dw)

            # Update weights and biases using gradient descent
            if n_classes == 2:
                # Update for binary classification
                self.theta -= self.learning_rate * dw
    def predict_proba(self, X):
        """
        Predicts probabilities using the logistic regression model.

        Args:
            X (np.ndarray): Data features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        X = np.array(X)
        if self.classes_.shape[0] == 2:
            # Binary classification
            z = np.dot(X, self.theta) + self.bias
            y_hat = self._sigmoid(z)
            return np.vstack([1 - y_hat, y_hat]).T  # Return probabilities for both classes
        else:
            # Multi-class classification
            z = np.dot(X, self.theta.T) + self.bias
            y_hat = self._softmax(z)
            return y_hat  # Probabilities for each class
        
    def predict(self, X):
        """
        Predicts class labels for given data.

        Args:
            X (np.ndarray): Data features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X = np.array(X)
        if self.classes_.shape[0] == 2:
            # Binary classification
            z = np.dot(X, self.theta) + self.bias
            y_hat = self._sigmoid(z)
            return (y_hat >= 0.5).astype(int)
        else:
            # Multi-class classification
            z = np.dot(X, self.theta.T) + self.bias
            y_hat = self._softmax(z)
            y_pred_indices = np.argmax(y_hat, axis=1)
            return self.classes_[y_pred_indices]


# logistic_regression_scratch.py


class LogisticRegressionTorch:
    """
    A PyTorch-based implementation of Logistic Regression from scratch.

    Attributes:
        penalty (str): Regularisation type ('l1', 'l2', or 'none').
        C (float): Inverse of regularisation strength; smaller values specify stronger regularisation.
        solver (str): Optimisation algorithm ('gradient_descent' or 'stochastic_gradient_descent').
        learning_rate (float): Learning rate for gradient descent.
        max_iter (int): Maximum number of iterations for the solver.
        tol (float): Tolerance for stopping criteria.
        multi_class (str): Multi-class strategy ('ovr' for One-vs-Rest).
        class_weight (dict or 'balanced'): Weights associated with classes.
        random_state (int): Seed for random number generator.
        theta (torch.Tensor): Weights for the logistic regression model.
        bias (torch.Tensor): Bias term(s) for the model.
        classes_ (torch.Tensor): Unique class labels.
        device (torch.device): Device to run computations on (CPU or GPU).
    """

    def __init__(self, penalty='l2', C=1.0, solver='gradient_descent', learning_rate=0.01,
                 max_iter=1000, tol=1e-4, multi_class='ovr', class_weight=None, random_state=None):
        """
        Initialises the LogisticRegressionScratch instance with given hyperparameters.

        Args:
            penalty (str): Regularisation type ('l1', 'l2', or 'none').
            C (float): Inverse of regularisation strength.
            solver (str): Optimisation algorithm.
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for stopping criteria.
            multi_class (str): Multi-class strategy.
            class_weight (dict or 'balanced'): Class weights for handling class imbalance.
            random_state (int): Seed for random number generation.
        """
        self.penalty = penalty              # Type of regularisation to use
        self.C = C                          # Inverse of regularisation strength
        self.solver = solver                # Optimisation algorithm
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.max_iter = max_iter            # Maximum number of iterations
        self.tol = tol                      # Tolerance for convergence criteria
        self.multi_class = multi_class      # Strategy for multi-class classification
        self.class_weight = class_weight    # Class weights for handling imbalance
        self.random_state = random_state    # Seed for random number generation
        self.theta = None                   # Weights of the model (initialised later)
        self.bias = None                    # Bias term(s) of the model (initialised later)
        self.classes_ = None                # Unique class labels (determined during fitting)
        # Set device to GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _initialize_parameters(self, n_features, n_classes):
        """
        Initialises weights and biases.

        Args:
            n_features (int): Number of features.
            n_classes (int): Number of classes.
        """
        # Set the random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        if n_classes == 2:
            # For binary classification, initialise weights as a 1D tensor
            self.theta = torch.randn(n_features, device=self.device) * 0.01
            self.bias = torch.tensor(0.0, device=self.device)  # Scalar bias term
        else:
            # For multi-class classification, initialise weights as a 2D tensor
            self.theta = torch.randn(n_classes, n_features, device=self.device) * 0.01
            self.bias = torch.zeros(n_classes, device=self.device)  # Bias term for each class

    def _sigmoid(self, z):
        """
        Computes the sigmoid function.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sigmoid of input.
        """
        return torch.sigmoid(z)

    def _softmax(self, z):
        """
        Computes the softmax function for multi-class classification.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Softmax probabilities.
        """
        return torch.softmax(z, dim=1)

    def _compute_loss(self, y, y_hat, weights):
        """
        Computes the loss function with regularisation.

        Args:
            y (torch.Tensor): True labels.
            y_hat (torch.Tensor): Predicted probabilities.
            weights (torch.Tensor): Model weights.

        Returns:
            torch.Tensor: Computed loss.
        """
        m = y.shape[0]  # Number of samples

        # Compute regularisation term based on the penalty
        if self.penalty == 'l2':
            reg = (1 / (2 * self.C)) * torch.sum(weights ** 2)  # L2 regularisation
        elif self.penalty == 'l1':
            reg = (1 / self.C) * torch.sum(torch.abs(weights))   # L1 regularisation
        else:
            reg = 0.0  # No regularisation

        # Compute the loss
        epsilon = 1e-15
        y_hat = torch.clamp(y_hat, epsilon, 1 - epsilon)

        if self.classes_.shape[0] == 2:
            # Binary classification loss (log loss)
            loss = (-1 / m) * torch.sum(
                y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)
            ) + reg
        else:
            # Multi-class classification loss (cross-entropy loss)
            loss = (-1 / m) * torch.sum(y * torch.log(y_hat)) + reg

        return loss

    def _apply_regularization(self, dw):
        """
        Applies regularisation to the weight gradients.

        Args:
            dw (torch.Tensor): Weight gradients.

        Returns:
            torch.Tensor: Regularised weight gradients.
        """
        # Apply appropriate regularisation to gradients
        if self.penalty == 'l2':
            dw += (1 / self.C) * self.theta  # L2 regularisation term
        elif self.penalty == 'l1':
            dw += (1 / self.C) * torch.sign(self.theta)  # L1 regularisation term
        # No change if no regularisation
        return dw

    def _encode_labels(self, y):
        """
        Encodes labels into one-hot vectors for multi-class classification.

        Args:
            y (torch.Tensor): True labels.

        Returns:
            torch.Tensor: One-hot encoded labels.
        """
        m = y.shape[0]  # Number of samples
        n_classes = self.classes_.shape[0]  # Number of classes

        # Initialise a zero matrix for one-hot encoding
        y_encoded = torch.zeros((m, n_classes), device=self.device)
        for idx, val in enumerate(y):
            class_idx = (self.classes_ == val).nonzero(as_tuple=True)[0]
            y_encoded[idx, class_idx] = 1
        return y_encoded

    def fit(self, X, y):
        """
        Trains the logistic regression model using the specified solver.

        Args:
            X (torch.Tensor or np.ndarray): Training data features.
            y (torch.Tensor or np.ndarray): Training data labels.
        """
        # Convert inputs to PyTorch tensors and move to the appropriate device
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # X = X.clone().detach().to(dtype=torch.float32, device=self.device)
        # y = y.clone().detach().to(dtype=torch.float32, device=self.device)


        m, n_features = X.shape  # Number of samples and features
        self.classes_ = torch.unique(y)  # Unique class labels
        n_classes = self.classes_.shape[0]  # Number of unique classes

        # Initialise model parameters (weights and biases)
        self._initialize_parameters(n_features, n_classes)

        # Handle class weights for imbalance
        if self.class_weight == 'balanced':
            # Compute class frequencies
            class_counts = torch.bincount(y.long())
            total_samples = y.shape[0]
            # Compute class weights inversely proportional to class frequencies
            class_weights = total_samples / (len(self.classes_) * class_counts)
            # Assign weights to each sample
            sample_weights = class_weights[y.long()]
        elif isinstance(self.class_weight, dict):
            # Use custom class weights provided in a dictionary
            class_weights = torch.ones(len(self.classes_), device=self.device)
            for class_index, class_label in enumerate(self.classes_):
                if class_label.item() in self.class_weight:
                    class_weights[class_index] = self.class_weight[class_label.item()]
            sample_weights = class_weights[y.long()]
        else:
            # No class weighting
            sample_weights = torch.ones(m, device=self.device)

        # Training loop for gradient descent
        for iteration in range(self.max_iter):
            if n_classes == 2:
                # Binary classification
                z = torch.matmul(X, self.theta) + self.bias  # Linear combination
                y_hat = self._sigmoid(z)                     # Sigmoid activation
                loss = self._compute_loss(y, y_hat, self.theta)  # Compute loss
                error = y_hat - y                            # Compute error term
                dw = (1 / m) * torch.matmul(X.T, error * sample_weights)  # Gradient w.r.t weights
                db = (1 / m) * torch.sum(error * sample_weights)          # Gradient w.r.t bias
            else:
                # Multi-class classification
                z = torch.matmul(X, self.theta.T) + self.bias  # Linear combination
                y_hat = self._softmax(z)                       # Softmax activation
                y_encoded = self._encode_labels(y)             # One-hot encode labels
                loss = self._compute_loss(y_encoded, y_hat, self.theta)  # Compute loss
                error = y_hat - y_encoded                      # Compute error term
                # Gradient w.r.t weights
                dw = (1 / m) * torch.matmul((error * sample_weights.unsqueeze(1)).T, X)
                # Gradient w.r.t biases
                db = (1 / m) * torch.sum(error * sample_weights.unsqueeze(1), axis=0)

            # Apply regularisation to gradients
            dw = self._apply_regularization(dw)

            # Update weights and biases using gradient descent
            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check for convergence
            if torch.norm(dw) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

    def predict_prob(self, X):
        """
        Predicts probabilities using the logistic regression model.

        Args:
            X (torch.Tensor or np.ndarray): Data features.

        Returns:
            torch.Tensor: Predicted probabilities.
        """
        # Convert input to PyTorch tensor and move to device
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        if self.classes_.shape[0] == 2:
            # Binary classification
            z = torch.matmul(X, self.theta) + self.bias  # Linear combination
            return self._sigmoid(z)                      # Sigmoid activation
        else:
            # Multi-class classification
            z = torch.matmul(X, self.theta.T) + self.bias  # Linear combination
            return self._softmax(z)                        # Softmax activation

    def predict(self, X):
        """
        Predicts class labels for given data.

        Args:
            X (torch.Tensor or np.ndarray): Data features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        y_hat_prob = self.predict_prob(X)
        if self.classes_.shape[0] == 2:
            # Binary classification
            y_hat = (y_hat_prob >= 0.5).long()
            return y_hat.cpu().numpy()
        else:
            # Multi-class classification
            y_hat_indices = torch.argmax(y_hat_prob, axis=1)
            return self.classes_[y_hat_indices].cpu().numpy()
