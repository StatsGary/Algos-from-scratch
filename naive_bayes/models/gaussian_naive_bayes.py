import numpy as np
import torch

class GaussianNaiveBayesNumpy:
    """
    Gaussian Naive Bayes classifier using NumPy.
    """

    # Constructor method to initialize the classifier
    def __init__(self, priors=None):
        """
        Initialize the classifier.

        Args:
            priors (array-like, shape (n_classes,)): Prior probabilities of the classes.
        """
        # Store the prior probabilities if provided
        self.priors = priors
        # These attributes will be set during fitting
        self.classes_ = None          # Array of unique class labels
        self.class_prior_ = None      # Prior probabilities of classes
        self.theta_ = None            # Mean of each feature per class
        self.sigma_ = None            # Variance of each feature per class

    # Method to fit the model to the training data
    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model according to X, y.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data.
            y (array-like, shape (n_samples,)): Target values.
        """
        # Convert inputs to NumPy arrays for processing
        X = np.array(X)
        y = np.array(y)
        # Get the number of samples and features
        n_samples, n_features = X.shape
        # Identify unique class labels in y
        self.classes_ = np.unique(y)
        # Based on these unique class labels get the shape to indicate the classification type
        n_classes = self.classes_.shape[0]

        # Initialise arrays to store class-specific statistics
        self.theta_ = np.zeros((n_classes, n_features))       # Mean of features per class - np.zeros fills with zeros until argument passed
        self.sigma_ = np.zeros((n_classes, n_features))       # Variance of features per class
        self.class_prior_ = np.zeros(n_classes)               # Prior probabilities per class

        # Compute statistics for each class
        for idx, cls in enumerate(self.classes_):
            # Select samples belonging to the current class
            X_c = X[y == cls]
            # Compute mean of each feature for the current class
            self.theta_[idx, :] = X_c.mean(axis=0)
            # Compute variance of each feature for the current class
            self.sigma_[idx, :] = X_c.var(axis=0) + 1e-9      # Add a small value to avoid division by zero
            # Compute prior probability of the current class
            self.class_prior_[idx] = X_c.shape[0] / n_samples

        # Override prior probabilities if they are provided
        if self.priors is not None:
            self.class_prior_ = np.array(self.priors)

        # Return the fitted model
        return self

    # Internal method to calculate log probabilities
    def _calculate_log_probability(self, X):
        """
        Calculate the log probability of X for each class.

        Args:
            X (array-like, shape (n_samples, n_features)): Input data.

        Returns:
            log_prob (array, shape (n_samples, n_classes)): Log probabilities.
        """
        # Get the number of samples and features
        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]
        # Initialize an array to hold log probabilities
        log_prob = np.zeros((n_samples, n_classes))

        # Calculate log probabilities for each class
        for idx in range(n_classes):
            # Retrieve mean and variance for the current class
            mean = self.theta_[idx]
            var = self.sigma_[idx]
            # Compute the log of the Gaussian (Normal Distribution) probability density function (also known at PDF) for each feature
            # Compute the exponent term
            exponent = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            # Compute the normalization term
            log_det = -0.5 * np.sum(np.log(2. * np.pi * var))
            # Add log prior probability
            log_prior = np.log(self.class_prior_[idx])
            # Combine terms to get log probability for the class
            log_prob[:, idx] = exponent + log_det + log_prior

        # Return log probabilities
        return log_prob

    # Method to compute class probabilities for input data
    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Args:
            X (array-like, shape (n_samples, n_features)): Input data.

        Returns:
            probabilities (array, shape (n_samples, n_classes)): Probability estimates.
        """
        # Convert input data to NumPy array
        X = np.array(X)
        # Calculate log probabilities
        log_prob = self._calculate_log_probability(X)
        # Use the log-sum-exp trick for numerical stability and to maintain precision involving very small or very large numbers
        max_log_prob = log_prob.max(axis=1, keepdims=True)
        # Take away the log_prob from the max_log_prob and assign back to log_prob (alternate form log_prob = log_prob - max_log_prob)
        log_prob -= max_log_prob
        # Convert log probabilities to probabilities
        prob = np.exp(log_prob)
        # Normalize probabilities to sum to 1 for each sample
        # Shorthand for prob = prob / prob.sum(axis=1, keepdims=True)
        # Axis 1 refers to rows, with 0 being columar
        prob /= prob.sum(axis=1, keepdims=True)
        # Return probabilities
        return prob

    # Method to predict class labels for input data
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Args:
            X (array-like, shape (n_samples, n_features)): Input data.

        Returns:
            y_pred (array, shape (n_samples,)): Predicted target values.
        """
        # Convert input data to NumPy array
        X = np.array(X)
        # Calculate log probabilities
        log_prob = self._calculate_log_probability(X)
        # Predict the class with the highest log probability for each sample - essentially pick the most likely class label
        # based on the probablistic estimate
        y_pred_indices = np.argmax(log_prob, axis=1)
        y_pred = self.classes_[y_pred_indices]
        # Return predicted class labels
        return y_pred

    # Method to compute the accuracy of the classifier
    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like, shape (n_samples, n_features)): Test samples.
            y (array-like, shape (n_samples,)): True labels for X.

        Returns:
            score (float): Mean accuracy.
        """
        # Predict class labels for test data
        y_pred = self.predict(X)
        # Compute the proportion of correct predictions
        accuracy = np.mean(y_pred == y)
        # Return accuracy score
        return accuracy


class GaussianNaiveBayesTorch:
    """
    Gaussian Naive Bayes classifier using PyTorch.
    """

    def __init__(self, priors=None):
        """
        Initialize the classifier.

        Args:
            priors (array-like, shape (n_classes,)): Prior probabilities of the classes.
        """
        self.priors = priors
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # Mean of each feature per class
        self.sigma_ = None  # Variance of each feature per class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model according to X, y.

        Args:
            X (array-like or torch.Tensor, shape (n_samples, n_features)): Training data.
            y (array-like or torch.Tensor, shape (n_samples,)): Target values.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        n_samples, n_features = X.shape
        self.classes_ = torch.unique(y)
        n_classes = self.classes_.shape[0]

        # Initialize mean, variance, and prior probability for each class
        self.theta_ = torch.zeros((n_classes, n_features), device=self.device)
        self.sigma_ = torch.zeros((n_classes, n_features), device=self.device)
        self.class_prior_ = torch.zeros(n_classes, device=self.device)

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.theta_[idx, :] = X_c.mean(dim=0)
            self.sigma_[idx, :] = X_c.var(dim=0) + 1e-9  # Adding epsilon to avoid division by zero
            self.class_prior_[idx] = X_c.shape[0] / n_samples

        # Override priors if provided
        if self.priors is not None:
            self.class_prior_ = torch.tensor(self.priors, device=self.device, dtype=torch.float32)

        return self

    def _calculate_log_probability(self, X):
        """
        Calculate the log probability of X for each class.

        Args:
            X (torch.Tensor, shape (n_samples, n_features)): Input data.

        Returns:
            log_prob (torch.Tensor, shape (n_samples, n_classes)): Log probabilities.
        """
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]
        log_prob = torch.zeros((n_samples, n_classes), device=self.device)

        for idx in range(n_classes):
            mean = self.theta_[idx]
            var = self.sigma_[idx]
            # Compute the log probability density function for Gaussian distribution
            log_prob[:, idx] = -0.5 * torch.sum(torch.log(2. * torch.pi * var))
            log_prob[:, idx] -= 0.5 * torch.sum(((X - mean) ** 2) / var, dim=1)
            log_prob[:, idx] += torch.log(self.class_prior_[idx])

        return log_prob

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Args:
            X (array-like or torch.Tensor, shape (n_samples, n_features)): Input data.

        Returns:
            probabilities (numpy.ndarray, shape (n_samples, n_classes)): Probability estimates.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        log_prob = self._calculate_log_probability(X)
        # Use log-sum-exp for numerical stability
        log_prob = log_prob - log_prob.max(dim=1, keepdim=True)[0]
        prob = torch.exp(log_prob)
        prob = prob / prob.sum(dim=1, keepdim=True)
        return prob.cpu().numpy()

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Args:
            X (array-like or torch.Tensor, shape (n_samples, n_features)): Input data.

        Returns:
            y_pred (numpy.ndarray, shape (n_samples,)): Predicted target values.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        log_prob = self._calculate_log_probability(X)
        y_pred_indices = torch.argmax(log_prob, dim=1)
        y_pred = self.classes_[y_pred_indices]
        return y_pred.cpu().numpy()

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (array-like or torch.Tensor, shape (n_samples, n_features)): Test samples.
            y (array-like or torch.Tensor, shape (n_samples,)): True labels for X.

        Returns:
            score (float): Mean accuracy.
        """
        y_pred = self.predict(X)
        y = np.array(y)
        return np.mean(y_pred == y)
