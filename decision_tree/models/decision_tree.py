import numpy as np

class DecisionTreeClassifier:
    """Decision Tree Classifier for binary classification."""

    def __init__(self, 
                 criterion='gini',
                 max_depth=None, 
                 min_samples_split=2, 
                 min_samples_leaf=1, 
                 random_state=None):
        """
        Initializes the Decision Tree Classifier.

        Parameters:
        criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        random_state (int): Controls the randomness of the estimator.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        """Fits the model to the training data."""
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """Predicts the class labels for the input samples."""
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _grow_tree(self, X, y, depth=0):
        """Recursively grows the tree."""
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria
        if (len(unique_classes) == 1 or
                (self.max_depth is not None and depth >= self.max_depth) or
                num_samples < self.min_samples_split):
            return np.bincount(y).argmax()

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Check for minimum samples in leaves
        if left_indices.sum() < self.min_samples_leaf or right_indices.sum() < self.min_samples_leaf:
            return np.bincount(y).argmax()

        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        """Finds the best feature and threshold to split on."""
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None
        best_gain = -1

        for feature in range(num_features):
            # Sort the data along the feature
            sorted_indices = np.argsort(X[:, feature])
            thresholds = X[sorted_indices, feature]
            classes = y[sorted_indices]

            num_left = np.zeros(len(np.unique(y)), dtype=int)
            num_right = np.bincount(classes, minlength=len(np.unique(y)))

            for i in range(1, num_samples):
                class_index = classes[i - 1]
                num_left[class_index] += 1
                num_right[class_index] -= 1

                # Skip if the current threshold is the same as the previous one
                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Calculate information gain
                gain = self._information_gain(num_left, num_right)

                # If the gain is better than the best gain found so far, update the best values
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_feature, best_threshold

    def _information_gain(self, num_left, num_right):
        """Calculates the information gain from a split based on the chosen criterion."""
        total = num_left.sum() + num_right.sum()
        if total == 0:
            return 0

        # Calculate parent impurity
        parent_counts = num_left + num_right
        if self.criterion == 'gini':
            parent_impurity = self._gini(parent_counts)
            left_impurity = self._gini(num_left)
            right_impurity = self._gini(num_right)
        elif self.criterion == 'entropy':
            parent_impurity = self._entropy(parent_counts)
            left_impurity = self._entropy(num_left)
            right_impurity = self._entropy(num_right)
        else:
            raise ValueError("Invalid criterion. Supported criteria: 'gini', 'entropy'.")

        # Calculate weighted impurity after split
        weighted_impurity = (num_left.sum() / total) * left_impurity + (num_right.sum() / total) * right_impurity

        # Information gain is reduction in impurity
        gain = parent_impurity - weighted_impurity
        return gain

    def _gini(self, counts):
        """Calculates the Gini impurity."""
        total = counts.sum()
        if total == 0:
            return 0
        p = counts / total
        return 1.0 - np.sum(p ** 2)

    def _entropy(self, counts):
        """Calculates the entropy."""
        total = counts.sum()
        if total == 0:
            return 0
        p = counts / total
        return -np.sum(p * np.log2(p + 1e-9))  # Adding a small value to avoid log(0)

    def _predict_sample(self, sample, tree):
        """Predicts the class label for a single sample."""
        if isinstance(tree, tuple):
            feature, threshold, left, right = tree
            if sample[feature] < threshold:
                return self._predict_sample(sample, left)
            else:
                return self._predict_sample(sample, right)
        else:
            return tree

    def visualize(self, tree=None, feature_names=None, class_names=None, filename="decision_tree_numpy"):
        """
        Visualizes the decision tree using graphviz.

        Parameters:
        tree: The tree structure, default is self.tree.
        feature_names (list): Names of the features for display.
        class_names (list): Names of the classes for display.
        filename (str): The name of the output file (without extension).

        Returns:
        None: Saves the visualization as a PNG file.
        """
        if tree is None:
            tree = self.tree

        dot = Digraph()
        self._add_nodes(dot, tree, feature_names, class_names, 0)
        dot.render(filename, format='png', cleanup=True)
        print(f"Decision tree visualization saved as {filename}.png")

    def _add_nodes(self, dot, tree, feature_names, class_names, node_id):
        """
        Recursively adds nodes and edges to the graph.

        Parameters:
        dot (Digraph): The graphviz Digraph object.
        tree: The tree structure.
        feature_names (list): Names of the features for display.
        class_names (list): Names of the classes for display.
        node_id (int): Unique identifier for the current node.

        Returns:
        None
        """
        if isinstance(tree, tuple):
            feature, threshold, left, right = tree
            feature_name = feature_names[feature] if feature_names else f"Feature {feature}"
            dot.node(str(node_id), f"{feature_name} < {threshold:.2f}")
            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2
            dot.edge(str(node_id), str(left_child_id), label="True")
            dot.edge(str(node_id), str(right_child_id), label="False")
            self._add_nodes(dot, left, feature_names, class_names, left_child_id)
            self._add_nodes(dot, right, feature_names, class_names, right_child_id)
        else:
            class_name = class_names[int(tree)] if class_names else f"Class {int(tree)}"
            dot.node(str(node_id), class_name)


# PyTorch implementation 

import torch
from graphviz import Digraph

# decision_tree_pytorch.py

import torch
from graphviz import Digraph

class DecisionTreeClassifierPyTorch:
    """Decision Tree Classifier for binary classification using PyTorch."""

    def __init__(self, 
                 criterion='gini', 
                 max_depth=None, 
                 min_samples_split=2, 
                 min_samples_leaf=1, 
                 random_state=None, 
                 device=None):
        """
        Initializes the Decision Tree Classifier.

        Parameters:
        criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        random_state (int): Controls the randomness of the estimator.
        device (torch.device or str): The device to run the model on ('cpu' or 'cuda').
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
    
    def fit(self, X, y):
        """
        Fits the model to the training data.

        Parameters:
        X (torch.Tensor): Feature data of shape (n_samples, n_features).
        y (torch.Tensor): Target labels of shape (n_samples,).
        """
        # Ensure X and y are on the same device as the model
        X = X.to(self.device)
        y = y.to(self.device)
        self.tree = self._grow_tree(X, y)
    
    def predict(self, X):
        """
        Predicts the class labels for the input samples.

        Parameters:
        X (torch.Tensor): Feature data of shape (n_samples, n_features).

        Returns:
        torch.Tensor: Predicted class labels of shape (n_samples,).
        """
        # Ensure X is on the same device as the model
        X = X.to(self.device)
        predictions = []
        for sample in X:
            pred = self._predict_sample(sample, self.tree)
            predictions.append(pred)
        return torch.tensor(predictions, dtype=torch.long, device=self.device)
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the tree.

        Parameters:
        X (torch.Tensor): Feature data at the current node.
        y (torch.Tensor): Target labels at the current node.
        depth (int): Current depth of the tree.

        Returns:
        tuple or int: Returns a leaf value (class label) or a decision node 
                      (feature, threshold, left subtree, right subtree).
        """
        num_samples, num_features = X.size()
        unique_classes = y.unique()

        # Stopping criteria
        if (len(unique_classes) == 1 or
                (self.max_depth is not None and depth >= self.max_depth) or
                num_samples < self.min_samples_split):
            return y.mode().values.item()

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return y.mode().values.item()

        # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Check for minimum samples in leaves
        if left_indices.sum().item() < self.min_samples_leaf or right_indices.sum().item() < self.min_samples_leaf:
            return y.mode().values.item()

        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)
    
    def _best_split(self, X, y):
        """
        Finds the best feature and threshold to split on.

        Parameters:
        X (torch.Tensor): Feature data.
        y (torch.Tensor): Target labels.

        Returns:
        tuple: Best feature index and best threshold for splitting.
        """
        num_samples, num_features = X.size()
        best_feature, best_threshold = None, None
        best_gain = float('-inf')

        for feature in range(num_features):
            # Sort the data based on the current feature
            sorted_indices = torch.argsort(X[:, feature])
            thresholds = X[sorted_indices, feature]
            classes = y[sorted_indices]

            num_unique_classes = len(classes.unique())
            num_left = torch.zeros(num_unique_classes, dtype=torch.float32, device=self.device)
            num_right = torch.bincount(classes, minlength=num_unique_classes).float().to(self.device)

            for i in range(1, num_samples):
                class_index = classes[i - 1].item()
                num_left[class_index] += 1
                num_right[class_index] -= 1

                # Skip if the current threshold is the same as the previous one
                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Calculate information gain
                gain = self._information_gain(num_left, num_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_feature, best_threshold
    
    def _information_gain(self, num_left, num_right):
        """
        Calculates the information gain from a split based on the chosen criterion.

        Parameters:
        num_left (torch.Tensor): Counts of classes in the left split.
        num_right (torch.Tensor): Counts of classes in the right split.

        Returns:
        float: The information gain from the split.
        """
        total = num_left.sum() + num_right.sum()
        if total == 0:
            return 0

        # Calculate parent impurity
        parent_counts = num_left + num_right
        if self.criterion == 'gini':
            parent_impurity = self._gini(parent_counts)
            left_impurity = self._gini(num_left)
            right_impurity = self._gini(num_right)
        elif self.criterion == 'entropy':
            parent_impurity = self._entropy(parent_counts)
            left_impurity = self._entropy(num_left)
            right_impurity = self._entropy(num_right)
        else:
            raise ValueError("Invalid criterion. Supported criteria: 'gini', 'entropy'.")

        # Calculate weighted impurity after split
        weighted_impurity = (num_left.sum() / total) * left_impurity + (num_right.sum() / total) * right_impurity

        # Information gain is reduction in impurity
        gain = parent_impurity - weighted_impurity
        return gain
    
    def _gini(self, counts):
        """
        Calculates the Gini impurity.

        Parameters:
        counts (torch.Tensor): Counts of each class in the node.

        Returns:
        float: The Gini impurity.
        """
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - torch.sum(p ** 2).item()
    
    def _entropy(self, counts):
        """
        Calculates the entropy.

        Parameters:
        counts (torch.Tensor): Counts of each class in the node.

        Returns:
        float: The entropy.
        """
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        # Adding a small value to avoid log(0)
        entropy = -torch.sum(p * torch.log2(p + 1e-9)).item()
        return entropy
    
    def _predict_sample(self, sample, tree):
        """
        Predicts the class label for a single sample.

        Parameters:
        sample (torch.Tensor): A single sample.
        tree: The decision tree structure.

        Returns:
        int: Predicted class label.
        """
        if isinstance(tree, tuple):
            feature, threshold, left, right = tree
            if sample[feature] < threshold:
                return self._predict_sample(sample, left)
            else:
                return self._predict_sample(sample, right)
        else:
            return tree
    
    def visualize(self, tree=None, feature_names=None, class_names=None, filename="decision_tree_pytorch"):
        """
        Visualizes the decision tree using graphviz.

        Parameters:
        tree: The tree structure, default is self.tree.
        feature_names (list): Names of the features for display.
        class_names (list): Names of the classes for display.
        filename (str): The name of the output file (without extension).

        Returns:
        None: Saves the visualization as a PNG file.
        """
        if tree is None:
            tree = self.tree

        dot = Digraph()
        self._add_nodes(dot, tree, feature_names, class_names, 0)
        dot.render(filename, format='png', cleanup=True)
        print(f"Decision tree visualization saved as {filename}.png")
    
    def _add_nodes(self, dot, tree, feature_names, class_names, node_id):
        """
        Recursively adds nodes and edges to the graph.

        Parameters:
        dot (Digraph): The graphviz Digraph object.
        tree: The tree structure.
        feature_names (list): Names of the features for display.
        class_names (list): Names of the classes for display.
        node_id (int): Unique identifier for the current node.

        Returns:
        None
        """
        if isinstance(tree, tuple):
            feature, threshold, left, right = tree
            feature_name = feature_names[feature] if feature_names else f"Feature {feature}"
            dot.node(str(node_id), f"{feature_name} < {threshold:.2f}")
            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2
            dot.edge(str(node_id), str(left_child_id), label="True")
            dot.edge(str(node_id), str(right_child_id), label="False")
            self._add_nodes(dot, left, feature_names, class_names, left_child_id)
            self._add_nodes(dot, right, feature_names, class_names, right_child_id)
        else:
            class_name = class_names[int(tree)] if class_names else f"Class {int(tree)}"
            dot.node(str(node_id), class_name)
