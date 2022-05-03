import numpy as np
import tree.tree as tree


class DefaultClassifier:
    def __init__(self, max_depth=None, min_samples_stop=0):
        self.max_depth = max_depth
        self.min_samples_stop = min_samples_stop
        self.n_classes_ = None
        self.n_samples = None
        self.n_features_ = None
        self.tree_: tree = None

    def fit(self, X, y, modified_factor=1, gamma_factor=None, pruning=False):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_samples = len(y)
        self.n_features_ = X.shape[1]
        feature_index_occurrences = [0] * self.n_features_
        self.tree_ = self._grow_tree(X, y, feature_index_occurrences=feature_index_occurrences,
                                     modified_factor=modified_factor, gamma_factor=gamma_factor)
        if pruning:
            self.tree_ = tree.Node.get_pruned_tree(self.tree_)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def score(self, X, y):
        correct = 0
        predicted_labels = self.predict(X)
        for i in range(len(y)):
            if predicted_labels[i] == y[i]:
                correct += 1
        return correct / len(y)

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    def _gini(self, y):
        """Compute Gini impurity of a non-empty node.

        Gini impurity is defined as Σ p(1-p) over all classes, with p the frequency of a
        class within the node. Since Σ p = 1, this is equivalent to 1 - Σ p^2.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def get_score(self, X_train, y_train, X_test, y_test, modified_factor=1, gamma_factor=None, pruning=False):
        self.fit(X_train, y_train, modified_factor=modified_factor, gamma_factor=gamma_factor, pruning=pruning)
        return round(self.score(X_train, y_train), 3), round(self.score(X_test, y_test), 3)

    def get_score_without_fit(self, X_train, y_train, X_test, y_test):
        return round(self.score(X_train, y_train), 3), round(self.score(X_test, y_test), 3)

    def get_explainability_metrics(self):
        # Returns unbalanced_splits, max_depth, max_depth_redundant, wad, waes, nodes, distinct features
        return self.tree_.get_explainability_metrics(self.n_features_)

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1):
        raise NotImplementedError()

    def _grow_tree(self, X, y, depth=0, feature_index_occurrences=None, modified_factor=1, calculate_gini=True,
                   gamma_factor=None):
        raise NotImplementedError()
