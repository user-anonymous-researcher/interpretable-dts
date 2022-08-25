"""
    Base Algorithm to derive others
"""
import copy

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
        self.score_post_pruning = None  # For post-pruning

    def fit(self, X, y, modified_factor=1, gamma_factor=None, pruning=True, post_pruning=False, X_val=None, y_val=None):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_samples = len(y)
        self.n_features_ = X.shape[1]
        feature_index_occurrences = [0] * self.n_features_
        self.tree_ = self._grow_tree(X, y, feature_index_occurrences=feature_index_occurrences,
                                     modified_factor=modified_factor, gamma_factor=gamma_factor)
        if pruning:
            self.tree_ = tree.Node.get_pruned_tree(self.tree_)
        if post_pruning:
            self.tree_ = self.get_post_pruned_tree(X_val, y_val)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def score(self, X, y):
        """ Calculates score accuracy """
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

    def get_score(self, X_train, y_train, X_test, y_test, modified_factor=1, gamma_factor=None, pruning=True,
                  post_pruning=False, X_val=None, y_val=None):
        """
        Get score accuracy
        """
        self.fit(X_train, y_train, modified_factor=modified_factor, gamma_factor=gamma_factor, pruning=pruning,
                 post_pruning=post_pruning, X_val=X_val, y_val=y_val)
        return round(self.score(X_train, y_train), 3), round(self.score(X_test, y_test), 3)

    def get_score_without_fit(self, X_train, y_train, X_test, y_test):
        """
        Get score without fitting
        """
        if X_train is not None and y_train is not None:
            return round(self.score(X_train, y_train), 3), \
                   round(self.score(X_test, y_test), 3)
        else:
            return None, round(self.score(X_test, y_test), 3)

    def get_explainability_metrics(self):
        """
        Get exaplanaibility metrics
        """
        # Returns unbalanced_splits, max_depth, max_depth_redundant, wad, waes, nodes, distinct features
        return self.tree_.get_explainability_metrics(self.n_features_)

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1):
        """
         Find the next split for a node.
        """
        raise NotImplementedError()

    def _grow_tree(self, X, y, depth=0, feature_index_occurrences=None, modified_factor=1, calculate_gini=True,
                   gamma_factor=None):
        """
            Build a decision tree by recursively finding the best split.
        """
        raise NotImplementedError()

    def get_post_pruned_tree(self, X_val, y_val):
        """
            Returns post pruned tree
        """

        _, self.score_post_pruning = self.get_score_without_fit(None, None, X_val, y_val)

        def get_node_pruned(node):
            """
            Returns true if it is better (in accuracy) to cut off subtree induced by node,
            false otherwise
            """
            if not node.left and not node.right:
                node.post_pruning = True
                return True

            left_pruned = None
            right_pruned = None
            if node.left:
                left_pruned = get_node_pruned(node.left)
            if node.right:
                right_pruned = get_node_pruned(node.right)

            if left_pruned and right_pruned:
                # Try join children and see whether accuracy increases
                node_left = copy.deepcopy(node.left)
                node_right = copy.deepcopy(node.right)
                node.left = None
                node.right = None
                _, new_score = self.get_score_without_fit(None, None, X_val, y_val)
                if new_score >= self.score_post_pruning:
                    node.post_pruning = True
                    self.score_post_pruning = new_score
                else:
                    # Revert Changes
                    node.left = node_left
                    node.right = node_right
                    node.post_pruning = False
            else:
                node.post_pruning = False
            return node.post_pruning

        # Get pruned class info
        get_node_pruned(self.tree_)

        # Prune nodes
        def prune_node(node):
            if node.post_pruning:
                node.left = None
                node.right = None
            if node.left:
                prune_node(node.left)
            if node.right:
                prune_node(node.right)

        prune_node(self.tree_)
        return self.tree_
