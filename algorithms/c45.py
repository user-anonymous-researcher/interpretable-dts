"""
    Implementation of the C4.5 algorithm to train decision tree classifiers.
"""

import math
import numpy as np

from algorithms.cart import Cart


class C45(Cart):
    def __init__(self, max_depth=None, min_samples_stop=0):
        super().__init__(max_depth, min_samples_stop)

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1, father_feature=None):
        """Find the best split for a node.

        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.

        To find the best split, we loop through all the features, and consider all the
        midpoints between adjacent training samples as possible thresholds. We compute
        the Gini impurity of the split generated by that particular feature/threshold
        pair, and return the pair with smallest impurity.

        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        impurity_parent = - sum((n / m) * math.log2(n / m) for n in num_parent if n)
        best_impurity = math.inf
        # best_gain_ratio = 0

        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                impurity_left = - sum(
                    (num_left[x] / i) * math.log2(num_left[x] / i) for x in range(self.n_classes_)
                    if num_left[x]
                )
                impurity_right = - sum(
                    (num_right[x] / (m - i)) * math.log2(num_right[x] / (m - i)) for x in range(self.n_classes_)
                    if num_right[x]
                )

                # impurity of a split is the weighted average of the impurity of the children.
                impurity = (i * impurity_left + (m - i) * impurity_right) / m
                # split_info = - (i / m) * math.log2(i / m) - ((m - i) / m) * math.log2((m - i) / m)
                # gain_ratio = (impurity_parent - impurity) / split_info

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr
