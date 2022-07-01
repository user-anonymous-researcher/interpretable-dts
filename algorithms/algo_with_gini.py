"""
    SER-DT Algorithm but considering Gini as impurity measure
"""
import math
import numpy as np

from algorithms.algo import Algo


class AlgoWithGini(Algo):
    def __init__(self, max_depth=None, min_samples_stop=0):
        super().__init__(max_depth, min_samples_stop)

    def _get_best_threshold(self, X, y, a, classes_parent, node_pairs, node_product, gamma_factor):
        """
            Get the best threshold given an attribute.
        """
        m = y.size  # tirar
        thresholds, classes_thr = zip(*sorted(zip(X[:, a], y)))
        classes_left = [0] * self.n_classes_
        classes_right = classes_parent.copy()
        pairs_left = 0
        pairs_right = node_pairs
        t_best_impurity = None
        best_impurity = math.inf
        all_thresholds = []

        # For the 2 case split
        cost_star = self._get_attr_cost(thresholds, classes_thr)
        # Threshold that characterizes the smallest threshold of a binary split such that the left part has larger
        # pair-by-weight product thant node_product
        t_star = None

        for i in range(1, m):
            node_class = classes_thr[i - 1]
            classes_left[node_class] += 1
            classes_right[node_class] -= 1
            pairs_left += sum([classes_left[j] for j in range(self.n_classes_) if j != node_class])
            pairs_right -= sum([classes_right[j] for j in range(self.n_classes_) if j != node_class])
            if thresholds[i] == thresholds[i - 1]:
                continue
            threshold = (thresholds[i] + thresholds[i - 1]) / 2
            all_thresholds.append(threshold)
            prod_left = i * pairs_left
            prod_right = (m - i) * pairs_right
            cost = max(prod_left, prod_right)

            impurity_left = 1.0 - sum(
                (classes_left[x] / i) ** 2 for x in range(self.n_classes_)
            )
            impurity_right = 1.0 - sum(
                (classes_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
            )
            impurity = (i * impurity_left + (m - i) * impurity_right) / m

            if t_star is None and prod_left > gamma_factor * node_product:
                t_star = threshold
            if impurity < best_impurity and cost <= gamma_factor * node_product:
                best_impurity = impurity
                t_best_impurity = threshold

        cost_star = math.inf if t_star is None else cost_star
        return t_best_impurity, best_impurity, t_star, cost_star, all_thresholds

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1, gamma_factor=0.5,
                    father_feature=None):
        """
        Find the next split for a node.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
            next_thr: If 2-split is necessary, returns the
            balanced: Whether the split is balanced in terms of cost (product of pairs and weight).
                      In this case, 2-split is required
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None, None, None

        node_pairs = self._number_pairs(y)
        node_product = node_pairs * y.size
        classes_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        impurity_parent = 1.0 - sum((n / m) ** 2 for n in classes_parent)
        best_modified_impurity = math.inf

        # variables for the 2-step partition
        a_min_impurity = None  # attribute that minimizes Gini and satisfies cost <= 0.5 * node_product
        t_min_impurity = None  # threshold relative to previous attribute

        # variables for the 3-step partition
        a_star = None
        t_star = None  # t* - threshold relative to previous attribute
        cost_attr_min = math.inf
        all_thresholds = []
        has_balanced = False

        # Loop through all features.
        for a in range(self.n_features_):
            threshold_a_balanced, impurity_a_balanced, threshold_a_star, cost_a_star, thresholds_a = \
                self._get_best_threshold(X, y, a, classes_parent, node_pairs, node_product, gamma_factor)
            modified_impurity_a_balanced = impurity_a_balanced * modified_factor if feature_index_occurrences[a] else \
                impurity_a_balanced

            all_thresholds.append(thresholds_a)
            if threshold_a_balanced is not None:
                has_balanced = True
            if cost_a_star < cost_attr_min:
                a_star = a
                t_star = threshold_a_star
                cost_attr_min = cost_a_star
            if threshold_a_balanced is not None and impurity_a_balanced < impurity_parent and \
                    modified_impurity_a_balanced < best_modified_impurity:
                a_min_impurity = a
                t_min_impurity = threshold_a_balanced
                best_modified_impurity = modified_impurity_a_balanced

        if has_balanced and a_min_impurity is not None:
            return a_min_impurity, t_min_impurity, None, True
        elif has_balanced:
            return None, None, None, None

        # Else, Perform a 2-step partition, there's no balanced partition
        if t_star is not None:
            thresholds = all_thresholds[a_star]
            t_index = thresholds.index(t_star)
            if t_index > 0:
                return a_star, t_star, thresholds[t_index - 1], False
            else:
                return a_star, t_star, None, False
        else:
            return None, None, None, None
