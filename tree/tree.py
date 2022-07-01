"""
    Binary tree with decision tree semantics and ASCII visualization.
"""
import copy
import dot2tex as d2t
from pathlib import Path
import pydot
import subprocess

from utils.random import COLOR_LIST


class Node:
    """A decision tree node."""

    def __init__(self, num_samples, num_samples_per_class, predicted_class, gini=0, feature_index=None, threshold=None,
                 feature_index_occurrences=None, balanced_split=None):
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.gini = gini
        self.feature_index = feature_index
        self.feature_index_occurrences = feature_index_occurrences  # number of occurrences for each feature

        # for each feature, +1 if feature x_i appears in x_i > j,
        # -1 if x_i appears in x_i <= j, and
        # 2 if x_if appears in both <= j and > j,
        # 0 otherwise
        self.feature_index_occurrences_redundant = None

        self.threshold = threshold

        # Whether the node split is balanced in terms of cost (product of pairs and weight)
        self.balanced_split = balanced_split

        self.left = None
        self.right = None

        self.pruned_class = None  # For pruning

    def get_node_depth(self):
        """
            Get node depth
        """
        # Assert that get_explainability_metrics was previously run
        return sum(x for x in self.feature_index_occurrences)

    def get_node_explainability(self):
        """
            Get node explability: Number of distinct features in path to leaf
        """
        # Path to node less the redundant nodes
        # Assert that get_explainability_metrics was previously run
        # return sum(abs(x) for x in self.feature_index_occurrences_redundant)

        # Now node explainability is the number of distinct features in path to leaf
        return sum([1 for i in self.feature_index_occurrences if i])

    def get_next_feature_occurrences_redundant(self, idx, direction):
        """
            Get next feature redundant ocurrences given attribute and direction (left or right)
        """
        if direction != 'right' and direction != 'left':
            raise ValueError("Value of direction not permitted: ", direction)

        new_feature_list = self.feature_index_occurrences_redundant.copy()
        if self.feature_index_occurrences_redundant[idx] == 0:
            if direction == 'right':
                new_feature_list[idx] = 1
            else:
                new_feature_list[idx] = -1
        elif self.feature_index_occurrences_redundant[idx] == 1:
            if direction == 'right':
                new_feature_list[idx] = 1
            else:
                new_feature_list[idx] = 2
        elif self.feature_index_occurrences_redundant[idx] == -1:
            if direction == 'right':
                new_feature_list[idx] = 2
            else:
                new_feature_list[idx] = -1
        elif self.feature_index_occurrences_redundant[idx] == 2:
            # Do nothing +2 is the maximum
            pass
        else:
            raise ValueError("Some value not permitted on feature index redundant")
        return new_feature_list

    def debug(self, feature_names, class_names, show_details):
        """Print an ASCII visualization of the tree."""
        lines, _, _, _ = self._debug_aux(
            feature_names, class_names, show_details, root=True
        )
        for line in lines:
            print(line)

    def _debug_aux(self, feature_names, class_names, show_details, root=False):
        """
            Auxiliary funcion to print ASCII visualization of the tree.
        """
        # See https://stackoverflow.com/a/54074933/1143396 for similar code.
        is_leaf = not self.right
        if is_leaf:
            lines = [class_names[self.predicted_class]]
        else:
            lines = [
                "{} < {:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "feature_occurrences = {}".format(self.feature_index_occurrences),
                "feat_redund_occurr = {}".format(self.feature_index_occurrences_redundant),
                "gini = {:.2f}".format(self.gini),
                "balanced_split = {}".format(self.balanced_split),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1:]
            return lines, width, height, middle

        # If not a leaf, must have two children.
        left, n, p, x = self.left._debug_aux(feature_names, class_names, show_details)
        right, m, q, y = self.right._debug_aux(feature_names, class_names, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
                top_lines
                + [middle_line, bottom_line]
                + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1:]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle

    def debug_pydot(self, output_file: str):
        """
            Print tree with latex format
        """
        pydot_graph = pydot.Dot("tree", graph_type="digraph", dpi=300)

        def get_node_name(node):
            return f"{node.get_node_depth()}.{node.num_samples}.{node.feature_index}.{node.threshold}." \
                   f"{node.feature_index_occurrences}.{node.feature_index_occurrences_redundant}"

        def get_pydot_node(node):
            node_name = get_node_name(node)
            # fill_color = "none" if not node.left and not node.right else COLOR_LIST[node.feature_index]
            fill_color = "gold" if not node.left and not node.right else "none"

            threshold = str(round(node.threshold, 3) + 0) if node.left or node.right else None
            # Scientific values for small numbers
            if node.threshold is not None and abs(node.threshold) < 0.001:
                # threshold = "{:.2e}".format(node.threshold).replace("e-0", "e-")
                threshold = "0.00"
            elif node.threshold is not None:
                threshold = str(round(node.threshold, 2))
            tex_label = f"$D{node.feature_index} \\leq " + threshold + "$" if node.left or node.right \
                else "$\\begin{matrix}" + "\\text{Samples: }" + str(node.num_samples) + "\\\\" + \
                     "\\text{Class: }" + str(node.predicted_class) + "\\end{matrix}$"
            return pydot.Node(node_name, shape="box", fillcolor=fill_color, style="filled", texlbl=tex_label,
                              align="left", width=1.0)

        def visit_node(pydot_tree: pydot.Dot, node: Node, parent_node_name: str = "none"):
            node_name = get_node_name(node)

            if parent_node_name != "none":
                pydot_tree.add_edge(pydot.Edge(parent_node_name, node_name))

            if node.left:
                pydot_left_node = get_pydot_node(node.left)
                pydot_tree.add_node(pydot_left_node)
                visit_node(pydot_tree, node.left, node_name)
            if node.right:
                pydot_right_node = get_pydot_node(node.right)
                pydot_tree.add_node(pydot_right_node)
                visit_node(pydot_tree, node.right, node_name)

        # Call recursive function
        pydot_root_node = get_pydot_node(self)
        pydot_graph.add_node(pydot_root_node)
        visit_node(pydot_graph, self)

        # Save Results
        # Create dir if not exists
        dir_index = output_file.rfind("/")
        dir_name = output_file[:dir_index]
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        texcode = d2t.dot2tex(pydot_graph.to_string(), crop=True, autosize=True)
        # pydot_graph.write_png(output_file)
        output_file_tex = output_file + ".tex"
        with open(output_file_tex, 'w') as file:
            file.write(texcode)
        subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory", dir_name,
                        output_file_tex])
        subprocess.run(["convert", "-density", "300", f"{output_file}.pdf", "-quality", "100",
                        output_file])

    def get_explainability_metrics(self, num_features):
        """
            Get explainability metrics.
            Returns:
            number of unbalanced nodes, max_depth, max_redundant_depth, wad, waes, nodes and distinct_features

        """
        wad_by_node = []  # Weighted Path by leaf. List of pairs (depth, num_samples)
        waes_by_node = []  # Same as wad_by_node but discarding redundant features
        nodes_number_metric = [0]
        features_distinct_metric = [0] * num_features
        unbalanced_splits = [0]

        # Recursive function
        def visit_node(node: Node, _feature_index_occurs, _feature_index_redundant_occurs, nodes_number,
                       features_distinct, unbal_splits):

            # Copy lists from father
            node.feature_index_occurrences = _feature_index_occurs.copy()
            node.feature_index_occurrences_redundant = _feature_index_redundant_occurs.copy()

            # Update metrics
            if node.left is not None and node.right is not None:
                node.feature_index_occurrences[node.feature_index] += 1
            nodes_number[0] += 1
            if (node.left or node.right) and not features_distinct[node.feature_index]:
                features_distinct[node.feature_index] += 1
            unbal_splits[0] += 1 if node.balanced_split is False else 0

            # Recurse
            left_node: Node = node.left
            right_node: Node = node.right
            if not left_node and not right_node:
                # leaf
                depth = node.get_node_depth()
                redundant_depth = node.get_node_explainability()
                num_samples = node.num_samples
                wad_by_node.append((depth, num_samples))
                waes_by_node.append((redundant_depth, num_samples))

            if left_node:
                next_index_occurs = node.get_next_feature_occurrences_redundant(node.feature_index, 'left')
                left_node.feature_index_occurrences_redundant = next_index_occurs
                visit_node(left_node, node.feature_index_occurrences.copy(), next_index_occurs.copy(),
                           nodes_number, features_distinct, unbal_splits)
            if right_node:
                next_index_occurs = node.get_next_feature_occurrences_redundant(node.feature_index, 'right')
                right_node.feature_index_occurrences_redundant = next_index_occurs
                visit_node(right_node, node.feature_index_occurrences.copy(), next_index_occurs.copy(),
                           nodes_number, features_distinct, unbal_splits)

        # Call recursive function
        self.feature_index_occurrences = [0] * num_features
        self.feature_index_occurrences_redundant = [0] * num_features
        visit_node(self, self.feature_index_occurrences.copy(), self.feature_index_occurrences_redundant.copy(),
                   nodes_number_metric, features_distinct_metric, unbalanced_splits)

        # Calculate metrics
        max_depth = max(x[0] for x in wad_by_node)
        max_redundant_depth = max(x[0] for x in waes_by_node)

        total_samples = sum(x[1] for x in wad_by_node)
        wad_sum = 0
        for p in wad_by_node:
            wad_sum += p[0] * p[1]
        wad = wad_sum / total_samples
        wad = round(wad, 3)

        waes_sum = 0
        for p in waes_by_node:
            waes_sum += p[0] * p[1]
        waes = waes_sum / total_samples
        waes = round(waes, 3)

        nodes = nodes_number_metric[0]
        unbalanced = unbalanced_splits[0]
        distinct_features = sum(features_distinct_metric)
        return unbalanced, max_depth, max_redundant_depth, wad, waes, nodes, distinct_features

    def get_pruned_tree(self):
        """
            Returns pruned tree
        """

        def get_node_class(node):
            """
            Returns -1 if there is more than one class attributed in the subtree induced by node
            Returns the class if there is only one such class.
            """
            if not node.left and not node.right:
                node.pruned_class = node.predicted_class
                return node.predicted_class

            left_class = None
            right_class = None
            if node.left:
                left_class = get_node_class(node.left)
            if node.right:
                right_class = get_node_class(node.right)

            if left_class == right_class:
                node.pruned_class = left_class
            else:
                node.pruned_class = -1
            return node.pruned_class

        # Get pruned class info
        new_tree = copy.deepcopy(self)
        get_node_class(new_tree)

        # Prune nodes
        def prune_node(node):
            if node.pruned_class != -1:
                node.left = None
                node.right = None
            if node.left:
                prune_node(node.left)
            if node.right:
                prune_node(node.right)

        prune_node(new_tree)
        return new_tree
