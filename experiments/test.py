"""
    Base Test Class with utilities

"""
from enum import Enum
import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from time import perf_counter_ns

from algorithms.algo_with_gini import AlgoWithGini
from algorithms.cart import Cart
from tree.tree import Node
from utils.file_utils import create_dir


class MetricType(Enum):
    """
        Possible metrics for a test
    """
    gini_factor = "gini_factor"
    gamma_factor = "gamma"
    train_accuracy = "train_accuracy"
    test_accuracy = "test_accuracy"
    max_depth = "max_depth"
    max_depth_redundant = "max_depth_redundant"
    wad = "wad"
    waes = "waes"
    unbalanced_splits = "unbalanced_splits"
    nodes = "nodes"
    features = "features"
    execution_time = "execution_time"


class Metrics:
    def __init__(self, gini_factor, gamma_factor, train_accuracy, test_accuracy, max_depth, max_depth_redundant, wad,
                 waes, unbalanced_splits, nodes, features, execution_time):
        self.gini_factor = gini_factor
        self.gamma_factor = gamma_factor
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.unbalanced_splits = unbalanced_splits
        self.max_depth = max_depth
        self.max_depth_redundant = max_depth_redundant
        self.wad = wad
        self.waes = waes
        self.nodes = nodes
        self.features = features
        self.execution_time = execution_time

    def get_metrics(self):
        """
            Get metrics for the test
        """
        return {
            MetricType.gini_factor: self.gini_factor,
            MetricType.gamma_factor: self.gamma_factor,
            MetricType.train_accuracy: round(self.train_accuracy, 3),
            MetricType.test_accuracy: round(self.test_accuracy, 3),
            MetricType.max_depth: round(self.max_depth, 3),
            MetricType.max_depth_redundant: round(self.max_depth_redundant, 3),
            MetricType.wad: round(self.wad, 3),
            MetricType.waes: round(self.waes, 3),
            MetricType.unbalanced_splits: self.unbalanced_splits,
            MetricType.nodes: self.nodes,
            MetricType.features: self.features,
            MetricType.execution_time: self.execution_time
        }

    @staticmethod
    def get_cols():
        """
            Return names of metrics
        """
        return [MetricType.gini_factor.value,
                MetricType.gamma_factor.value,
                MetricType.train_accuracy.value,
                MetricType.test_accuracy.value,
                MetricType.unbalanced_splits.value,
                MetricType.max_depth.value,
                MetricType.max_depth_redundant.value,
                MetricType.wad.value,
                MetricType.waes.value,
                MetricType.nodes.value,
                MetricType.features.value,
                MetricType.execution_time.value]


class ResultJson:
    """Class to manage json result"""

    def __init__(self, dataset, algorithm, max_depth_stop, min_samples_stop, results):
        self.dataset = dataset
        self.algorithm = algorithm
        self.max_depth_stop = max_depth_stop
        self.min_samples_stop = min_samples_stop
        self.results = results

    @staticmethod
    def get_cols():
        """Get columns that appear in json result"""
        return ["dataset", "algorithm", "max_depth_stop", "min_samples_stop"]

    def get_result_json(self):
        return {
            'dataset': self.dataset,
            'algorithm': self.algorithm,
            'max_depth_stop': self.max_depth_stop,
            'min_samples_stop': self.min_samples_stop,
            # 'opt_factor': self._get_opt_factor(),
            'results': [{key.value: metric[key] for key in metric} for metric in self.results]  # Convert to string keys
        }


class Test:
    """
        Base class to run and save tests
    """

    def __init__(self, classifier, dataset_name: str, csv_file: str, max_depth_stop,
                 col_class_name: str, cols_to_delete: list = None, min_samples_stop: int = 0,
                 gini_factors=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                               0.97, 0.98, 0.99, 1), gamma_factors=(0.5, 0.6, 0.7, 0.8, 0.9), results_folder="results"):
        self.classifier = classifier
        self.clf_obj = None  # in case of pickle
        self.dataset_name = dataset_name
        self.csv_file = csv_file
        self.col_class_name = col_class_name
        self.cols_to_delete = cols_to_delete
        self.max_depth_stop = max_depth_stop
        self.min_samples_stop = min_samples_stop
        self.results_folder = results_folder
        self.gini_factors = gini_factors
        self.gamma_factors = gamma_factors
        self.n_classes = None  # Need to update via run()
        self.n_samples = None  # Need to update via run()
        self.n_features = None  # Need to update via run()
        self.results = None  # Need to update via run()

    def _get_filename(self, extension: str, gini_factor: float = None, gamma_factor: float = None,
                      sub_folder=None, iteration=None) -> str:
        """
            Get filename to save result
        """
        folder = f"{self.results_folder}/{sub_folder}" if sub_folder else f"{self.results_folder}"
        filename = f"{folder}/{self.dataset_name}_{self.classifier.__name__}_depth_{self.max_depth_stop}" \
                   f"_samples_{self.min_samples_stop}"
        if gini_factor:
            filename += f"_gini-factor_{gini_factor}"
        if gamma_factor:
            filename += f"_gamma_{round(gamma_factor, 2)}"
        if iteration:
            filename += f"_iteration_{iteration}"

        filename += f".{extension}"
        return filename

    def _plot_graphic(self):
        """
            Plot graphic for tests
        """
        factors = [metric[MetricType.gini_factor] for metric in self.results]
        train_accuracy_list = [metric[MetricType.train_accuracy] for metric in self.results]
        test_accuracy_list = [metric[MetricType.test_accuracy] for metric in self.results]
        max_depth_list = [metric[MetricType.max_depth] for metric in self.results]
        max_depth_redundant_list = [metric[MetricType.max_depth_redundant] for metric in self.results]
        wad_list = [metric[MetricType.wad] for metric in self.results]
        waes_list = [metric[MetricType.waes] for metric in self.results]

        figure(figsize=(12, 10), dpi=300)
        plt.subplot(2, 1, 1)
        dataset_name = self.dataset_name
        plt.title(dataset_name, fontsize=16)
        plt.plot(factors, train_accuracy_list, label="Train Accuracy", color='red', marker='o')
        plt.plot(factors, test_accuracy_list, label="Test Accuracy", color='darkred', marker='o')
        plt.ylabel("Accuracy", fontsize=16)
        plt.legend(loc="lower right", fontsize=10)

        plt.subplot(2, 1, 2)
        plt.plot(factors, max_depth_list, label="Max Depth", color='cyan', marker='o')
        plt.plot(factors, max_depth_redundant_list, label="Redundant Max Depth", color='blue', marker='o')
        plt.plot(factors, wad_list, label="WAD (Weighted Average Depth", color='gray', marker='o')
        plt.plot(factors, waes_list, label="WAES (Weighted Average Explanation Size)", color='black', marker='o')
        plt.xlabel("Gini Factor", fontsize=16)
        plt.ylabel("Metric", fontsize=16)
        plt.legend(loc="lower right", fontsize=10)
        filename = self._get_filename("png", sub_folder="img")
        plt.savefig(filename)

    def _store_results(self, iteration=0):
        """
            Store results in json for iteration
        """
        filename: str = self._get_filename("json", sub_folder="json", iteration=iteration)
        result_json = {
            'dataset': self.dataset_name,
            'algorithm': self.classifier.__name__,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'max_depth_stop': self.max_depth_stop,
            'min_samples_stop': self.min_samples_stop,
            # 'opt_factor': self._get_opt_factor(),
            'results': [{key.value: metric[key] for key in metric} for metric in self.results]  # Convert to string keys
        }
        # Create dir if not exists
        create_dir(filename)
        with open(filename, 'w') as f:
            json.dump(result_json, f, indent=2)

    def _get_opt_factor(self, opt_factor=0.90):
        """
        Get the lower factor that allow a decreasing in test_accuracy not lower than opt_factor.
        Also, get the diff compared to original values (factor=1) in percentage
        """
        # Assume that results are ordered by increasing factors
        reversed_results = sorted(self.results, key=lambda x: x[MetricType.gini_factor], reverse=True)
        max_test_accuracy = reversed_results[0][MetricType.test_accuracy]

        max_metrics = reversed_results[0]

        opt_index = -1
        for i in range(1, len(reversed_results)):
            if reversed_results[i][MetricType.test_accuracy] <= opt_factor * max_test_accuracy:
                opt_index = i - 1
                break
        opt_metrics = reversed_results[opt_index]

        diff_results = dict()
        for key in max_metrics:
            if key == MetricType.gini_factor:
                diff_results[key.value] = opt_metrics[key]
            else:
                diff_metric = round(100 * (opt_metrics[key] - max_metrics[key]) / max_metrics[key])
                diff_results[key.value] = str(diff_metric) + "%"

        return diff_results

    def parse_dataset(self):
        """
            Read CSV and clean dataset
        """
        # Read CSV
        np.set_printoptions(suppress=True)
        data = pd.read_csv(self.csv_file)
        if self.cols_to_delete:
            data = data.drop(labels=self.cols_to_delete, axis=1)
        data = data.astype({self.col_class_name: str})

        # Read cols as float
        cols = list(data.columns)
        cols.remove(self.col_class_name)
        for col in cols:
            data[col] = data[col].astype(float)

        data[self.col_class_name] = data[self.col_class_name].rank(method='dense', ascending=False).astype(int)

        X = data.drop(labels=self.col_class_name, axis=1).to_numpy()
        y = data[self.col_class_name].astype('int').to_numpy() - 1

        return X, y

    @staticmethod
    def split_train_test(X, y, random_state=42, train_size=0.7):
        """
            Create train and test sets
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=train_size)
        return X_train, X_test, y_train, y_test

    def run(self, store_results=True, plot_graphic=False, debug=False, iteration=0, pruning=True):
        """ Run the tests and store results"""

        print("\n#######################################################################")
        print(f"### Running tests for {self.dataset_name} with {self.classifier.__name__}, "
              f"max_depth {self.max_depth_stop}, min_samples_stop {self.min_samples_stop}")
        if iteration:
            print("Iteration", iteration)

        # Read CSV
        X, y = self.parse_dataset()

        # Populate dataset metrics
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(set(y))

        # Training Models
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, random_state=iteration)

        results = []
        for gini_factor in self.gini_factors:
            for gamma_factor in self.gamma_factors:
                gamma_factor = round(gamma_factor, 2) if gamma_factor else gamma_factor
                initial_time = perf_counter_ns()
                dt = self.classifier(max_depth=self.max_depth_stop, min_samples_stop=self.min_samples_stop)
                score_train, score_test = dt.get_score(X_train, y_train, X_test, y_test, modified_factor=gini_factor,
                                                       gamma_factor=gamma_factor, pruning=pruning)
                unbalanced_splits, max_depth, max_depth_redundant, wad, waes, nodes, features = \
                    dt.get_explainability_metrics()
                final_time = perf_counter_ns()
                execution_time = round((final_time - initial_time) / 1e9, 4)
                if debug:
                    # dt.tree_.debug(
                    #     feature_names=["Attribute {}".format(i) for i in range(len(X_train))],
                    #     class_names=["Class {}".format(i) for i in range(len(y_train))],
                    #     show_details=True
                    # )
                    tree_img_file = self._get_filename(extension="png", gini_factor=gini_factor,
                                                       gamma_factor=gamma_factor, sub_folder="img",
                                                       iteration=iteration)
                    dt.tree_.debug_pydot(tree_img_file)
                print(f"\nTrain/test accuracy for gini factor {gini_factor} and gamma {gamma_factor}: "
                      f"{score_train}, {score_test}")
                print(f"max_depth: {max_depth}, max_depth_redundant: {max_depth_redundant}, wad: {wad}, "
                      f"waes: {waes}, nodes: {nodes}, features: {features}, execution time: {execution_time}")
                print("----------------------------")
                metrics = Metrics(gini_factor=gini_factor, gamma_factor=gamma_factor,
                                  train_accuracy=score_train, test_accuracy=score_test,
                                  max_depth=max_depth, max_depth_redundant=max_depth_redundant,
                                  wad=wad, waes=waes,
                                  unbalanced_splits=unbalanced_splits, nodes=nodes,
                                  features=features, execution_time=execution_time)

                results.append(metrics.get_metrics())

                # Save Tree objects
                if store_results:
                    pickle_name = self._get_filename(extension="pickle", gini_factor=gini_factor,
                                                     gamma_factor=gamma_factor, sub_folder="pickle",
                                                     iteration=iteration)
                    create_dir(pickle_name)
                    with open(pickle_name, 'wb') as handle:
                        pickle.dump(dt.tree_, handle)

        self.results = results
        if plot_graphic:
            self._plot_graphic()
        if store_results:
            self._store_results(iteration=iteration)

        print(f"\n### Ended tests for {self.dataset_name} with with {self.classifier.__name__}, "
              f"max_depth {self.max_depth_stop}, min_samples_stop {self.min_samples_stop}\n")

    @staticmethod
    def load_test_from_pickle(pickle_filepath, csv_file=None, col_class_name=None, cols_to_delete=None,
                              results_folder="results"):
        """
            Load test from pickle file
        """
        final_slash_idx = pickle_filepath.rfind("/")
        pickle_filename = pickle_filepath[final_slash_idx + 1:]
        pickle_info = pickle_filename.replace(".pickle", "").split("_")
        dataset = pickle_info[0]
        algorithm = pickle_info[1]
        depth_index = pickle_info.index("depth")
        max_depth_stop = int(pickle_info[depth_index + 1])
        min_samples_stop_index = int(pickle_info.index("samples"))
        min_samples_stop = pickle_info[min_samples_stop_index + 1]
        gini_factor_index = pickle_info.index("gini-factor")
        gini_factor = float(pickle_info[gini_factor_index + 1])
        iteration_index = pickle_info.index("iteration")
        iteration = float(pickle_info[iteration_index + 1])
        try:
            gamma_factor_index = pickle_info.index("gamma")
            gamma = float(pickle_info[gamma_factor_index + 1])
        except ValueError:
            gamma = None
        # Get Train and test accuracy
        if algorithm == "AlgoWithGini":
            clf = AlgoWithGini
        elif algorithm == "Cart":
            clf = Cart
        else:
            raise ValueError("Classifier not found!")

        with open(pickle_filepath, "rb") as f:
            tree_obj: Node = pickle.load(f)
            clf_obj = clf(max_depth=max_depth_stop, min_samples_stop=min_samples_stop)
            clf_obj.tree_ = tree_obj
            test = Test(classifier=clf, dataset_name=dataset, csv_file=csv_file,
                        max_depth_stop=max_depth_stop, col_class_name=col_class_name, cols_to_delete=cols_to_delete,
                        min_samples_stop=min_samples_stop, gini_factors=[gini_factor], gamma_factors=[gamma],
                        results_folder=results_folder)
            test.clf_obj = clf_obj
            return test, iteration
