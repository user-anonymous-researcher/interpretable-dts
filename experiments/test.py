"""
    Base Test Class with utilities
"""

import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from time import perf_counter_ns
from typing import Dict, Any

from algorithms.ser_dt import SERDT
from algorithms.cart import CART
from experiments.models import MetricType, ParameterType, Result, Dataset
from tree.tree import Node
from utils.convert import convert_to_str
from utils.file_utils import create_dir


class Test:
    """
        Base class to run and save tests
    """

    def __init__(self, classifier, dataset: Dataset, max_depth_stop, min_samples_stop: int = 0,
                 min_samples_frac: float = None, gini_factor: float = 1, gamma_factor: float = None,
                 results_folder="results",
                 iteration=0, pruning=True, post_pruning=False):
        self.classifier = classifier
        self.clf_obj = None  # in case of pickle
        self.dataset = dataset
        self.max_depth_stop = max_depth_stop
        self.min_samples_stop = min_samples_stop
        self.min_samples_frac = min_samples_frac
        self.results_folder = results_folder
        self.gini_factor = gini_factor
        self.gamma_factor = gamma_factor
        self.iteration = iteration
        self.pruning = pruning  # Always pruning
        self.post_pruning = post_pruning
        self.n_classes = None  # Need to update via run()
        self.n_samples = None  # Need to update via run()
        self.n_features = None  # Need to update via run()

    def get_parameters(self) -> Dict[ParameterType, Any]:
        return {ParameterType.dataset: self.dataset.name,
                ParameterType.algorithm: self.classifier.__name__,
                ParameterType.pruning: self.pruning,
                ParameterType.post_pruning: self.post_pruning,
                ParameterType.max_depth: self.max_depth_stop,
                ParameterType.min_samples: self.min_samples_stop,
                ParameterType.min_samples_frac: self.min_samples_frac,
                ParameterType.gini_factor: self.gini_factor,
                ParameterType.gamma_factor: self.gamma_factor,
                ParameterType.iteration: self.iteration,
                }

    def _get_filename(self, extension: str, sub_folder=None) -> str:
        """
            Get filename to save result
        """
        folder = f"{self.results_folder}/{sub_folder}" if sub_folder else f"{self.results_folder}"
        filename = f"{folder}/"
        for param, value in self.get_parameters().items():
            filename += f"{param.value.replace('_', '-')}_{convert_to_str(value)}_"
        filename = filename[:-1]
        filename += f".{extension}"
        return filename

    def _store_results(self, result: Result):
        """
            Store results in json for iteration
        """
        sub_folder = "json" if not self.post_pruning else "json_post_pruning"
        filename: str = self._get_filename("json", sub_folder=sub_folder)
        result_json = result.get_json()
        result_formatted = {key.value: value for key, value in result_json.items()}
        # Create dir if not exists
        create_dir(filename)
        with open(filename, 'w') as f:
            json.dump(result_formatted, f, indent=2)

    def parse_dataset(self):
        """
            Read CSV and clean dataset.
            Set min_samples_stop if min_samples_frac is set
        """
        # Read CSV
        np.set_printoptions(suppress=True)
        data = pd.read_csv(self.dataset.csv_file)
        if self.dataset.cols_to_delete:
            data = data.drop(labels=self.dataset.cols_to_delete, axis=1)
        data = data.astype({self.dataset.col_class_name: str})

        # Read cols as float
        cols = list(data.columns)
        cols.remove(self.dataset.col_class_name)
        for col in cols:
            data[col] = data[col].astype(float)

        data[self.dataset.col_class_name] = data[self.dataset.col_class_name].\
            rank(method='dense', ascending=False).astype(int)

        X = data.drop(labels=self.dataset.col_class_name, axis=1).to_numpy()
        y = data[self.dataset.col_class_name].astype('int64').to_numpy() - 1

        return X, y

    @staticmethod
    def split_train_test(X, y, random_state=42, train_size=0.7):
        """
            Create train and test sets
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=train_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=random_state, train_size=0.5)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run(self, store_results=True, debug=False):
        """ Run the tests and store results"""

        params = self.get_parameters()
        print("\n#######################################################################")
        print(f"### Running tests for:")
        for param_type, param_value in params.items():
            print(f"{param_type.value}: {param_value}")

        # Read CSV
        X, y = self.parse_dataset()

        # Populate dataset metrics
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(set(y))

        # Training Models
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_test(X, y, random_state=self.iteration)

        gamma_factor = round(self.gamma_factor, 2) if self.gamma_factor else self.gamma_factor
        min_samples_stop = self.min_samples_stop if not self.min_samples_frac else self.min_samples_frac * len(y)
        initial_time = perf_counter_ns()
        dt = self.classifier(max_depth=self.max_depth_stop, min_samples_stop=min_samples_stop)
        score_train, score_test = dt.get_score(X_train, y_train, X_test, y_test, modified_factor=self.gini_factor,
                                               gamma_factor=gamma_factor, pruning=self.pruning,
                                               post_pruning=self.post_pruning,
                                               X_val=X_val, y_val=y_val)
        explainability_metrics = dt.get_explainability_metrics()
        metrics = {MetricType.train_accuracy: score_train,
                   MetricType.test_accuracy: score_test}
        final_time = perf_counter_ns()
        execution_time = round((final_time - initial_time) / 1e9, 4)
        metrics = {**metrics, **explainability_metrics, MetricType.execution_time: execution_time}

        result = Result(self.get_parameters(), metrics)
        if debug:
            tree_img_file = self._get_filename(extension="png", sub_folder="img")
            dt.tree_.debug_pydot(tree_img_file)
        print("\nResults:")
        for metric_type, metric_value in metrics.items():
            print(f"{metric_type.value}: {metric_value}")
        print("----------------------------")

        # Saving Results in Json
        if store_results:
            self._store_results(result)

        # Save Tree objects
        if store_results:
            sub_folder = "pickle" if not self.post_pruning else "pickle_post_pruning"
            pickle_name = self._get_filename(extension="pickle", sub_folder=sub_folder)
            create_dir(pickle_name)
            with open(pickle_name, 'wb') as handle:
                pickle.dump(dt.tree_, handle)

    @staticmethod
    def load_test_from_pickle(pickle_filepath, datasets_by_name: Dict[str, Dataset]=None, results_folder="results"):
        """
            Load test from pickle file
        """
        # TODO: Parametrize this: create classes for parameters that specifies type (str, int, etc.).
        final_slash_idx = pickle_filepath.rfind("/")
        pickle_filename = pickle_filepath[final_slash_idx + 1:]
        pickle_info = pickle_filename.replace(".pickle", "").split("_")
        dataset_index = pickle_info.index(ParameterType.dataset.value)
        dataset = pickle_info[dataset_index + 1]
        algorithm_index = pickle_info.index(ParameterType.algorithm.value)
        algorithm = pickle_info[algorithm_index + 1]
        depth_index = pickle_info.index("max-depth")
        try:
            max_depth_stop = int(pickle_info[depth_index + 1])
        except ValueError:
            # In case of math.inf
            max_depth_stop = float(pickle_info[depth_index + 1])

        min_samples_stop_index = int(pickle_info.index("min-samples"))
        min_samples_stop = int(pickle_info[min_samples_stop_index + 1])
        try:
            min_samples_frac_index = int(pickle_info.index("min-samples-frac"))
            min_samples_frac = float(pickle_info[min_samples_frac_index + 1])
        except ValueError:
            min_samples_frac = None
        gini_factor_index = pickle_info.index("gini-factor")
        gini_factor = float(pickle_info[gini_factor_index + 1])
        iteration_index = pickle_info.index("iteration")
        iteration = int(pickle_info[iteration_index + 1])
        pruning_index = pickle_info.index("pruning")
        pruning = pickle_info[pruning_index + 1] == "True"
        post_pruning_index = pickle_info.index("post-pruning")
        post_pruning = pickle_info[post_pruning_index + 1] == "True"
        try:
            gamma_factor_index = pickle_info.index("gamma")
            gamma = float(pickle_info[gamma_factor_index + 1])
        except ValueError:
            gamma = None
        # Get Train and test accuracy
        if algorithm == "SERDT":
            clf = SERDT
        elif algorithm == "CART":
            clf = CART
        else:
            raise ValueError("Classifier not found!")

        with open(pickle_filepath, "rb") as f:
            tree_obj: Node = pickle.load(f)
            clf_obj = clf(max_depth=max_depth_stop, min_samples_stop=min_samples_stop)
            clf_obj.tree_ = tree_obj
            test = Test(classifier=clf, dataset=datasets_by_name[dataset],
                        max_depth_stop=max_depth_stop,
                        min_samples_stop=min_samples_stop, min_samples_frac=min_samples_frac,
                        gini_factor=gini_factor, gamma_factor=gamma,
                        results_folder=results_folder, iteration=iteration, pruning=pruning,
                        post_pruning=post_pruning)
            test.clf_obj = clf_obj
            return test
