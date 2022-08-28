from enum import Enum
from typing import Any, Dict, List


class MetricType(Enum):
    """
        Possible metrics for a test
    """
    train_accuracy = "train_accuracy"
    test_accuracy = "test_accuracy"
    expl_avg = "expl_avg"
    expl_wc = "expl_wc"
    depth_avg = "depth_avg"
    depth_wc = "depth_wc"
    nodes = "nodes"
    features = "features"
    unbalanced_splits = "unbalanced_splits"
    execution_time = "execution_time"


class ParameterType(Enum):
    """
        Possible parameters for running a test
    """
    dataset = "dataset"
    algorithm = "algorithm"
    iteration = "iteration"
    max_depth = "max_depth"
    min_samples = "min_samples"
    min_samples_frac = "min_samples_frac"
    pruning = "pruning"
    post_pruning = "post_pruning"
    gini_factor = "gini_factor"
    gamma_factor = "gamma"


class Result:
    """Class to manage result"""

    def __init__(self, parameters: Dict[ParameterType, Any], metrics: Dict[MetricType, Any]):
        self.parameters = parameters
        self.metrics = metrics

    @staticmethod
    def get_param_list():
        return [param for param in ParameterType]

    @staticmethod
    def get_metric_list(get_execution_time=True):
        metrics = [metric for metric in MetricType]
        if not get_execution_time:
            metrics.remove(MetricType.execution_time)
        return metrics

    def get_json(self, get_execution_time=True):
        metrics = self.get_metric_list(get_execution_time=get_execution_time).copy()

        # Put params and metrics in order
        param_dict = {param: self.parameters[param] for param in self.get_param_list()}
        metric_dict = {metric: self.metrics[metric] for metric in metrics}
        return {**param_dict, **metric_dict}


class Dataset:
    def __init__(self, name: str, csv_file: str, col_class_name: str, categorical_cols: List[str],
                 cols_to_delete: List[str] = None):
        self.name = name
        self.csv_file = csv_file
        self.col_class_name = col_class_name
        self.categorial_cols = categorical_cols
        self.cols_to_delete = cols_to_delete
