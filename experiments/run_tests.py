"""
    Main file to run tests
"""
import csv
import json
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict

from experiments.test import Test, Result, MetricType
from algorithms.cart import CART
from algorithms.ser_dt import SERDT
from experiments.models import Dataset
from utils.file_utils import create_dir

# Folder to save results
RESULTS_FOLDER = "results"


def change_all_jsons(results_folder):
    """
    Change all json inside a folder
    """
    files = [f for f in listdir(results_folder) if isfile(join(results_folder, f))]
    json_files = [f for f in files if 'json' in f]

    for file in json_files:
        with open(results_folder + "/" + file, "r+") as json_file:
            file_data = json.load(json_file)

            # Change json
            # insert code here

            json_file.seek(0)
            json.dump(file_data, json_file, indent=2, sort_keys=True)
            json_file.truncate()
        print("Changed file:", file)


def create_bins(csv_file, bins=10, cols_to_remove=None):
    """
        Discretize dataset
    """
    data = pd.read_csv(csv_file)
    cols = data.columns
    cols = [col for col in cols if col not in cols_to_remove]
    for col in cols:
        possible_values = set(data[col].tolist())
        if len(possible_values) > 10:
            min_value = min(possible_values)
            max_value = max(possible_values)
            limits = np.linspace(min_value, max_value, num=bins + 1)
            conditions = [data[col] <= choice for choice in limits[1:]]
            choices = [(limits[i] + limits[i + 1]) / 2 for i in range(len(limits) - 1)]
            data[col] = np.select(conditions, choices)

    index_str = csv_file.find(".csv")
    outfile = csv_file[:index_str] + "_bins" + ".csv"
    data.to_csv(outfile, encoding='utf-8', index=None)


def save_pruned_trees(pickle_dir, ds_by_name, pruned_dir, pickle_file=None):
    """
        Save pruned trees
    """
    if not pickle_file:
        files = [f for f in listdir(pickle_dir) if isfile(join(pickle_dir, f))]
        pickle_files = sorted([f for f in files if 'pickle' in f])
    else:
        pickle_files = [pickle_file]

    for pickle_filename in pickle_files:
        print(f"### Punning tree {pickle_filename}")
        test = Test.load_test_from_pickle(f"{pickle_dir}/{pickle_filename}", datasets_by_name=ds_by_name)
        clf_obj = test.clf_obj
        test.post_pruning = True
        X, y = test.parse_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = test.split_train_test(X, y, random_state=test.iteration)
        pruned_tree = clf_obj.get_post_pruned_tree(X_val, y_val)
        test.results_folder = pruned_dir
        pruned_filepath = test._get_filename("pickle")
        create_dir(pruned_filepath)
        with open(pruned_filepath, 'wb') as handle:
            pickle.dump(pruned_tree, handle)


def generate_consolidates_csv(csv_file, result_dir, filename=None, load_from="json",
                              ds_by_name: Dict[str, Dataset] = None):
    """
        Generates consolidate csv with all metrics for datasets
    """
    if not filename:
        files = sorted([f for f in listdir(result_dir) if isfile(join(result_dir, f))])
    else:
        files = [filename]
    test_info_header = Result.get_param_list()
    metrics_header = Result.get_metric_list(get_execution_time=False)
    header = test_info_header + metrics_header
    header = [hd.value for hd in header]

    if load_from == "json":
        json_files = [f for f in files if 'json' in f]
        rows = []
        for file in json_files:
            with open(result_dir + "/" + file) as json_file:
                file_data = json.load(json_file)
                row = [file_data[info_header.value] for info_header in test_info_header]
                row += [file_data[metric.value] for metric in metrics_header]
                rows.append(row)
    # Load from pickles
    elif load_from == "pickle":
        print("####################")
        print("Generating consolidated results:")
        pickle_files = [f for f in files if 'pickle' in f]
        pickle_files = sorted(pickle_files)
        rows = []
        for pickle_filename in pickle_files:
            test = Test.load_test_from_pickle(f"{result_dir}/{pickle_filename}", datasets_by_name=ds_by_name)
            clf_obj = test.clf_obj
            tree_obj = clf_obj.tree_
            print("\n#######################################################################")
            print(f"### Running tests for:")
            params = test.get_parameters()
            for param_type, param_value in params.items():
                print(f"{param_type.value}: {param_value}")

            X, y = test.parse_dataset()
            X_train, X_val, X_test, y_train, y_val, y_test = test.split_train_test(X, y, random_state=test.iteration)
            score_train, score_test = clf_obj.get_score_without_fit(X_train, y_train, X_test, y_test)
            num_features = len(tree_obj.feature_index_occurrences)

            explainability_metrics = tree_obj.get_explainability_metrics(num_features)
            metrics = {MetricType.train_accuracy: score_train, MetricType.test_accuracy: score_test}
            metrics = {**metrics, **explainability_metrics}
            result = Result(test.get_parameters(), metrics).get_json(get_execution_time=False)
            row = [value for key, value in result.items()]
            rows.append(row)
            print("\nResults:")
            for metric_type, metric_value in metrics.items():
                print(f"{metric_type.value}: {metric_value}")
            print("----------------------------")

    else:
        raise ValueError("load_from can be json or pickle")

    rows = sorted(rows)
    # Create dir if not exists
    dir_index = csv_file.rfind("/")
    dir_name = csv_file[:dir_index]
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    with open(csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(rows)


def one_hot_encoding(csv_file, categorical_cols=None, cols_to_remove=None):
    """
    For the categorical (and not numerical) columns, apply one hot encoding
    """
    data = pd.read_csv(csv_file)
    if cols_to_remove:
        data = data.drop(columns=cols_to_remove)
    if categorical_cols:
        prefix_names = {col: f"dummy_{col}" for col in categorical_cols}
        data = pd.get_dummies(data, columns=categorical_cols, prefix=prefix_names)

    index_str = csv_file.find(".csv")
    outfile = csv_file[:index_str] + "_formatted.csv"
    data.to_csv(outfile, encoding='utf-8', index=False)


if __name__ == "__main__":
    """
        Main section
    """
    datasets = [
        # name, path, col_class_name, categorical_cols, cols_to_delete
        ['anuran', '../data/anuran/anuran_formatted.csv', 'Family', [], []],

        ['audit risk', '../data/audit_data/audit_risk_formatted.csv', 'Risk', [], []],

        ['avila', '../data/avila/avila_formatted.csv', 'Class', [], []],

        ['banknote', '../data/banknote/data_banknote_authentication.csv', 'class', [], []],

        ['bankruptcy polish', '../data/bankruptcy_polish/3year.csv', 'class', [], []],

        ['cardiotocography', '../data/cardiotocography/CTG_formatted.csv', 'CLASS', [], []],

        ['collins', "../data/collins/collins_formatted.csv", 'Corp.Genre', [], []],

        ['default credit card', "../data/defaults_credit_card/defaults_credit_card_formatted.csv",
         'default payment next month', ['SEX', 'EDUCATION', 'MARRIAGE'], []],

        ['dry bean', "../data/dry_bean/Dry_Bean_Dataset_formatted.csv", 'Class', [], []],

        ['eeg eye state', '../data/eeg_eye_state/eeg_eye_state_formatted.csv', 'eyeDetection', [], []],

        ['htru2', '../data/HTRU2/HTRU_2.csv', 'class', [], []],

        ['iris', '../data/iris/iris.csv', 'class', [], []],

        ['letter recognition', '../data/letter_recognition/letter-recognition_formatted.csv', 'lettr', [], []],

        ['mice', '../data/mice/mice_formatted.csv', 'class', ["Genotype", "Treatment", "Behavior"], []],

        ['obs network', '../data/obs_network/obs_network_dataset_formatted.csv', 'Class',
         ['Node', 'NodeStatus'], []],

        ['occupancy room', '../data/occupancy_room/Occupancy_Estimation_formatted.csv', 'Room_Occupancy_Count',
         [], []],

        # Should also include 'TrafficType' for categorial col, but has many values
        ['online shoppers intention', '../data/online_shoppers_intention/online_shoppers_intention_formatted.csv',
         'Revenue', ['Month', 'OperatingSystems', 'Browser', 'Region', 'Weekend'], []],

        ['pen digits', "../data/pen_digits/pendigits_formatted.csv", 'digit', [], []],

        ['poker hand', "../data/poker_hand/poker_hand.csv", 'class', [], []],

        ['sensorless', "../data/sensorless/sensorless_drive_diagnosis.csv", 'class', [], []]

    ]

    dataset_objs = [Dataset(*dataset) for dataset in datasets]

    # Add discrete versions to dataset list:
    # new_datasets = datasets.copy()
    # for ds in datasets:
    #     # create_bins(ds[1], cols_to_remove=ds[-1] + [ds[-2]])
    #     extension_index = ds[1].find(".csv")
    #     bins_filename = ds[1][:extension_index] + "_bins.csv"
    #     new_ds = ds.copy()
    #     new_ds[1] = bins_filename
    #     new_ds[2] = True
    #     new_datasets.append(new_ds)

    # Apply one hot encoding to datasets:
    # for ds in datasets:
    #     one_hot_encoding(ds[1], ds[4], ds[5])

    depths = [6]
    min_samples_list = [0]
    min_samples_frac_list = [None]
    iterations = range(2, 11)

    # Run tests
    for it in iterations:
        for ds in dataset_objs:
            for depth in depths:
                for min_samples_stop in min_samples_list:
                    for min_samples_frac in min_samples_frac_list:
                        test1 = Test(classifier=CART, dataset=ds, max_depth_stop=depth,
                                     min_samples_stop=min_samples_stop,
                                     min_samples_frac=min_samples_frac, gini_factor=1, gamma_factor=None,
                                     results_folder="results/cart", iteration=it, post_pruning=False)
                        test2 = Test(classifier=SERDT, dataset=ds, max_depth_stop=depth,
                                     min_samples_stop=min_samples_stop,
                                     min_samples_frac=min_samples_frac, gini_factor=0.97, gamma_factor=0.5,
                                     results_folder="results/ser_dt", iteration=it, post_pruning=False)
                        # test1.run()
                        # test2.run()
    datasets_info_dict = {ds.name: ds for ds in dataset_objs}

    # Generate consolidated results
    # generate_consolidates_csv("results/consolidated/cart_experiments.csv", "results/cart/json",
    #                           load_from="json")
    # generate_consolidates_csv("results/consolidated/ser_dt_experiments.csv", "results/ser_dt/json",
    #                           load_from="json")

    # Pickle
    generate_consolidates_csv("results/consolidated/cart_experiments_post_pruning.csv",
                              "results/cart/pickle_post_pruned",
                              load_from="pickle", ds_by_name=datasets_info_dict)
    generate_consolidates_csv("results/consolidated/ser_dt_experiments_post_pruning.csv",
                              "results/ser_dt/pickle_post_pruned",
                              load_from="pickle", ds_by_name=datasets_info_dict)

    # Save pickles
    # save_pruned_trees("results/ser_dt/pickle", datasets_info_dict, "results/ser_dt/pickle_post_pruned")
    # save_pruned_trees("results/cart/pickle", datasets_info_dict, "results/cart/pickle_post_pruned")
