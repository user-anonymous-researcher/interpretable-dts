"""
    Main file to run tests
"""
import copy
import csv
import json
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
import pickle

from experiments.test import MetricType, Test, ResultJson, Metrics
from algorithms.cart import Cart
from algorithms.algo_with_gini import AlgoWithGini
from algorithms.algo_info_gain import AlgoInfoGain
from algorithms.c45 import C45
from algorithms.ec2 import EC2
from tree.tree import Node
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


def save_pruned_trees(pickle_dir, ds_by_name, pruned_dir, pickle_file=None, pruning=True, post_pruning=False):
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
        test = Test.load_test_from_pickle(f"{pickle_dir}/{pickle_filename}")
        test.csv_file = ds_by_name[test.dataset_name]['csv']
        test.col_class_name = ds_by_name[test.dataset_name]['col_class']
        clf_obj = test.clf_obj
        tree_obj: Node = clf_obj.tree_
        pruned_tree = copy.deepcopy(tree_obj)
        if pruning:
            pruned_tree = tree_obj.get_pruned_tree()
        if post_pruning:
            X, y = test.parse_dataset()
            X_train, X_val, X_test, y_train, y_val, y_test = test.split_train_test(X, y, random_state=test.iteration)
            pruned_tree = clf_obj.get_post_pruned_tree(X_val, y_val)
        pruned_filepath = f"{pruned_dir}/{pickle_filename}"
        create_dir(pruned_filepath)
        with open(pruned_filepath, 'wb') as handle:
            pickle.dump(pruned_tree, handle)


def generate_consolidates_csv(csv_file, result_dir, filename=None, load_from="json", ds_by_name: dict = None,
                              pruning=True, post_pruning=False):
    """
        Generates consolidate csv with all metrics for datasets
    """
    if not filename:
        files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
    else:
        files = [filename]
    test_info_header = ResultJson.get_cols()
    metrics_header = Metrics.get_cols()
    header = test_info_header + metrics_header

    if load_from == "json":
        json_files = [f for f in files if 'json' in f]
        rows = []
        for file in json_files:
            with open(result_dir + "/" + file) as json_file:
                file_data = json.load(json_file)
                for result in file_data['results']:
                    if MetricType.execution_time.value not in result:
                        break
                    row = [file_data[info_header] for info_header in test_info_header]
                    row += [result[metric] for metric in metrics_header]
                    rows.append(row)
    # Load from pickles
    elif load_from == "pickle":
        print("####################")
        print("Generating consolidated results:")
        pickle_files = [f for f in files if 'pickle' in f]
        pickle_files = sorted(pickle_files)
        rows = []
        for pickle_filename in pickle_files:
            test = Test.load_test_from_pickle(f"{result_dir}/{pickle_filename}", pruning=pruning,
                                              post_pruning=post_pruning)
            test.csv_file = ds_by_name[test.dataset_name]['csv']
            test.col_class_name = ds_by_name[test.dataset_name]['col_class']
            clf_obj = test.clf_obj
            tree_obj = clf_obj.tree_
            num_features = len(tree_obj.feature_index_occurrences)
            clf_name = clf_obj.__class__.__name__
            print(f"### Generating results for {test.dataset_name} with {clf_name}, "
                  f"max_depth {test.max_depth_stop}, min_samples_stop {test.min_samples_stop}, "
                  f"min_samples_frac {test.min_samples_frac}, "
                  f"gini factor {test.gini_factors[0]}, gamma {test.gamma_factors[0]}, "
                  f"post pruning {test.post_pruning}  and iteration {test.iteration}")

            X, y = test.parse_dataset()
            X_train, X_val, X_test, y_train, y_val, y_test = test.split_train_test(X, y, random_state=test.iteration)
            score_train, score_test = clf_obj.get_score_without_fit(X_train, y_train, X_test, y_test)
            metrics = tree_obj.get_explainability_metrics(num_features)
            row = [test.dataset_name, clf_name, test.max_depth_stop, test.min_samples_stop, test.gini_factors[0],
                   test.gamma_factors[0], score_train, score_test, *metrics]
            rows.append(row)

    else:
        raise ValueError("load_from can be json or pickle")

    rows = sorted(rows, key=lambda x: (x[0], x[4], x[5], x[6]))
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
        ['anuran', '../data/anuran/anuran_formatted.csv', 'Family', [], ['Genus', 'Species', 'RecordID']],

        ['audit risk', '../data/audit_data/audit_risk_formatted.csv', 'Risk', [], []],

        ['avila', '../data/avila/avila_formatted.csv', 'Class', [], []],

        ['banknote', '../data/banknote/data_banknote_authentication.csv', 'class', [], []],

        ['bankruptcy polish', '../data/bankruptcy_polish/3year.csv', 'class', [], []],

        ['cardiotocography', '../data/cardiotocography/CTG_formatted.csv', 'CLASS', [],
         ['b', 'e', 'LBE', 'DR', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']],

        ['collins', "../data/collins/collins_formatted.csv", 'Corp.Genre', [],
         ["Text", "Genre", "Counter", "Corpus"]],

        ['default credit card', "../data/defaults_credit_card/defaults_credit_card_formatted.csv",
         'default payment next month', ['SEX', 'EDUCATION', 'MARRIAGE'], ['ID']],

        ['dry bean', "../data/dry_bean/Dry_Bean_Dataset_formatted.csv", 'Class', [], []],

        ['eeg eye state', '../data/eeg_eye_state/eeg_eye_state_formatted.csv', 'eyeDetection', [], []],

        ['htru2', '../data/HTRU2/HTRU_2.csv', 'class', [], []],

        ['iris', '../data/iris/iris.csv', 'class', [], []],

        ['letter recognition', '../data/letter_recognition/letter-recognition_formatted.csv', 'lettr', [], []],

        ['mice', '../data/mice/mice_formatted.csv', 'class', ["Genotype", "Treatment", "Behavior"], ["MouseID"]],

        ['obs network', '../data/obs_network/obs_network_dataset_formatted.csv', 'Class',
         ['Node', 'NodeStatus'], ['id']],

        ['occupancy room', '../data/occupancy_room/Occupancy_Estimation_formatted.csv', 'Room_Occupancy_Count',
         [], ['Date', 'Time']],

        # Should also include 'TrafficType' for categorial col, but has many values
        ['online shoppers intention', '../data/online_shoppers_intention/online_shoppers_intention_formatted.csv',
         'Revenue', ['Month', 'OperatingSystems', 'Browser', 'Region', 'Weekend'], []],

        ['pen digits', "../data/pen_digits/pendigits_formatted.csv", 'digit', [], []],

        ['poker hand', "../data/poker_hand/poker_hand.csv", 'class', [], []],

        ['sensorless', "../data/sensorless/sensorless_drive_diagnosis.csv", 'class', [], []]

    ]

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

    all_datasets = sorted(datasets, key=lambda x: (x[0], x[2]))
    depths = [6]
    min_samples_list = [0]
    min_samples_frac_list = [None]
    iterations = range(1, 11)

    # Run tests
    for it in iterations:
        for ds in all_datasets:
            for depth in depths:
                for min_samples_stop in min_samples_list:
                    for min_samples_frac in min_samples_frac_list:
                        # algo_name, path, col_class_name, categorical_cols, cols_to_delete
                        # Cols to deleted was already deleted
                        test1 = Test(classifier=Cart, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                                     col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                                     min_samples_frac=min_samples_frac, iteration=it,
                                     results_folder="results/cart", gamma_factors=[None], gini_factors=[1])
                        test2 = Test(classifier=AlgoWithGini, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                                     col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                                     min_samples_frac=min_samples_frac, iteration=it,
                                     results_folder="results/algo_gini", gamma_factors=[0.5], gini_factors=[0.97])
                        # test3 = Test(classifier=AlgoInfoGain, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                        #              col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                        #              min_samples_frac=min_samples_frac, iteration=it,
                        #              results_folder="results/algo_info_gain", gamma_factors=[0.5], gini_factors=[0.97])
                        # test4 = Test(classifier=C45, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                        #              col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                        #              min_samples_frac=min_samples_frac, iteration=it,
                        #              results_folder="results/c45", gamma_factors=[None], gini_factors=[1])
                        # test5 = Test(classifier=EC2, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                        #              col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                        #              min_samples_frac=min_samples_frac, iteration=it,
                        #              results_folder="results/ec2", gamma_factors=[None], gini_factors=[1])

                        # test1.run(debug=True)
                        # test2.run(debug=False)
                        # test3.run(debug=False)
                        # test4.run(debug=False)
                        # test5.run(debug=True)
                        #
    datasets_by_name = {
        ds[0]: {
            "csv": ds[1],
            "col_class": ds[2],
        }
        for ds in datasets}

    # generate_consolidates_csv("results/consolidated/cart_experiments.csv", "results/cart/json",
    #                           load_from="json")
    # generate_consolidates_csv("results/consolidated/algo_gini_experiments.csv", "results/algo_gini/json",
    #                           load_from="json")
    # generate_consolidates_csv("results/consolidated/algo_info_gain_experiments.csv", "results/algo_info_gain/json",
    #                           load_from="json")
    # generate_consolidates_csv("results/consolidated/c45_experiments.csv", "results/c45/json",
    #                           load_from="json")
    # generate_consolidates_csv("results/consolidated/ec2_experiments.csv", "results/ec2/json",
    #                           load_from="json")

    # Pickle
    # generate_consolidates_csv("results/consolidated/cart_experiments.csv", "results/cart/pickle",
    #                           load_from="pickle", ds_by_name=datasets_by_name)
    # filename = "obs network_AlgoWithGini_depth_6_samples_0_gini-factor_0.94_gamma_0.5_iteration_3.pickle"
    # generate_consolidates_csv("results/consolidated/algo_gini_experiments.csv", "results/algo_gini/pickle",
    #                           filename=filename, load_from="pickle", ds_by_name=datasets_by_name, post_pruning=False)
    # generate_consolidates_csv("results/consolidated/algo_gini_experiments_post_pruned.csv", "results/algo_gini/pickle_post_pruned",
    #                           filename=filename, load_from="pickle", ds_by_name=datasets_by_name, post_pruning=True)
    #
    # Save Pickes
    save_pruned_trees("results/algo_gini/pickle", datasets_by_name, "results/algo_gini/pickle_post_pruned",
                      post_pruning=True)
    # save_pruned_trees("results/cart/pickle", datasets_by_name, "results/cart/pickle_post_pruned",
    #                   post_pruning=True)
    # save_pruned_trees("results/algo_gini/pickle", datasets_by_name, "results/algo_gini/pickle_post_pruned",
    #                   pickle_file="obs network_AlgoWithGini_depth_6_samples_0_gini-factor_0.94_gamma_0.5_iteration_3.pickle",
    #                   pruning=False,
    #                   post_pruning=True)
