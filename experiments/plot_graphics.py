from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join, isfile
import pandas as pd
from scipy.stats import t, sem

from experiments.test import Test


def plot_factor_graphics(csf_file, output_file, col, y_label):
    data = pd.read_csv(csf_file)
    data_algo = data[data['algorithm'] == 'AlgoWithGini']
    accuracy_trains_by_factor = defaultdict(list)
    for index, row in data_algo.iterrows():
        factor = row["gini_factor"]
        accuracy = row[col]
        accuracy_trains_by_factor[factor].append(accuracy)

    x = []
    average = []
    err = []
    for factor, value in accuracy_trains_by_factor.items():
        x.append(factor)
        average.append(sum(value) / len(value))
        error_interval = t.interval(0.95, len(value) - 1, loc=np.mean(value), scale=sem(value))
        err.append((error_interval[1] - error_interval[0]) / 2)

    f = plt.figure(figsize=(12, 4), dpi=300)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.plot(x, average, color="blue")
    # plt.errorbar(x, average, yerr=err, color="blue")
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    plt.xlabel("Gini Factor")
    plt.ylabel(y_label)
    plt.margins(x=0.05, y=0.05)
    plt.savefig(output_file, bbox_inches='tight')


def plot_trees(results_dir, pickle_filename=None, pickle_dir=None, pruned=False):
    if pickle_filename:
        pickle_files = [pickle_filename]
    else:
        files = [f for f in listdir(pickle_dir) if isfile(join(pickle_dir, f))]
        pickle_files = [f for f in files if 'pickle' in f]
        pickle_files = [f"{pickle_dir}/{pickle_file}" for pickle_file in pickle_files]
        pickle_files = sorted(pickle_files)
    for pickle_file in pickle_files:
        print(f"### Generate image for tree {pickle_file}")
        test = Test.load_test_from_pickle(pickle_file, results_folder=results_dir)
        sub_folder = "img" if not pruned else "img_pruned"
        tree_img_file = test._get_filename(extension="png", gini_factor=test.gini_factors[0],
                                           gamma_factor=test.gamma_factors[0], sub_folder=sub_folder)
        test.clf_obj.tree_.debug_pydot(tree_img_file)
        print("---------")


if __name__ == "__main__":
    plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/accuracy_factors.jpg",
                         "test_accuracy", "Test Accuracy")
    plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/wad_factors.jpg",
                         "wad", "WAD")
    plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/waes_factors.jpg",
                         "waes", "WAES")
    plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/nodes_factors.jpg",
                         "nodes", "Nodes")
    plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/features_factors.jpg",
                         "features", "Features")

    # plot_trees(
    #     pickle_filename="results/algo_gini/pickle_pruned/sensorless_AlgoWithGini_depth_4_samples_0_gini-factor_0.97_gamma_0.9.pickle",
    #     results_dir="results/algo_gini", pruned=True)
    # plot_trees(
    #     pickle_filename="results/cart/pickle_pruned/sensorless_Cart_depth_4_samples_0_gini-factor_1.pickle",
    #     results_dir="results/cart", pruned=True)
