"""
    Auxiliary function to plot graphics
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join, isfile
import pandas as pd
import seaborn as sns
from scipy.stats import t, sem

from experiments.test import Test


def plot_factor_graphics(csf_file, output_file, col, y_label):
    """
        Plot graphic for datasets with many factors
    """
    data = pd.read_csv(csf_file)
    data_algo = data[data['algorithm'] == 'AlgoWithGini']
    values_by_factor = defaultdict(list)
    for index, row in data_algo.iterrows():
        factor = row["gini_factor"]
        accuracy = row[col]
        values_by_factor[factor].append(accuracy)

    x = []
    average = []
    err = []
    for factor, value in values_by_factor.items():
        x.append(factor)
        value_to_add = sum(value) / len(value)
        if col == 'test_accuracy':
            value_to_add *= 100
        average.append(value_to_add)
        error_interval = t.interval(0.95, len(value) - 1, loc=np.mean(value), scale=sem(value))
        err.append((error_interval[1] - error_interval[0]) / 2)

    f = plt.figure(figsize=(12, 4), dpi=300)
    plt.rcParams.update({
        'text.usetex': True,
        # 'font.family': 'monospace',
        'font.size': 20,
        # 'font.monospace': ['Computer Modern Typewriter']})
    })

    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    plt.plot(x, average, color="blue")
    # ax.yaxis.set_label_position("right")
    # ax.set(ylabel=None)
    # plt.errorbar(x, average, yerr=err, color="blue")
    plt.xlabel("\\texttt{FactorExpl}")
    plt.ylabel(y_label, labelpad=-25)
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    if col == 'test_accuracy':
        plt.yticks(np.arange(62, 86, 2))
    else:
        plt.yticks(np.arange(1, 4, 0.5))

    plt.margins(x=0.05, y=0.05)
    plt.savefig(output_file, bbox_inches='tight')


def plot_trees(results_dir, pickle_filename=None, pickle_dir=None, post_pruned=False):
    """
        Plot trees in folder specified
    """
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
        if post_pruned:
            sub_folder = "img_post_pruned"
            test.post_pruning = True
        else:
            test.post_pruning = False
            sub_folder = "img"

        tree_img_file = test._get_filename(extension="png", gini_factor=test.gini_factors[0],
                                           gamma_factor=test.gamma_factors[0], sub_folder=sub_folder)
        test.clf_obj.tree_.debug_pydot(tree_img_file)
        print("---------")


def plot_boxplot(csf_file, output_file, column, column_label, use_tex=False):
    """
        Plot boxplot graphics
    """
    data = pd.read_csv(csf_file)
    values_by_dataset = defaultdict(list)

    if use_tex:
        plt.rcParams.update({
            'text.usetex': True,
            'font.size': 16,
        })

    for index, row in data.iterrows():
        dataset = row['dataset']
        accuracy = row[column]
        values_by_dataset[dataset].append(accuracy)

    # Accuracy
    if column == 'test_accuracy':
        datasets_1 = [key for key, values in values_by_dataset.items() if 0.92 < sum(values) / len(values)]
        datasets_2 = [key for key, values in values_by_dataset.items() if 0.8 < sum(values) / len(values) <= 0.92]
        datasets_3 = [key for key, values in values_by_dataset.items() if 0.5 < sum(values) / len(values) <= 0.8]
        datasets_4 = [key for key, values in values_by_dataset.items() if sum(values) / len(values) <= 0.5]
        datasets = [datasets_1, datasets_2, datasets_3, datasets_4]

    elif column == 'waes':
        datasets_1 = [key for key, values in values_by_dataset.items() if 4 < sum(values) / len(values)]
        datasets_2 = [key for key, values in values_by_dataset.items() if 3.5 < sum(values) / len(values) <= 4]
        datasets_3 = [key for key, values in values_by_dataset.items() if 2 < sum(values) / len(values) <= 3.5]
        datasets_4 = [key for key, values in values_by_dataset.items() if sum(values) / len(values) <= 2]
        datasets = [datasets_1, datasets_2, datasets_3, datasets_4]
    else:
        raise NotImplementedError()

    for count, dataset_list in enumerate(datasets):
        df = data[data['dataset'].isin(dataset_list)]
        plt.figure(figsize=(12, 8), dpi=300)
        ax = sns.boxplot(x="dataset", y=column, hue="algorithm", data=df)
        ax.legend(title="Algorithms")
        ax.legend_.texts[0].set_text("\\texttt{SER-DT}")
        ax.legend_.texts[1].set_text("\\texttt{CART}")
        plt.xticks(rotation='50', ha='right', fontsize=16)
        # plt.xlabel("Dataset", fontsize=16)
        ax.set(xlabel=None)
        plt.ylabel(column_label, fontsize=16)
        output_file = output_file.replace(".jpg", "")
        plt.savefig(f"{output_file}_{count}.jpg", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/accuracy_factors.jpg",
    #                      "test_accuracy", "Test Accuracy")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/waes_factors.jpg",
    #                      "waes", "expl\\textsubscript{avg}")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/wad_factors.jpg",
    #                      "wad", "WAD")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/nodes_factors.jpg",
    #                      "nodes", "Nodes")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/features_factors.jpg",
    #                      "features", "Features")

    # plot_boxplot("results/consolidated/experiments.csv", "results/consolidated/boxplot_accuracy.jpg", 'test_accuracy',
    #              "Test Accuracy", use_tex=True)
    # plot_boxplot("results/consolidated/experiments.csv", "results/consolidated/boxplot_waes.jpg", 'waes',
    #              "\\texttt{expl\\textsubscript{avg}}",
    #              use_tex=True)

    filename = "obs network_AlgoWithGini_depth_6_samples_0_gini-factor_0.94_gamma_0.5_iteration_3.pickle"
    plot_trees(
        pickle_filename=f"results/algo_gini/pickle_post_pruned/{filename}",
        results_dir="results/algo_gini", post_pruned=True)
    plot_trees(
        pickle_filename=f"results/algo_gini/pickle/{filename}",
        results_dir="results/algo_gini", post_pruned=False)
    # plot_trees(
    #     pickle_filename="results/algo_gini/pickle/banknote_AlgoWithGini_depth_6_samples_0_gini-factor_1_gamma_0.5_iteration_1.pickle",
    #     results_dir="results/algo_gini", pruned=True)
