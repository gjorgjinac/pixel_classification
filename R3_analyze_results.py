import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from globals import models
from utils.utils_model_training import generate_confusion_matrix_figure


def generate_metrics_boxplot(binarize_labels):
    '''
    Generates a boxplot of the results from all folds for the specified binarization of labels.
    :param binarize_labels: whether to plot results for the binary or multiclass classification
    '''
    all_results = []
    for resample in ['none', 'over', 'under']:
        results = pd.concat(
            [pd.read_csv(
                f'results/project_split_all_binarized_{binarize_labels}_do_resampling_{resample}/results_fold_{fold}.csv',
                index_col=0) for \
             fold in range(0, 10) if os.path.isfile(
                f'results/project_split_all_binarized_{binarize_labels}_do_resampling_{resample}/results_fold_{fold}.csv')])
        all_results += [results.assign(resample=resample)]
    all_results = pd.concat(all_results)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    results_melted = all_results.melt(['Model', 'Fold', 'resample'], metrics)
    print(results_melted.groupby(['Model', 'resample', 'variable']).mean())
    print(results_melted.groupby(['Model', 'resample', 'variable']).median())
    plt.figure(figsize=(10, 6))
    sns.catplot(results_melted, y='value', x='variable', hue='resample', col='Model', kind='box', col_wrap=3, height=3)
    plt.tight_layout()
    plt.savefig(f'figures/results_binarize_{binarize_labels}_resampling.png')
    plt.show()


if __name__ == "__main__":
    generate_metrics_boxplot(True)
    generate_metrics_boxplot(False)




