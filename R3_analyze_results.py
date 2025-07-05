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
    results = pd.concat(
        [pd.read_csv(f'results/project_split_all_binarized_{binarize_labels}/results_fold_{fold}.csv', index_col=0) for
         fold in range(0, 10)])
    print(results)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    results_melted = results.melt(['Model', 'Fold'], metrics)
    print(results_melted)
    plt.figure(figsize=(10, 6))
    sns.boxplot(results_melted, y='value', x='variable', hue='Model')
    plt.tight_layout()
    plt.savefig(f'figures/results_binarize_{binarize_labels}.png')
    plt.show()


# Read results from all folds from results/project_split_all_binarized_{binarize_labels}/results_fold_{fold}.csv
if __name__ == "__main__":
    binarize_labels = False
    # generate_metrics_boxplot(binarize_labels)
    for model_name in models.keys():
        predictions = pd.concat(
            [pd.read_csv(
                f'results/project_split_all_binarized_{binarize_labels}/{model_name}_fold_{fold}_test_and_predictions.csv',
                index_col=0) for fold in range(0, 10)])
        confusion_matrix=generate_confusion_matrix_figure(predictions['target'], predictions['predictions'], f'figures/final_confusion_matrix_binarized_{binarize_labels}_{model_name}.png')


