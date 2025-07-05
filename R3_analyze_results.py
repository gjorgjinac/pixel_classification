import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Read results from all folds from results/project_split_all_binarized_{binarize_labels}/results_fold_{fold}.csv
if __name__ == "__main__":
    binarize_labels=True
    results=pd.concat([pd.read_csv(f'results/project_split_all_binarized_{binarize_labels}/results_fold_{fold}.csv',index_col=0) for fold in range(0,10)])
    print(results)
    metrics=['Accuracy','Precision','Recall','F1 Score']
    results_melted=results.melt(['Model','Fold'],metrics)
    print(results_melted)
    plt.figure(figsize=(10, 6))
    sns.boxplot(results_melted,y='value',x='variable',hue='Model')
    plt.tight_layout()
    plt.savefig(f'figures/results_binarize_{binarize_labels}.png')
    plt.show()
