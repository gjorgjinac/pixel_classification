import os
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_feature_histogram(df:pd.DataFrame, feature_names:List[str], classes_names:List[str], output_dir:str):
    '''
    Generates histograms for each feature in the DataFrame, showing the distribution of values for each class.
    :param df: dataframe
    :param feature_names: names of the features to visualize
    :param classes_names: names of the classes
    :param output_dir: output directory where to save the figure
    '''
    for band in feature_names:
        plt.figure(figsize=(10, 6))
        for label in classes_names:
            subset = df[df['classes'] == label][band]
            sns.histplot(subset, label=label)
        plt.title(f'Distribution of {band}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'histogram_{band}.png'))
        plt.show()

def generate_correlation_heatmap(df:pd.DataFrame, feature_names:List[str], output_dir:str):
    '''
    Generates a correlation heatmap for the specified features in the DataFrame.
    :param df: dataframe
    :param feature_names: names of the features to visualize
    :param output_dir: output directory where to save the figure
    '''
    corr = df[feature_names].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Band Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'band_correlation_matrix.png'))
    plt.show()

def generate_class_distribution(df: pd.DataFrame, classes_names:List[str], output_dir:str):
    '''
    Generates a count plot showing the distribution of classes in the DataFrame.
    :param df: dataframe
    :param classes_names: names of the classes
    :param output_dir: output directory where to save the figure
    '''
    plt.figure(figsize=(8, 5))
    sns.countplot(x='classes', data=df, order=classes_names)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_distribution.png'))
    plt.show()

def generate_pairplot(df:pd.DataFrame, feature_names:List[str], output_dir:str):
    '''
    Generates a pairplot for the specified features in the DataFrame, colored by class.
    :param df: dataframe
    :param feature_names: names of the features to visualize
    :param output_dir: output directory where to save the figure
    '''
    plt.figure(figsize=(10, 10))
    sns.pairplot(df[feature_names + ['classes']].sample(5000), hue="classes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.show()

def generate_2d_pca_visualization(df:pd.DataFrame, feature_names:List[str], output_dir:str):
    '''
    Generates a 2D PCA visualization of the specified features in the DataFrame.
    :param df: dataframe
    :param feature_names: names of the features to visualize
    :param output_dir: output directory where to save the figure
    '''
    pca = PCA(n_components=2)
    df_scaled = StandardScaler().fit_transform(df[feature_names])
    X_pca = pca.fit_transform(df_scaled)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='classes', data=df.sample(5000), palette='Set2', alpha=0.6)
    plt.title('2D PCA of Spectral Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_2d_visualization.png'))
    plt.show()