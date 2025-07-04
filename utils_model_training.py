import os
import random
from dataclasses import dataclass
from datetime import time
from time import time
from typing import Callable, List, Optional

import h5py
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from globals import file_path


def get_preprocessor(X):
    """
    Creates a preprocessing pipeline for numeric features in the given dataset.

    The pipeline includes:
    - Imputation of missing values using the mean strategy.
    - Standard scaling of numeric features.

    Args:
        X (pd.DataFrame): The input dataset containing features.

    Returns:
        ColumnTransformer: A transformer that preprocesses numeric features and passes through other columns.
    """
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def log_run_to_mlflow(train_eval_output, train_eval_input, fold, run_name, other_info):
    mlflow.log_figure(train_eval_output.confusion_matrix_figure, f"confusion_matrix_fold_{fold}.png")
    if train_eval_output.feature_importance_figure is not None:
        mlflow.log_figure(train_eval_output.feature_importance_figure, f"feature_importance_fold_{fold}.png")
    mlflow.log_params(other_info)
    mlflow.log_metrics(train_eval_output.fold_results)
    signature = infer_signature(train_eval_input.X_train, train_eval_output.model.predict(train_eval_input.X_train))
    # Log the model, which inherits the parameters and metric
    '''model_info = mlflow.sklearn.log_model(
        sk_model=train_eval_output.model,
        name=run_name,
        signature=signature,
        input_example=train_eval_input.X_train,
        registered_model_name=run_name
    )'''


def preprocess_columns(df, binarize_labels=False):
    with h5py.File(file_path, 'r') as f:
        class_id_to_name = {int(k): v.decode('utf-8') for k, v in zip(f['class_ids'][()], f['class_names'][()])}
        print(class_id_to_name)
    if binarize_labels:
        class_id_to_name = {k: 'Other' if v != 'Cloud' else v for k, v in class_id_to_name.items()}
    df['classes'] = df['classes'].map(class_id_to_name)
    # df['product_id'] = LabelEncoder().fit_transform(df['product_id'].astype(str))
    target_col = 'classes'
    features_to_use = list(filter(lambda x: x.startswith('spectra_'), df.columns))
    X = df[features_to_use]
    y = df[target_col]
    return df, X, y, features_to_use


def preprocess_categorical(X):
    """
    Encodes categorical features in the given dataset using LabelEncoder.

    Args:
        X (pd.DataFrame): The input dataset containing features.

    Returns:
        pd.DataFrame: The dataset with categorical features encoded as integers.
    """
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
    for col in categorical_features:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X


def create_feature_importance_figure(model, features_to_use):
    """
    Generates a bar plot of the top 20 feature importances for a given model.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): The trained model with feature importance attributes.
        features_to_use (list): List of feature names used in the model.

    Returns:
        tuple: A tuple containing:
            - feature_importance_figure (matplotlib.figure.Figure): The generated feature importance plot.
            - feature_imp_df (pd.DataFrame): A DataFrame containing feature names and their importance scores.
    """
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame(
        {'Feature': features_to_use, 'Gini Importance': importances}).sort_values(
        'Gini Importance', ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    sns.barplot(data=feature_imp_df.head(20), x="Gini Importance", y="Feature", ax=ax_imp)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.close(fig_imp)
    return fig_imp, feature_imp_df


def generate_confusion_matrix_figure(y_test, y_pred):
    """
    Generates a confusion matrix plot for the given true and predicted labels.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        matplotlib.figure.Figure: The generated confusion matrix plot.
    """
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax_cm)
    plt.title(f"Confusion Matrix")
    plt.close(fig_cm)
    return fig_cm


def calculate_scores(y_test, y_pred):
    """
    Calculates evaluation metrics for classification models.

    Metrics include:
    - Accuracy
    - Precision (weighted)
    - Recall (weighted)
    - F1 Score (weighted)

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }


@dataclass
class TrainEvalInput:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    train_index: List[int]
    test_index: List[int]
    feature_names: List[str]
    test_product_ids:List[str]


@dataclass
class TrainEvalOutput:
    model: object
    feature_importance_figure: Optional[object]
    test_and_predictions: pd.DataFrame
    confusion_matrix_figure: object
    fold_results: dict


def prepare_fold_data(df: pd.DataFrame, binarize_labels: bool, fold: int) -> TrainEvalInput:
    """
    Prepares training and testing data for a given fold in cross-validation.
    Reads the test product IDs for the specified fold from a CSV file and uses them to split the dataset.
    Splits data into training and testing sets, based on the product ids for the given fold.
    Preprocesses data by scaling numerical features.

    :param df: DataFrame containing the dataset
    :param binarize_labels: Boolean indicating whether to binarize labels
    :param fold: Fold number for cross-validation
    :return: TrainEvalInput dataclass with all input parameters
    """
    df, X, y, features_to_use = preprocess_columns(df, binarize_labels)
    split_file= 'data/project_test_ids_per_fold.csv'
    #If file with test product IDs does not exist, raise an error
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"File {split_file} not found. Please run split_folds.py first.")

    # Read test product IDs for the specified fold
    test_product_ids = pd.read_csv(split_file).query('fold==@fold')['product_id'].tolist()
    df['product_id'] = df['product_id'].astype(str)  # Ensure product_id is string for matching
    test_index = df[df['product_id'].isin(test_product_ids)].index
    train_index = df[~df['product_id'].isin(test_product_ids)].index

    #Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #Preprocess data
    preprocessor = get_preprocessor(X_train)
    X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=features_to_use)
    X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=features_to_use)

    train_eval_input = TrainEvalInput(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=features_to_use,
        train_index=train_index,
        test_index=test_index,
        test_product_ids=test_product_ids
    )
    return train_eval_input




def fix_random_seed(s):
    np.random.seed(s)
    random.seed(s)


def train_and_evaluate_model(train_eval_input: TrainEvalInput, model_class: Callable,
                             model_name: str) -> TrainEvalOutput:
    '''
    Trains and evaluates a classification model and returns evaluation artifacts.

    :param train_eval_input,: TrainEvalInput dataclass with all input parameters
    :return: TrainEvalOutput dataclass with all output artifacts
    '''
    model = model_class()
    start_training_time = time()
    model.fit(train_eval_input.X_train, train_eval_input.y_train)
    end_training_time = time()

    feature_importance_figure = None
    if model_name == "Random Forest":
        feature_importance_figure, _ = create_feature_importance_figure(model, train_eval_input.feature_names)

    y_pred = model.predict(train_eval_input.X_test)
    test_and_predictions = pd.DataFrame(list(zip(list(train_eval_input.y_test), list(y_pred))),
                                        columns=['target', 'predictions'], index=train_eval_input.test_index)

    confusion_matrix_figure = generate_confusion_matrix_figure(train_eval_input.y_test, y_pred)
    fold_results = calculate_scores(train_eval_input.y_test, y_pred)
    fold_results['Training time'] = end_training_time - start_training_time
    return TrainEvalOutput(
        model=model,
        feature_importance_figure=feature_importance_figure,
        test_and_predictions=test_and_predictions,
        confusion_matrix_figure=confusion_matrix_figure,
        fold_results=fold_results

    )
