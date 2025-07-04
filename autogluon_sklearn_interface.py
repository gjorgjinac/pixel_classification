import numpy as np
import pandas as pd
import autogluon as ag
import sklearn
from autogluon.core.metrics import f1, make_scorer
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

def macro_f1(y_true, y_pred):
    """
    Calculate the macro F1 score.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Macro F1 score
    """
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro')

#Credits:https://github.com/autogluon/autogluon/issues/1493
class AutogluonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, eval_metric=None, time_limit=3600, hyperparameters=None, verbosity=2):
        if eval_metric is None:
            eval_metric = make_scorer("macrof1", macro_f1, needs_class=True, needs_pos_label=True)
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.hyperparameters = hyperparameters
        self.verbosity = verbosity
        self.tabular_predictor = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        X = pd.DataFrame(X)
        X['target'] = list(y.values)
        print(y.replace([np.inf,-np.inf],np.nan).isna().sum())
        self.tabular_predictor = TabularPredictor(label='target', eval_metric=self.eval_metric, problem_type='multiclass')
        self.tabular_predictor.fit(train_data=X, time_limit=self.time_limit, hyperparameters=self.hyperparameters, verbosity=self.verbosity)

        self.classes_ = self.tabular_predictor._learner.label_cleaner.ordered_class_labels
        self.n_classes_ = len(self.classes_)

    def predict(self, X):
        if self.tabular_predictor is None:
            raise NotFittedError("Model has not been fitted. Call 'fit' before 'predict'.")
        return self.tabular_predictor.predict(X)

    def predict_proba(self, X):
        if self.tabular_predictor is None:
            raise NotFittedError("Model has not been fitted. Call 'fit' before 'predict_proba'.")
        return self.tabular_predictor.predict_proba(X)