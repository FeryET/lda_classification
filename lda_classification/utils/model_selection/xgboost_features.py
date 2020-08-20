from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from tqdm import tqdm
from xgboost.sklearn import XGBClassifier, XGBRegressor
from xgboost.plotting import plot_importance
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def _evaluate_thresholds(model, thresholds, x_train, y_train, x_test=None,
                         y_test=None):
    results = []
    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_x_train = selection.transform(x_train)
        selection_model = XGBClassifier()
        selection_model.fit(select_x_train, y_train)
        select_x_test = selection.transform(x_test)
        predictions = selection_model.predict(select_x_test)
        acc = accuracy_score(y_test, predictions)
        results.append(
                {"n_features": select_x_train.shape[1], "threshold": thresh,
                 "accuracy": acc * 100.0})
    return results


def _optimal_values(results):
    df = pd.DataFrame(results)

    df["count"] = 1
    df = df.groupby("n_features").sum()
    df = df[df["count"] > 5]
    df[["threshold", "accuracy"]] = df[["threshold", "accuracy"]].div(
            df["count"], axis=0)
    df = df.reset_index()
    df.sort_values("accuracy", ascending=False, ignore_index=True, inplace=True)
    n_features = df["n_features"][0]
    threshold = df["threshold"][0]
    return n_features, threshold


class XGBoostFeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, n_repeats=5, n_splits=10, **kwargs):
        """
        :param n_repeats: number of repeats for inner KFold crossvalidation
        :param n_splits: number of splits for inner KFold crossvalidation
        :param kwargs: parameters for training the inner XGBClassifer model.
        """
        self.model = XGBClassifier(**kwargs)
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.selected_indexes = None

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def fit(self, X, y):
        if y is None:
            raise ValueError(
                    "y should be provided, since this is a supervised method.")

        folds = RepeatedStratifiedKFold(n_repeats=self.n_repeats,
                                        n_splits=self.n_splits)
        scores = []
        for train_idx, test_idx in tqdm(folds.split(X, y),
                                        desc="Feature Selection",
                                        total=self.n_repeats * self.n_splits):
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]
            self.model.fit(x_train, y_train)
            thresholds = sorted(set(self.model.feature_importances_))
            scores.extend(_evaluate_thresholds(self.model, thresholds, x_train,
                                               y_train, x_test, y_test))
        optimal_n_features, optimal_threshold = _optimal_values(scores)
        self.model.fit(X, y)
        importances = sorted(list(enumerate(self.model.feature_importances_)),
                             key=lambda x: x[1], reverse=True)
        self.selected_indexes, _ = list(zip(*importances[:optimal_n_features]))
        self.selected_indexes = np.array(self.selected_indexes)
        return self

    def transform(self, X):
        if self.selected_indexes is None:
            raise RuntimeError("You should train the feature selector first.")
        return X[:, self.selected_indexes]

    def plot_importance(self, *args, **kwargs):
        """
        Checkout xgboost.plotting.plot_importance for a list of arguments
        :param args:
        :param kwargs:
        :return:
        """
        return plot_importance(self.model, *args, **kwargs)
