from feature_selector.feature_selector import FeatureSelector
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd


# def lgb_f1_score(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
#     return 'f1', f1_score(y_true, y_hat), True


def feature_filter(df_fe, df_label, task):
    fs = FeatureSelector(data=df_fe, labels=df_label)
    #
    fs.identify_missing(missing_threshold=0.6)
    #
    fs.identify_collinear(correlation_threshold=0.98)
    #
    # unique_target = np.unique(df_label.values)
    # if task == 'classifier' and len(unique_target) == 2:
    #     fs.identify_zero_importance(task='classification',
    #                                 eval_metric='binary_logloss',
    #                                 n_iterations=5,
    #                                 early_stopping=False)
    # elif task == 'classifier' and len(unique_target) > 2:
    #     fs.identify_zero_importance(task='classification',
    #                                 eval_metric='multi_logloss',
    #                                 n_iterations=5,
    #                                 early_stopping=False)
    # elif task == 'regression':
    #     fs.identify_zero_importance(task='regression',
    #                                 eval_metric='l2',
    #                                 n_iterations=5,
    #                                 early_stopping=False)
    # else:
    #     pass
    #
    # fs.identify_low_importance(cumulative_importance=0.99)
    train_removed = fs.remove(methods=['missing', 'collinear'])

    return train_removed.values, df_label.values


