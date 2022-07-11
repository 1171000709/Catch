import random
import copy
import json
import logging

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, \
    r2_score, accuracy_score, mean_absolute_error
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from collections import OrderedDict
from utility.base_utility import BaseUtility
from sklearn.model_selection import GridSearchCV
# from skopt import BayesSearchCV
# from sklearn.model_selection import StratifiedKFold
# import xgboost as xgb
# import pandas as pd


# def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
#                          n_iter=50, scoring=None, fit_params=None, n_jobs=1,
#                          n_points=1, iid=True, refit=True, cv=None, verbose=0,
#                          pre_dispatch='2*n_jobs', random_state=None,
#                          error_score='raise', return_train_score=False):
#
#     self.search_spaces = search_spaces
#     self.n_iter = n_iter
#     self.n_points = n_points
#     self.random_state = random_state
#     self.optimizer_kwargs = optimizer_kwargs
#     self._check_search_space(self.search_spaces)
#     self.fit_params = fit_params
#
#     super(BayesSearchCV, self).__init__(
#         estimator=estimator, scoring=scoring,
#         n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
#         pre_dispatch=pre_dispatch, error_score=error_score,
#         return_train_score=return_train_score)
#
# BayesSearchCV.__init__ = bayes_search_CV_init

def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum([np.abs(y_mean - y[i]) for i in range(len(y))])
    res = 1 - rae
    return res


class ModelUtility:
    @classmethod
    def model_metrics(cls, y_pred, y_real, task_type, eval_method, f1_average=None):
        if task_type == 'classifier':
            if eval_method == 'f1_score':
                unique_target = np.unique(y_real.reshape(len(y_real)))
                if len(unique_target) > 2:
                    # average = 'macro'
                    average = 'micro'
                else:
                    if f1_average:
                        average = f1_average
                    else:
                        average = 'binary'
                score = f1_score(y_real, y_pred, average=average)
            elif eval_method == 'acc':
                score = accuracy_score(y_real, y_pred)
            elif eval_method == 'ks':
                # print('y_pred', y_pred)
                # print(pd.value_counts(y_real))
                fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
                score = max(abs(fpr-tpr))

            elif eval_method == 'auc':
                fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
                score = auc(fpr, tpr)
                # score = roc_auc_score(y_real, y_pred)
            elif eval_method == 'confusion_matrix':
                cm = confusion_matrix(y_real, y_pred)
                ac = (cm[0, 0] + cm[1, 1]) / \
                     (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
                sp = cm[1, 1] / (cm[1, 0] + cm[1, 1])
                score = 0.5 * ac + 0.5 * sp
                # print('cm', score, cm, ac, sp)
            else:
                logging.info(f'er')
                score = None
        elif task_type == 'regression':
            if eval_method == 'mse':
                score = mean_squared_error(y_real, y_pred)
            elif eval_method == 'r_squared':
                score = r2_score(y_real, y_pred)
            elif eval_method == 'mae':
                score = mean_absolute_error(y_real, y_pred)
            elif eval_method == 'sub_rae':
                score = sub_rae(y_real, y_pred)
            else:
                logging.info(f'er')
                score = None
        else:
            logging.info(f'er')
            score = None
        return score

    # @classmethod
    # def xgb_cv_me

    @classmethod
    def drop_actions(cls, actions_dict, drop_prod, drop_num):
        #  actions
        order_actions = OrderedDict(actions_dict)
        for key, value in order_actions.items():
            if isinstance(value, dict):
                order_actions[key] = OrderedDict(value)

        actions_ops_items = []
        for index1, value1 in enumerate(order_actions):
            for index2, value2 in enumerate(order_actions[value1]):
                actions_ops_items.append((index1, index2))
        #  drop
        random_prod = random.uniform(0, 1)
        if random_prod < drop_prod:
            #  drop ops
            order_actions_copy = copy.deepcopy(order_actions)
            drop_items = random.sample(actions_ops_items, drop_num)
            for drop_item in drop_items:
                for index1, value1 in enumerate(order_actions):
                    for index2, value2 in enumerate(order_actions[value1]):
                        if drop_item == (index1, index2):
                            if isinstance(order_actions[value1], list):
                                order_actions_copy[value1].remove(value2)
                            else:
                                order_actions_copy[value1].pop(value2)
                                if value1 == 'continuous_bins':
                                    BaseUtility.list_remove_element(
                                        order_actions_copy['combine_2'],
                                        value2)
                                    BaseUtility.list_remove_element(
                                        order_actions_copy['combine_3'],
                                        value2)
                                    BaseUtility.list_remove_element(
                                        order_actions_copy['combine_4'],
                                        value2)
            #
            actions_drop_trans = json.loads(json.dumps(order_actions_copy))
        else:
            actions_drop_trans = json.loads(json.dumps(order_actions))
        return actions_drop_trans

    @classmethod
    def xgb_grid_search(cls, data_x, data_y, model, eval_method):

        parameters = {
            'max_depth': [3, 5, 7, 8, 10],
            'min_child_weight': [0, 2, 3, 5, 8],
            # 'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [200, 300, 500, 600],
            # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
            # 'subsample': [0.8, 0.85, 0.95],
            # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
            # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
            # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
        }
        scoring = cls.scoring_trans(eval_method, data_y)
        # if eval_method == 'f1_score':
        #     unique_target = np.unique(data_y.reshape(len(data_y)))
        #     if len(unique_target) > 2:
        #         scoring = 'f1_macro'
        #     else:
        #         scoring = 'f1'
        # elif eval_method == 'accuracy':
        #     scoring = 'accuracy'
        # elif eval_method == 'mse':
        #     scoring = 'neg_mean_squared_error'
        # elif eval_method == 'r_squared':
        #     scoring = 'r2'
        # else:
        #     scoring = None

        grid_search = GridSearchCV(
            model, param_grid=parameters, scoring=scoring, cv=3)
        grid_search.fit(data_x, data_y)

        best_score = grid_search.best_score_
        # print("Best score: %0.3f" % best_score)
        # print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
        # best_model = grid_search.best_estimator_
        return grid_search

    @classmethod
    def scoring_trans(cls, eval_method, label):
        if eval_method == 'f1_score':
            unique_target = np.unique(label.reshape(len(label)))
            if len(unique_target) > 2:
                scoring = 'f1_macro'
            else:
                scoring = 'f1'
        elif eval_method == 'accuracy':
            scoring = 'accuracy'
        elif eval_method == 'mse':
            scoring = 'neg_mean_squared_error'
        elif eval_method == 'r_squared':
            scoring = 'r2'
        else:
            scoring = None
        return scoring

    # @classmethod
    # def bayes_search_cv(cls, data_x, data_y, test_x, test_y, model, eval_method):
    #     scoring = cls.scoring_trans(eval_method, data_y)
    #     bayes_cv_tuner = BayesSearchCV(
    #         estimator=model,
    #         search_spaces={
    #             'learning_rate': (0.01, 0.1, 'log-uniform'),
    #             'min_child_weight': (0, 6),
    #             'max_depth': (3, 8),
    #             # 'max_delta_step': (0, 20),
    #             # 'subsample': (0.01, 1.0, 'uniform'),
    #             # 'colsample_bytree': (0.01, 1.0, 'uniform'),
    #             # 'colsample_bylevel': (0.01, 1.0, 'uniform'),
    #             # 'reg_lambda': (1e-9, 1000, 'log-uniform'),
    #             # 'reg_alpha': (1e-9, 1.0, 'log-uniform'),
    #             # 'gamma': (1e-9, 0.5, 'log-uniform'),
    #             'n_estimators': (50, 500),
    #             # 'scale_pos_weight': (1e-6, 500, 'log-uniform')
    #         },
    #         scoring=scoring,
    #         cv=StratifiedKFold(
    #             n_splits=3,
    #             shuffle=True,
    #             random_state=0),
    #         n_jobs=3,
    #         n_iter=10,
    #         verbose=0,
    #         refit=True,
    #         random_state=42,
    #         fit_params={
    #             'early_stopping_rounds': 20,
    #             'eval_set': [(test_x, test_y)],
    #             'verbose': 0
    #         }
    #     )
    #
    #     bayes_cv_tuner.fit(data_x, data_y)
    #     #
    #     # pd.set_option('display.max_columns', None)
    #     #
    #     # pd.set_option('display.max_rows', None)
    #     # print(pd.DataFrame(bayes_cv_tuner.cv_results_))
    #     # print(zz.total_iterations)
    #     return bayes_cv_tuner


