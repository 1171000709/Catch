# -*- coding: utf-8 -*-

import os
# import random
import time
import logging
import copy
import pandas as pd
import numpy as np
from pipline_thread_2N_batch_singlevalue import Pipline
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer
from pathlib import Path
# from constant import Operation
from feature_engineering.model_base import ModelBase
from utility.base_utility import BaseUtility
from utility.model_utility import ModelUtility, sub_rae
from dataset_split import DatasetSplit
import json
import random
import copy
from feature_engineering.feature_filter import FeatureFilterMath
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")
from xgboost.sklearn import XGBClassifier, XGBRegressor


class GetReward(object):
    def __init__(self, args, do_onehot=False):
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = Path(self.dataset_path).stem
        self.target_col = args.target_col
        self.task_type = args.task_type
        self.eval_method = args.eval_method
        self.continuous_col = args.continuous_col
        self.discrete_col = args.discrete_col
        #self.fe_num_limit = args.fe_num_limit
        if 'f1_average' in vars(args).keys():
            self.f1_average = args.f1_average
        else:
            self.f1_average = None
        self.do_onehot = do_onehot

        self.rep_num = None

        self.dataset_split = DatasetSplit(args)

        self.sample_path = Path('data/sample') / self.dataset_name

        self.pipline_ins = \
            Pipline(self.continuous_col, self.discrete_col, self.do_onehot)

    def filter_math_select(self, operation_idx_dict, x_train, y_train):

        ffm = FeatureFilterMath(operation_idx_dict)
        ffm.var_filter(x_train, threshold=0)

        ori_fe_len = x_train.shape[1]

        # ffm.columns_duplicates()

        # if self.task_type == 'classifier':
        #     ffm.chi2_filter(x_train, y_train, p_threshold=0.01)
        # ffm.mic_filter(x_train, y_train, task_type=self.task_type,
        #                mic_threshold=0)

        ffm.columns_duplicates(x_train)
        ffm.columns_na(x_train)

        ffm.update_delete_res()
        all_delete_idx = ffm.delete_idx_list
        new_train_fes = np.delete(x_train, all_delete_idx, axis=1)

        delete_idx_dict = ffm.delete_idx_dict
        rep_num = len(delete_idx_dict['delete_var_idx']) + \
                  len(delete_idx_dict['delete_duplicates_idx'])
        self.rep_num = rep_num / ori_fe_len

        return new_train_fes, all_delete_idx

    # @staticmethod
    # def remove_duplication_identical(data):
    #     # data = data[:, data.std(axis=0) != 0]
    #     _, idx = np.unique(data, axis=1, return_index=True)
    #     y = data[:, np.sort(idx)]
    #     return y, np.sort(idx)

    def feature_pipline_train(self, actions_trans, data):
        # Form a new dataset through actions on train
        new_fes_shape, operation_idx_dict = self.pipline_ins.calculate_shape(
            actions_trans, data, target_col=self.target_col)

        new_train_fes, train_label = self.pipline_ins.create_action_fes(
            actions=actions_trans, ori_dataframe=data,
            task_type=self.task_type, target_col=self.target_col, train=True)

        fe_params = self.pipline_ins.fes_eng.get_train_params()
        # print('fe_params', fe_params)
        # print('new_train_fes', new_train_fes)
        # print('operation_idx_dict', operation_idx_dict)
        if len(operation_idx_dict['ori_continuous_idx']):
            max_con = -1
        else:
            if len(operation_idx_dict['ori_continuous_idx']) == 0:
                max_con = -1
            else:
                max_con = max(operation_idx_dict['ori_continuous_idx'])
        operation_idx_dict['ori_discrete_idx'] = list(range(max_con+1,new_train_fes.shape[1]))
        return new_train_fes, train_label, fe_params, operation_idx_dict

    def feature_pipline_infer(self, fe_params, actions_trans, data):
        # Form a new dataset through actions on test
        new_test_fes, test_label = self.pipline_ins.create_action_fes(
            actions=actions_trans, ori_dataframe=data,
            task_type=self.task_type, target_col=self.target_col,
            train=False, train_params=fe_params)
        self.pipline_ins.fes_eng.clear_train_params()
        return new_test_fes, test_label

    def get_lr_model(self):
        #
        if self.task_type == 'classifier':
            model = ModelBase.lr_classify()
        elif self.task_type == 'regression':
            model = ModelBase.lr_regression()
        else:
            logging.info(f'er')
            model = None
        return model

    def get_svm_model(self):
        # 选择模型
        if self.task_type == 'classifier':
            model = ModelBase.svm_liner_svc()
        elif self.task_type == 'regression':
            model = ModelBase.svm_liner_svr()
        else:
            logging.info(f'er')
            model = None
        return model

    def get_rf_model(self, hyper_param):

        #
        if self.task_type == 'classifier':
            model = ModelBase.rf_classify()
        elif self.task_type == 'regression':
            model = ModelBase.rf_regeression()
        else:
            logging.info(f'er')
            model = None
        if hyper_param is not None and model is not None:
            model.set_params(**hyper_param)
        return model

    def get_xgb_model(self):
        #
        if self.task_type == 'classifier':
            model = ModelBase.xgb_classify()
        elif self.task_type == 'regression':
            model = ModelBase.xgb_regression()
        else:
            logging.info(f'er')
            model = None
        return model

    def train_test_score(self, train_data, test_data, action_trans,
                         math_select=True):

        train_data_copy = copy.deepcopy(train_data)
        test_data_copy = copy.deepcopy(test_data)
        #
        res_tuple = self.feature_pipline_train(action_trans, train_data_copy)
        if res_tuple is None:
            return None
        new_train_fes, train_label, fe_params, operation_idx_dict = res_tuple

        #
        new_test_fes, test_label = self.feature_pipline_infer(
            fe_params, action_trans, test_data_copy)

        #
        # print('action_trans', action_trans)
        if math_select:
            # math, mic
            new_train_fes, all_delete_idx = self.filter_math_select(
                operation_idx_dict, new_train_fes, train_label)
            new_test_fes = np.delete(new_test_fes, all_delete_idx, axis=1)

        # new_test_fes = new_test_fes[0]
        # print('：', new_train_fes.shape, new_test_fes.shape)

        #
        #model = self.get_lr_model()
        model = self.get_rf_model(None)
        model.fit(new_train_fes, train_label)

        #
        valid_score = self.predict_score(new_test_fes, test_label, model)
        # valid_score = valid_score - repeat_pun
        # valid_score = self.predict_score(raw_fes, raw_label, model)

        return valid_score

    def gbdt_improve(self, train_data, train_label, test_data, test_label,
                     operation_idx_dict):
        from xgb_lr import XGB_LR
        d_len = len(operation_idx_dict['ori_discrete_idx'])

        c_len = train_data.shape[1] - d_len
        c_indexs = list(range(c_len))
        d_indexs = list(range(c_len, train_data.shape[1]))

        result_list = []
        xgb_params = {
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimator': 800,
            'use_label_encoder': False,
            'verbosity': 0,
            'scale_pos_weight': None,
            'reg_lambda': 0.05,
            'min_child_weight': 3}

        max_depth_list = [i for i in range(1, 10, 2)]
        min_child_weight_list = [2, 4, 6, 8, 9]
        learn_rate_list = [0.01, 0.05, 0.1]
        # xgb_params_list = []
        # for single_max_depth in max_depth:
        #     new_params = copy.deepcopy(xgb_params)
        #     new_params['max_depth'] = single_max_depth
        #     xgb_params_list.append(new_params)

        max_depth_result_list = []
        for single_max_depth in max_depth_list:

            train_data_copy = copy.deepcopy(train_data)
            train_label_copy = copy.deepcopy(train_label)
            test_data_copy = copy.deepcopy(test_data)
            test_label_copy = copy.deepcopy(test_label)

            xgb_params['max_depth'] = single_max_depth
            for single_child_weight in min_child_weight_list:
                xgb_params['min_child_weight'] = single_child_weight
                for single_lr in learn_rate_list:
                    xgb_params['learning_rate'] = single_lr

                    if 'f1_score' in self.eval_method:
                        xgblr = XGB_LR(train_data_copy, train_label_copy, c_indexs, d_indexs,
                                       xgb_params, score_type='f1')
                        gbdt_fit_params = {'eval_metric': 'logloss'}
                    else:
                        xgblr = XGB_LR(train_data_copy, train_label_copy, c_indexs, d_indexs,
                                       xgb_params, score_type=self.eval_method)
                        gbdt_fit_params = {'eval_metric': 'auc'}

                    linear_fit_params = {}
                    model = xgblr.fit_valid(gbdt_fit_params, linear_fit_params)

                    score = self.predict_score(test_data_copy, test_label_copy, model)
                    max_depth_result_list.append(score)

        #
        max_res = max(max_depth_result_list)
        print('max_depth_result_list', max_depth_result_list)
        print('max_res', max_res)
        exit()
        # depth_index = max_depth_result_list.index(max_res)
        # opt_max_depth = max_depth[depth_index]
        # print('opt_max_depth', opt_max_depth)
        # print('max_depth_result_list', max_depth_result_list)
        #
        # xgb_params['max_depth'] = opt_max_depth

        learn_rate_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]

        learning_rate_list = []
        for single_learning_rate in learn_rate_list:
            xgb_params['learning_rate'] = single_learning_rate
            train_data_copy = copy.deepcopy(train_data)
            train_label_copy = copy.deepcopy(train_label)
            test_data_copy = copy.deepcopy(test_data)
            test_label_copy = copy.deepcopy(test_label)
            if 'f1_score' in self.eval_method:
                xgblr = XGB_LR(train_data_copy, train_label_copy, c_indexs, d_indexs,
                               xgb_params, score_type='f1')
                gbdt_fit_params = {'eval_metric': 'logloss'}
            else:
                xgblr = XGB_LR(train_data_copy, train_label_copy, c_indexs, d_indexs,
                               xgb_params, score_type=self.eval_method)
                gbdt_fit_params = {'eval_metric': 'auc'}

            linear_fit_params = {}
            model = xgblr.fit_valid(gbdt_fit_params, linear_fit_params)

            score = self.predict_score(test_data_copy, test_label_copy, model)
            learning_rate_list.append(score)
        print('learning_rate_list', learning_rate_list)

        max_res_lr = max(learning_rate_list)
        max_res_lr_index = learning_rate_list.index(max_res_lr)
        print('max_res_lr_index', max_res_lr_index)
        opt_learning_rate = learn_rate_list[max_res_lr_index]

        xgb_params['learning_rate'] = opt_learning_rate
        print('xgb_params', xgb_params)

        n_estimator_list = [2, 3, 4, 5, 6, 7, 8, 9]
        estimator_res_list = []
        for single_estimator in n_estimator_list:
            xgb_params['min_child_weight'] = single_estimator
            train_data_copy = copy.deepcopy(train_data)
            train_label_copy = copy.deepcopy(train_label)
            test_data_copy = copy.deepcopy(test_data)
            test_label_copy = copy.deepcopy(test_label)
            if 'f1_score' in self.eval_method:
                xgblr = XGB_LR(train_data_copy, train_label_copy, c_indexs, d_indexs,
                               xgb_params, score_type='f1')
                gbdt_fit_params = {'eval_metric': 'logloss'}
            else:
                xgblr = XGB_LR(train_data_copy, train_label_copy, c_indexs, d_indexs,
                               xgb_params, score_type=self.eval_method)
                gbdt_fit_params = {'eval_metric': 'auc'}

            linear_fit_params = {}
            model = xgblr.fit_valid(gbdt_fit_params, linear_fit_params)

            score = self.predict_score(test_data_copy, test_label_copy, model)
            estimator_res_list.append(score)
        print('estimator_res_list', estimator_res_list)
        exit()

        return score

    def xgb_lr_score(self, train_data, test_data, action_trans,
                     math_select=True):
        train_data_copy = copy.deepcopy(train_data)
        test_data_copy = copy.deepcopy(test_data)
        if action_trans is None:
            model = self.get_rf_model(None)
            model.fit(train_data_copy, train_data_copy[self.target_col])
            score = self.predict_score(test_data_copy, test_data_copy[self.target_col], model)
            return score
        #
        res_tuple = self.feature_pipline_train(action_trans, train_data_copy)
        if res_tuple is None:
            return None
        new_train_fes, train_label, fe_params, operation_idx_dict = res_tuple

        #
        new_test_fes, test_label = \
            self.feature_pipline_infer(fe_params, action_trans, test_data_copy)

        if math_select:
            # math, mic
            new_train_fes, all_delete_idx = self.filter_math_select(
                operation_idx_dict, new_train_fes, train_label)
            new_test_fes = np.delete(new_test_fes, all_delete_idx, axis=1)

        #
        model = self.get_rf_model(None)
        #model = self.get_xgb_model()
        # model = XGBRegressor(max_depth=6, learning_rate="0.1",
        #                              n_estimators=600, verbosity=0, subsample=0.8,
        #                              colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1, n_jobs=-1)
        # from xgb_lr import XGB_LR
        # d_len = len(operation_idx_dict['ori_discrete_idx'])
        #
        # c_len = new_train_fes.shape[1] - d_len
        # c_indexs = list(range(c_len))
        # d_indexs = list(range(c_len, new_train_fes.shape[1]))
        #
        # xgb_params = {
        #     'max_depth': 5,
        #     'learning_rate': 0.05,
        #     'n_estimator': 500,
        #     'use_label_encoder': False,
        #     'verbosity': 0,
        #     'scale_pos_weight': None,
        #     'reg_lambda': 0.05,
        #     'min_child_weight': 3}
        #
        # if 'f1_score' in self.eval_method:
        #     xgblr = XGB_LR(new_train_fes, train_label, c_indexs, d_indexs,
        #                    xgb_params, score_type='f1')
        #     gbdt_fit_params = {'eval_metric': 'logloss'}
        # else:
        #     xgblr = XGB_LR(new_train_fes, train_label, c_indexs, d_indexs,
        #                    xgb_params, score_type=self.eval_method)
        #     gbdt_fit_params = {'eval_metric': 'auc'}
        #
        # linear_fit_params = {}
        # model = xgblr.fit_valid(gbdt_fit_params, linear_fit_params)
        model.fit(new_train_fes, train_label)
        score = self.predict_score(new_test_fes, test_label, model)

        return score
    def k_fold_score(self, search_data, action_trans, hp_action=None,
                     math_select=True, is_base=False):

        #
        search_data_copy = copy.deepcopy(search_data)
        res_tuple = self.feature_pipline_train(action_trans, search_data_copy)
        if res_tuple is None:
            return None
        search_fes, search_label, fe_params, operation_idx_dict = res_tuple

        #
        if math_select and (not is_base):
            # math, mic
            search_fes, all_delete_idx = self.filter_math_select(
                operation_idx_dict, search_fes, search_label)

        # if self.task_type == 'classifier':
        #     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # elif self.task_type == 'regression':
        #     skf = KFold(n_splits=5, shuffle=True, random_state=42)
        # else:
        #     print('task_type ')
        #     skf = None

        model = self.get_rf_model(hp_action)
        if self.task_type == 'classifier':
            #score_list = cross_val_score(model, search_fes, search_label, scoring="f1_micro", cv=5)
            if not self.args.coreset:
                score_list = cross_val_score(model, search_fes, search_label, scoring="f1_micro", cv=5)
            else:
                score_list = cross_val_score(model, search_fes, search_label, scoring="f1_micro", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10))
        else:
            subrae = make_scorer(sub_rae, greater_is_better=True)
            #score_list = cross_val_score(model, search_fes, search_label, scoring=subrae, cv=5)
            if not self.args.coreset:
                score_list = cross_val_score(model, search_fes, search_label, scoring=subrae, cv=5)
            else:
                score_list = cross_val_score(model, search_fes, search_label, scoring=subrae, cv=KFold(n_splits=5, shuffle=True, random_state=10))
        # for trn_idx, val_idx in skf.split(search_fes, search_label):
        #     train_data = search_fes[trn_idx, :]
        #     train_label = search_label[trn_idx]
        #
        #     valid_data = search_fes[val_idx, :]
        #     valid_label = search_label[val_idx]
        #
        #     #
        #     model = self.get_rf_model(hp_action)
        #     # model = self.get_svm_model()
        #     #model = self.get_lr_model()
        #     # model = self.get_xgb_model()
        #     model.fit(train_data, train_label)
        #
        #     #
        #     valid_score = self.predict_score(valid_data, valid_label, model)
        #
        #     # valid_score = self.gbdt_improve(train_data, train_label,
        #     #                                 valid_data, valid_label,
        #     #                                 operation_idx_dict)
        #
        #     score_list.append(valid_score)
        if is_base:
            return score_list
        else:
            return score_list, search_fes.shape[1]

    def train_test_cv_score(self, train_data, test_data, action_trans,
                            math_select=True):
        train_data_copy = copy.deepcopy(train_data)
        test_data_copy = copy.deepcopy(test_data)
        #
        res_tuple = self.feature_pipline_train(action_trans, train_data_copy)
        if res_tuple is None:
            return None
        new_train_fes, train_label, fe_params, operation_idx_dict = res_tuple

        #
        new_test_fes, test_label = \
            self.feature_pipline_infer(fe_params, action_trans, test_data_copy)

        #
        # new_train_fes = pd.DataFrame(new_train_fes)
        # new_train_fes.to_csv('new_train_fes.csv')
        # print('new_train_fes', new_train_fes.shape)
        # print('action_trans', action_trans)
        if math_select:
            # math 特征筛选: 方差，卡方，mic
            # from utility.base_utility import BaseUtility
            #
            # BaseUtility.line_profiler(self.filter_math_select,
            #                 (operation_idx_dict, new_train_fes, train_label))

            new_train_fes, all_delete_idx = self.filter_math_select(
                operation_idx_dict, new_train_fes, train_label)
            new_test_fes = np.delete(new_test_fes, all_delete_idx, axis=1)

        #
        # model = self.get_lr_model()
        model = self.get_rf_model()
        # model = self.get_xgb_model()

        #
        model.fit(new_train_fes, train_label)

        #
        valid_score = self.predict_score(new_test_fes, test_label, model)

        return model, valid_score

    def predict_score(self, data, label, model):
        #
        if self.eval_method == 'ks' or self.eval_method == 'auc':
            y_pred = model.predict_proba(data)[:, 1]
        else:
            y_pred = model.predict(data)
        # np.set_printoptions(suppress=True)
        # np.savetxt('ans3.csv', y_pred, delimiter=',')
        score = ModelUtility.model_metrics(
            y_pred, label, self.task_type, self.eval_method, self.f1_average)

        return score


if __name__ == '__main__':
    pass
