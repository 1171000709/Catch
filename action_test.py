# -*- coding: utf-8 -*-
import copy

import pandas as pd

from dataset_split import DatasetSplit
from utility.base_utility import BaseUtility
from get_reward import GetReward
from constant import NO_ACTION
import numpy as np
from search_business import SearchBusiness
import shap


def test_action(_args, action_trans, with_gbdt=False):
    """
    Get the effect of the action in different data splitting situations
    :param _args:
    :param action_trans: Action of test
    :return:
    """

    # Use 10 different data splitting methods to test the performance before and after the action
    improve_values = []
    baseline_score = []
    actions_fe_score = []
    for i in range(1):

        dataset_split =  DatasetSplit(_args)
        search_data = dataset_split.load_all_data()

        keep_data = pd.read_csv('data/house_prices_test.csv')
        keep_data = keep_data[_args.continuous_col + _args.discrete_col + [_args.target_col]]
        print('all_data shape',keep_data.shape)

        print('去重之前 shape', keep_data.shape)
        keep_data.drop_duplicates(keep='first', inplace=True)
        keep_data.reset_index(drop=True, inplace=True)
        print('去重之后 shape', keep_data.shape)
        # search_data_cc = copy.deepcopy(search_data)
        # merge_dict = BaseUtility.get_merge_dict(
        #     search_data, _args.discrete_col)
        # print('merge_dict', merge_dict)
        #
        # search_data_m = BaseUtility.merge_categories(search_data, merge_dict)
        # keep_data_m = BaseUtility.merge_categories(keep_data, merge_dict)

        # search_data_m = search_data
        # keep_data_m = keep_data

        # dataset_split.split_search_keep_data(random_state=i)
        get_reward_ins = GetReward(_args)

        # no action
        # if with_gbdt:
        #     no_actions_score = get_reward_ins.xgb_lr_score(
        #         search_data, keep_data, NO_ACTION)
        # else:
        #     model, no_actions_score = get_reward_ins.train_test_cv_score(
        #         search_data, keep_data, NO_ACTION)
        # no_actions_score = get_reward_ins.xgb_lr_score(
        #     search_data, keep_data, NO_ACTION)

        # df_feature_train = search_data.drop(_args.target_col, axis=1)
        # df_feature_train_c = copy.deepcopy(df_feature_train)
        # explainer = shap.TreeExplainer(model)
        # shap_values_train = explainer(df_feature_train)
        # shap.plots.bar(shap_values_train)

        # from pandas.testing import assert_frame_equal
        # assert_frame_equal(search_data, search_data_m)

        # action
        if with_gbdt:
            actions_score = get_reward_ins.xgb_lr_score(
                search_data, keep_data, action_trans)
        # else:
        #     model, actions_score = get_reward_ins.train_test_cv_score(
        #         search_data, keep_data, action_trans)
        # actions_score = get_reward_ins.xgb_lr_score(
        #     search_data, keep_data, action_trans)

        # from pandas.testing import assert_frame_equal
        # assert_frame_equal(df_feature_train_c, df_feature_train)

        # explainer = shap.TreeExplainer(model)
        # # explainer = shap.LinearExplainer(model)
        # shap_values_train = explainer(df_feature_train)
        # shap.plots.bar(shap_values_train)

        # improve_values.append(actions_score - no_actions_score)
        # baseline_score.append(no_actions_score)
        # actions_fe_score.append(actions_score)
        #
        # print('no_actions_score: ', no_actions_score)
        # print('actions_score: ', actions_score)

    # baseline_score_mean = np.mean(baseline_score)
    # print(f' baseline ：{baseline_score}, '
    #       f'mean：{baseline_score_mean}')
    #
    # actions_score_mean = np.mean(actions_fe_score)
    # print(f'action z：{actions_fe_score}， '
    #       f'mean：{actions_score_mean}')
    #
    # improve_values_mean = np.mean(improve_values)
    # print(f'improve：{improve_values}, mean：{improve_values_mean}')


if __name__ == '__main__':
    # 第二轮测试
    # load_args = 'water_potability'
    # load_args = 'mobile_pricerange_train'
    # load_args = 'winequality_red'
    # load_args = 'Placement_Data_Full_Class'

    # load_args = 'adult_dataset'
    # load_args = 'titanic'
    # load_args = 'Customer_Segmentation'
    # load_args = 'titanic'
    # load_args = 'pre_data_hzd'
    # load_args = 'hzd_fillna'
    # load_args = 'p_hzd'
    # load_args = 'bank_add'
    # load_args = 'club_loan'
    # load_args = 'hzd_amend'
    # load_args = 'default_credit_card'
    # load_args = 'SPECTF'
    # load_args = 'winequality-white'
    # load_args = 'winequality_red'
    # load_args = 'messidor_features'
    #load_args = 'ionosphere'
    # load_args = 'credit-a'
    # load_args = 'PimaIndian'
    load_args = 'house_prices'
    from main import get_args
    args_ = get_args(load_args)
    actions = ""
    # ============================================================
    # from utility.base_utility import BaseUtility
    # BaseUtility.line_profiler(test_action, (args_, actions))
    # ============================================================

    test_action(args_, actions, with_gbdt=True)
