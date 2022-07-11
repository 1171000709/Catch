import copy
import logging

from dataset_split import DatasetSplit
from utility.base_utility import BaseUtility
from get_reward import GetReward
from constant import NO_ACTION
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from random import choice, randint
from sklearn.model_selection import StratifiedKFold


class SearchBusiness(object):

    def __init__(self, args):
        self.args = args
        pass

    @staticmethod
    def detect_outliers2(list_data):

        # 1st quartile (25%)
        q_1 = np.percentile(list_data, 35)
        # 3rd quartile (75%)
        q_3 = np.percentile(list_data, 85)
        print(q_1, q_3)

        iqr = q_3 - q_1

        # outlier step
        normal_data = []
        for data in list_data:
            if (data > q_1) and (data < q_3):
                normal_data.append(data)

        return normal_data

    def data_process(self, df_data):
        """

        :param df_data:
        :return:
        """
        #
        # print(df_data.shape)
        # print(df_data.duplicated())
        # df_data.drop_duplicates()
        # print(df_data.shape)
        #
        merge_dict = BaseUtility.get_merge_dict(df_data, self.args.discrete_col)
        print('merge_dict', merge_dict)
        df_data = BaseUtility.merge_categories(df_data, merge_dict)

        #

        return df_data

    def get_random_split_data(self, df_data, arch_epoch):
        # random_state = arch_epoch % 5
        # df_data_c = copy.deepcopy(df_data)
        # random_state = randint(0, 1)
        random_state = 1
        # split_ratio = [0.3, 0.4, 0.5, 0.6]
        split_ratio = [0.3, 0.4, 0.5, 0.6, 0.7]
        index_ratio = arch_epoch % 5
        choice_ratio = split_ratio[index_ratio]
        # choice_ratio = choice(split_ratio)
        print('a：', random_state, choice_ratio)

        #  merge
        # df_data = self.data_process(df_data)
        # discrete_ca_num = \
        #     BaseUtility.get_discrete_ca_num(df_data, self.args.discrete_col)

        #
        data_split = DatasetSplit(self.args)
        get_reward_ins = GetReward(self.args)


        train_data, test_data = data_split.split_dataset_with_ratio(
            df_data, choice_ratio, random_state=random_state)


        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        # raw_search_data = self.get_raw_search_data()

        return {
            'train_data': train_data,
            'test_data': test_data,
            # 'discrete_ca_num': discrete_ca_num
        }

    def get_k_fold_split_data(self, df_data):

        split_dataset_list = []
        df_data_copy = copy.deepcopy(df_data)
        df_target = df_data_copy[self.args.target_col]
        df_feature = df_data_copy.drop(self.args.target_col, axis=1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for trn_idx, val_idx in skf.split(df_feature, df_target):
            valid_data = df_data_copy.loc[val_idx, :]
            valid_data.reset_index(drop=True, inplace=True)
            split_dataset_list.append(valid_data)
        return split_dataset_list

    def get_index_split_data(self, df_data, index_ratio, random_state=1):
        # random_state = arch_epoch % 5
        df_data_c = copy.deepcopy(df_data)
        # random_state = randint(0, 1)
        # random_state = 1
        # split_ratio = [0.3, 0.4, 0.5, 0.6]
        split_ratio = [0.3, 0.4, 0.5, 0.6, 0.7]
        choice_ratio = split_ratio[index_ratio]

        # discrete_ca_num = \
        #     BaseUtility.get_discrete_ca_num(df_data, self.args.discrete_col)

        #
        data_split = DatasetSplit(self.args)
        get_reward_ins = GetReward(self.args)


        train_data, test_data = data_split.split_dataset_with_ratio(
            df_data_c, choice_ratio, random_state=random_state)


        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        # raw_search_data = self.get_raw_search_data()

        return {
            'train_data': train_data,
            'test_data': test_data,
        }

    def split_sample_most(self, df_data, random_state=1):

        df_data_copy = copy.deepcopy(df_data)
        from get_reward import GetReward
        get_reward_ins = GetReward(self.args)
        df_data_copy = self.data_process(df_data_copy)
        res_tuple = get_reward_ins.feature_pipline_train(
            NO_ACTION, df_data_copy)

        if res_tuple is None:
            return None
        new_train_fes, train_label, fe_params, operation_idx_dict = res_tuple
        clf = LocalOutlierFactor(contamination=0.1)
        y_pred = clf.fit_predict(new_train_fes)
        index_list = []
        for key, value in enumerate(y_pred):
            if value != -1:
                index_list.append(key)

        print('y_pred', y_pred)
        normal_data = df_data.loc[index_list, :]
        normal_data.reset_index(drop=True, inplace=True)
        target_cc = normal_data[self.args.target_col].value_counts()
        print('target_cc', target_cc)


        data_split = DatasetSplit(self.args)
        train_data, test_data = data_split.split_dataset_with_ratio(
            df_data, 0.5, random_state=random_state)


        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        return train_data, test_data

    def get_split_data_info(self, df_data):


        # df_data = self.data_process(df_data)
        discrete_ca_num = \
            BaseUtility.get_discrete_ca_num(df_data, self.args.discrete_col)
        zz = BaseUtility.get_filter_discrete_info(df_data, self.args.discrete_col)


        data_split = DatasetSplit(self.args)
        get_reward_ins = GetReward(self.args)

        split_eval_res = []
        for i in range(10):

            train_data, test_data = data_split.split_dataset_with_ratio(
                df_data, 0.3, random_state=i)


            train_idx = list(train_data.index)
            test_idx = list(test_data.index)


            train_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)


            model, valid_score = get_reward_ins.xgb_early_stop_best_model(
                train_data, test_data, NO_ACTION)

            split_eval_res.append({
                'random_state': i,
                'n_estimators': model.n_estimators,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'valid_score': valid_score
            })
            # print(i, model.n_estimators)

        n_estimators = [i['n_estimators'] for i in split_eval_res]
        normal_n_estimators = self.detect_outliers2(n_estimators)

        split_info = []
        for res in split_eval_res:
            if res['n_estimators'] in normal_n_estimators:
                split_info.append(res)
        split_info.sort(key=lambda x: x['n_estimators'], reverse=False)
        print('split_infodddd', len(split_info))
        split_info = split_info[-3:]
        # split_info = split_info
        # choose_random_state = [i['random_state'] for i in split_info[-3:]]

        split_dataset_info = []
        print('split_info', [i['n_estimators'] for i in split_info])
        for info in split_info:
            train_data, test_data = data_split.split_dataset_with_ratio(
                df_data, 0.3, random_state=info['random_state'])

            train_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)

            info['dataset'] = (train_data, test_data)
            info['discrete_ca_num'] = discrete_ca_num

            split_dataset_info.append(info)
        # print('split_dataset_info', split_dataset_info)

        return split_dataset_info

    def get_train_size(self, df_data):

        search_business = SearchBusiness(self.args)
        k_fold_list = search_business.get_k_fold_split_data(df_data)
        print('k_fold_list', len(k_fold_list))

    def sample_data_list(self):
        pass



if __name__ == '__main__':

    # load_args = 'adult_dataset'
    # load_args = 'titanic'
    # load_args = 'Customer_Segmentation'
    # load_args = 'bank_add'
    load_args = 'hzd_amend'
    from main import get_args
    args_ = get_args(load_args)
    search_ins = SearchBusiness(args_)
    dataset_split = DatasetSplit(args_)
    search_data, keep_data = dataset_split.split_search_keep_data()
    # search_ins.get_split_data_info(search_data)
    print(search_data.shape)
    search_ins.get_train_size(search_data)

    #
    # norm_data = search_ins.split_sample_most(search_data)
    # print(norm_data.shape)
    # search_ins.get_split_data_info(norm_data)








