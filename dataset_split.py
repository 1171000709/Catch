# -*- coding: utf-8 -*-
from pathlib import Path
import time
import logging
import os
import copy
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from feature_engineering.model_base import ModelBase
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import Counter

from xgboost import XGBClassifier

from reduce_mem import reduce_mem_usage


class DatasetSplit(object):
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.target_col = args.target_col
        self.task_type = args.task_type
        #self.search_sample_num = args.search_sample_num
        self.args = args

        self.dataset_name = Path(self.dataset_path).stem
        self.split_data_path = Path('data/sample') / self.dataset_name
        self.search_data_csv_name = 'search_data.csv'
        self.keep_data_csv_name = 'keep_data.csv'

    #
    def split_dataset_with_ratio(self, df, test_rate, random_state=1):
        df_target = df[self.target_col]
        df_feature = df.drop(self.target_col, axis=1)
        #
        if self.task_type == 'classifier':
            x_train, x_test, y_train, y_test = train_test_split(
                df_feature, df_target, test_size=test_rate,
                shuffle=True, stratify=df_target, random_state=random_state)
        elif self.task_type == 'regression':
            x_train, x_test, y_train, y_test = train_test_split(
                df_feature, df_target, test_size=test_rate,
                shuffle=True, random_state=random_state)
        else:
            logging.info('task_type error')
            x_test = x_train = y_train = y_test = None
        # feature target merge
        train_data = x_train.merge(pd.DataFrame(y_train), left_index=True,
                                   right_index=True, how='left')
        test_data = x_test.merge(pd.DataFrame(y_test), left_index=True,
                                 right_index=True, how='left')
        return train_data, test_data

    @staticmethod
    def fe_simple_pipline(df_fe):
        for header in df_fe.columns.values:
            if df_fe[header].dtype == "object":
                oe = preprocessing.OrdinalEncoder()
                trans_column = oe.fit_transform(df_fe[header].values.reshape(-1, 1))
                df_fe[header] = trans_column.reshape(1, -1)[0]
        return df_fe

    def split_search_keep_data(self, random_state=2, merge_bool=False):
        #raw_df = reduce_mem_usage(pd.read_csv(self.dataset_path), logging)
        raw_df = pd.read_csv(self.dataset_path)
        raw_df = raw_df[self.args.continuous_col+self.args.discrete_col+
                        [self.args.target_col]]

        # limit the row
        raw_df = raw_df.sample(min(raw_df.shape[0], 150000), axis=0, replace=False)
        raw_df.reset_index(drop = True, inplace=True)


        #
        search_data, keep_data = self.split_dataset_with_ratio(
            raw_df, 0.2, random_state=random_state)

        #
        search_data.drop_duplicates(keep='first', inplace=True)
        search_data.reset_index(drop=True, inplace=True)

        if merge_bool:
            #
            # ===============================================================
            from search_business import SearchBusiness
            search_business = SearchBusiness(self.args)
            search_data = search_business.data_process(search_data)
            # ===============================================================

            # ===============================================================

            #
            # ===============================================================
            for col in self.args.continuous_col:
                percent_1, percent_99 = \
                    np.percentile(search_data[col], (1, 99))
                search_data[col] = np.where(
                    search_data[col] < percent_1, percent_1, search_data[col])
                search_data[col] = np.where(
                    search_data[col] > percent_99, percent_99, search_data[col])
            # ===============================================================

        search_data.reset_index(drop=True, inplace=True)
        keep_data.reset_index(drop=True, inplace=True)

        #
        self.single_save_to_csv(keep_data, self.keep_data_csv_name)
        self.single_save_to_csv(search_data, self.search_data_csv_name)
        return search_data, keep_data

    @staticmethod
    def fe_simple_pipline(df_fe):
        for header in df_fe.columns.values:
            if df_fe[header].dtype == "object":
                oe = preprocessing.OrdinalEncoder()
                trans_column = oe.fit_transform(df_fe[header].values.reshape(-1, 1))
                df_fe[header] = trans_column.reshape(1, -1)[0]
        return df_fe

    def single_save_to_csv(self, df_data, save_file_name):
        df_data.reset_index(drop=True, inplace=True)
        # self.split_data_path = Path('data') / self.dataset_name
        if not os.path.exists(self.split_data_path):
            os.makedirs(self.split_data_path)
        df_data.to_csv(self.split_data_path / save_file_name, index=False)
        # df_data.to_csv(self.split_data_path / save_file_name)

    def save_to_csv(self, train_data, test_data):

        save_path = '{}_{}'.format(
            self.dataset_name, time.strftime("%Y%m%d-%H%M%S"))

        sampled_path = Path('data/sample') / save_path

        if not os.path.exists(sampled_path):
            os.makedirs(sampled_path)

        train_data.to_csv(sampled_path / 'train_data.csv', index=False)
        test_data.to_csv(sampled_path / 'test_data.csv', index=False)
        return sampled_path

    def load_all_data(self):
        #all_data = reduce_mem_usage(pd.read_csv(self.dataset_path), logging)
        all_data = pd.read_csv(self.dataset_path)
        all_data = all_data[self.args.continuous_col+self.args.discrete_col +
                            [self.args.target_col]]
        #all_data = all_data[0:1000]
        print('all_data shape', all_data.shape)
        #
        all_data.drop_duplicates(keep='first', inplace=True)
        all_data.reset_index(drop=True, inplace=True)

        return all_data

    def split_load_search_data(self):
        #all_data = reduce_mem_usage(pd.read_csv(self.dataset_path), logging)
        all_data = pd.read_csv(self.dataset_path)
        all_data = all_data[self.args.continuous_col+self.args.discrete_col +
                            [self.args.target_col]]
        print('all_data shape', all_data.shape)
        #
        search_data, keep_data = self.split_dataset_with_ratio(all_data, 0.2, random_state=1)

        #
        search_data.drop_duplicates(keep='first', inplace=True)
        search_data.reset_index(drop=True, inplace=True)


        search_data.reset_index(drop=True, inplace=True)
        keep_data.reset_index(drop=True, inplace=True)

        #
        self.single_save_to_csv(keep_data, self.keep_data_csv_name)
        self.single_save_to_csv(search_data, self.search_data_csv_name)
        return search_data

if __name__ == '__main__':
    pass





















