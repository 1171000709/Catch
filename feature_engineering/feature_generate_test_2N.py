# -*- coding: utf-8 -*-
import numpy as np
from feature_engineering.utils import ff
import pandas as pd
from copy import deepcopy
from feature_engineering.decision_tree_bin import DecisionTreeBin

class FeatureEngTest(object):
    def __init__(self,onehot_enc_dict, \
                 feature_eng_bins_dict, \
                 feature_eng_combine_dict, \
                 feature_normalization_dict, \
                 feature_categories2int_dict, \
                 feature_aggregation, \
                 feature_filter_dict):

        self.onehot_enc_dict = onehot_enc_dict
        self.feature_eng_bins_dict = feature_eng_bins_dict
        self.feature_eng_combine_dict = feature_eng_combine_dict
        self.feature_normalization_dict = feature_normalization_dict
        self.feature_categories2int_dict = feature_categories2int_dict
        self.feature_aggregation = feature_aggregation
        self.feature_filter_dict = feature_filter_dict

    def normalization(self,col,col_op):
        '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
        #
        col = np.array(col).reshape(-1)
        mean = self.feature_normalization_dict[str(col_op)][0]
        std = self.feature_normalization_dict[str(col_op)][1]
        if std != 0:
            return (col - mean) / std
        else:
            return col

    def max_min(self,col,col_op):
        '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
        #
        col = np.array(col).reshape(-1)
        max = self.feature_normalization_dict[str(col_op)][0]
        min = self.feature_normalization_dict[str(col_op)][1]
        if (max - min)!=0:
            return (col - min) / (max - min)
        else:
            return col

    def onehot_encoder(self,ori_fe,col_name):
        '''
        #
        :type ori_fe: list or np.array,
        '''
        ori_fe = np.array(ori_fe).reshape(-1,1)
        enc = self.onehot_enc_dict[col_name]
        onehot_fe = enc.transform(ori_fe).toarray()
        return onehot_fe

    def binning(self, ori_fe, bins, fe_name=None, method='frequency'):
        '''

        '''
        ori_fe = np.array(ori_fe)
        if fe_name not in self.feature_eng_bins_dict.keys():
            return ori_fe
        if method == 'frequency':
            fre_list = self.feature_eng_bins_dict[fe_name]
            new_fe = np.array([ff(x, fre_list) for x in ori_fe])
            return new_fe.reshape(-1)
        #
        elif method == 'distance':
            fre_list = self.feature_eng_bins_dict[fe_name]
            new_fe = np.array([ff(x, fre_list) for x in ori_fe])
            return new_fe.reshape(-1)

    def combine_onehot(self,ori_fes,fe_names,label,task_type):
        #
        combine_col = self.combine_noonehot(ori_fes, fe_names).reshape(-1,1)
        comb_onehot_res = self.onehot_encoder(combine_col, col_name = 'combs_' + str(fe_names))

        operation_idx_dict = {}
        operation_idx_dict['ori_continuous_idx'] = []
        operation_idx_dict['ori_discrete_idx'] = []
        operation_idx_dict['convert_idx'] = []
        operation_idx_dict['arithmetic_idx'] = []
        operation_idx_dict['combine_idx'] = list(range(comb_onehot_res.shape[1]))

        all_delete_idx = self.feature_filter_dict[fe_names]
        filter_comb_onehot = np.delete(comb_onehot_res, all_delete_idx, axis=1)
        return filter_comb_onehot


    def combine_noonehot(self,ori_fes, fe_names):
        #
        col_unique_dict = self.feature_eng_combine_dict[fe_names]
        fe_names = list(fe_names)
        cb_df = pd.DataFrame(ori_fes, columns=fe_names,dtype='int').astype(str)
        cb_df['keys'] = cb_df[fe_names].apply(lambda x: ''.join(x),axis=1)
        cb_df['coding'] = cb_df['keys'].map(col_unique_dict)
        cb_df['coding'] = cb_df['coding'].fillna(0).astype(int) # 将训练集中没有出现的组合，全部归并为新一类
        combine_col = cb_df['coding'].values
        return combine_col.reshape(-1, 1)


    def check_is_continuous(self,ori_fes,fe_names,continuous_columns,continuous_bins):
        for idx,name in enumerate(fe_names):
            if name in continuous_columns:
                bins = continuous_bins[name]
                if len(np.unique(ori_fes[:,idx])) > bins:
                    # raise ValueError(f'{name} unique value is {len(np.unique(ori_fes[:,idx]))} , but bins {bins}')
                    fes_bins ,_ = self.binning(ori_fes[:,idx],bins,fe_name = name,method = 'frequency')
                    ori_fes[:,idx] = fes_bins.reshape(len(fes_bins))
        return ori_fes

    def discrete_freq_bins(self, ori_fe, bins, fe_name):
        if fe_name in self.feature_eng_bins_dict.keys():
            ori_fe = ori_fe.reshape(-1).astype(int)
            merge_dict = self.feature_eng_bins_dict[fe_name]
            map_dict = {v: k for k, v in merge_dict.items() for v in v}
            bins_fe = pd.Series(ori_fe).map(map_dict).fillna(len(merge_dict)-1).astype(int).values # 如果有没有见过的类别，合并到最后一箱中
        else:
            bins_fe = ori_fe
        return bins_fe.reshape(-1)

    def decisiontree_bins_df(self, X_df, y_arr):
        res_X_df = deepcopy(X_df)
        for col in res_X_df.columns:
            boundary, label = self.feature_eng_bins_dict[col]
            binned_feature = pd.cut(x=res_X_df[col].values, bins=boundary, right=False, labels=label)
            res_X_df[col] = binned_feature
        return res_X_df

    def aggregation(self, ori_fe, fe_names, agg_action_list):
        '''
        ori_fe: 2d-array, array(discrete_col, continuous_col)
        fe_names: tuple, (discrete_col, continuous_col)
        agg_action_list: list, action list
        df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
        '''
        df = pd.DataFrame(ori_fe, columns=list(fe_names))
        df_group = self.feature_aggregation[(fe_names, agg_action_list)]

        merge_df = df.merge(df_group, on=fe_names[0], how='left')
        res = merge_df.drop(columns= fe_names[0]).values
        return res

    def categories_to_int(self,col,name):
        '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
        col = np.array(col).reshape(-1)
        categories_map = self.feature_categories2int_dict[name]
        # case1 :
        new_fe = pd.Series(col).map(categories_map).fillna(len(categories_map)-1).astype(int)
        # case2 :
        # new_fe = pd.Series(col).map(categories_map).fillna(len(categories_map)).astype(int)
        return new_fe.values.reshape(-1)

    def clear_train_params(self):
        self.onehot_enc_dict = {}
        self.feature_eng_bins_dict = {}
        self.feature_eng_combine_dict = {}
        self.feature_normalization_dict = {}
        self.feature_categories2int_dict = {}
        self.feature_aggregation = {}
        self.feature_filter_dict = {}


