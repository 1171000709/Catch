# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from feature_engineering.utils import ff
from collections import Counter
from feature_engineering.feature_filter import FeatureFilterMath
from copy import deepcopy
from feature_engineering.decision_tree_bin import DecisionTreeBin

class FeatureEngTrain(object):
    def __init__(self):
        self.onehot_enc_dict = {}
        self.feature_eng_bins_dict = {}
        self.feature_eng_combine_dict = {}
        self.feature_normalization_dict = {}
        self.feature_categories2int_dict = {}
        self.feature_aggregation = {}
        self.feature_filter_dict = {}

    def normalization(self, col, col_op):
        '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
        #
        col = np.array(col).reshape(-1)
        mean = np.mean(col)
        std = np.std(col)
        self.feature_normalization_dict[str(col_op)] = (mean, std)
        if std != 0:
            return (col - mean) / std
        else:
            return col

    def max_min(self, col, col_op):
        '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
        #
        col = np.array(col).reshape(-1)
        max = np.max(col)
        min = np.min(col)
        self.feature_normalization_dict[str(col_op)] = (max, min)
        if (max - min) != 0:
            return (col - min) / (max - min)
        else:
            return col


    def onehot_encoder(self,ori_fe,col_name):
        '''
        '''
        ori_fe = np.array(ori_fe).reshape(-1,1)
        encoder = OneHotEncoder(handle_unknown ='ignore')#,sparse=False
        enc = encoder.fit(ori_fe)
        onehot_fe = enc.transform(ori_fe).toarray()
        self.onehot_enc_dict[col_name] = enc
        return onehot_fe

    def binning(self, ori_fe, bins, fe_name=None, method='frequency'):
        '''

        :type ori_fe: list or np.array
        :type bins: int
        :type fe_name: str,feature_eng_bins_dict中
        :type method: str
        :rtype:1. np.array,shape = (len(array),2)，
               2.fre_list,list of floats,
               3.new_fe_encode,np.mat,
        '''
        ori_fe = np.array(ori_fe)
        if method == 'frequency':
            fre_list = [np.percentile(ori_fe, 100 / bins * i) for i in range(1, bins)]
            fre_list = sorted(list(set(fre_list)))
            new_fe = np.array([ff(x, fre_list) for x in ori_fe])
            self.feature_eng_bins_dict[fe_name] = fre_list
            return new_fe.reshape(-1)
        #
        elif method == 'distance':
            umax = np.percentile(ori_fe, 99.99)
            umin = np.percentile(ori_fe, 0.01)
            step = (umax - umin) / bins
            fre_list = [umin + i * step for i in range(bins)]
            new_fe = np.array([ff(x, fre_list) for x in ori_fe])
            self.feature_eng_bins_dict[fe_name] = fre_list
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
        ffm = FeatureFilterMath(operation_idx_dict)
        ffm.var_filter(comb_onehot_res, threshold=0)

        if task_type == 'classifier':
            ffm.chi2_filter(comb_onehot_res, label, p_threshold=0.05)
        ffm.mic_filter(comb_onehot_res, label, task_type=task_type,mic_threshold=0)
        ffm.columns_duplicates(comb_onehot_res)

        ffm.update_delete_res()
        all_delete_idx = ffm.delete_idx_list
        filter_comb_onehot = np.delete(comb_onehot_res, all_delete_idx, axis=1)
        self.feature_filter_dict[fe_names] = all_delete_idx

        return filter_comb_onehot


    def combine_noonehot(self,ori_fes, fe_names):
        #
        fe_names = list(fe_names)
        cb_df = pd.DataFrame(ori_fes, columns=fe_names,dtype='int').astype(str)
        uniuqe_idx = cb_df.groupby(fe_names).count().reset_index()[fe_names]
        uniuqe_idx = uniuqe_idx.sort_values(by=fe_names, ascending=True).astype(str)
        uniuqe_idx['keys'] = uniuqe_idx[fe_names].apply(lambda x: ''.join(x),axis=1)
        uniuqe_idx['coding'] = range(1, len(uniuqe_idx) + 1)
        col_unique_dict = dict(zip(uniuqe_idx['keys'],uniuqe_idx['coding']))
        cb_df = pd.merge(cb_df, uniuqe_idx[fe_names + ['coding']], on= fe_names, how='left')
        combine_col = cb_df['coding'].values.astype(int)
        self.feature_eng_combine_dict[tuple(fe_names)] = col_unique_dict
        return combine_col.reshape(-1, 1)

    def check_is_continuous(self,ori_fes,fe_names,continuous_columns,continuous_bins):
        for idx,name in enumerate(fe_names):
            if name in continuous_columns:
                bins = continuous_bins[name]
                if len(np.unique(ori_fes[:,idx])) > bins:
                    fes_bins ,_ = self.binning(ori_fes[:,idx],bins,fe_name = name,method = 'frequency')
                    ori_fes[:,idx] = fes_bins.reshape(len(fes_bins))
        return ori_fes


    def recursion_freq_bins(self,count_dict,bins,total_rate,res):
        # print(f'count_dict: {count_dict},bins: {bins}, total_rate: {total_rate}, res: {res}')
        if bins > 0 and len(count_dict)> 0:
            rate = total_rate / bins
            if count_dict[0][-1] >= rate:
                res.append(count_dict[0])
                bins = bins-1
                total_rate = total_rate - count_dict[0][-1]
                count_dict.remove(count_dict[0])
                res = self.recursion_freq_bins(count_dict,bins,total_rate,res)

            else:
                if len(count_dict) > 1:
                    count_dict[0] = (*count_dict[0][:-1] + count_dict[1][:-1],count_dict[0][-1] + count_dict[1][-1])
                    count_dict.remove(count_dict[1])
                    res = self.recursion_freq_bins(count_dict,bins,total_rate,res)
                else:
                    res.append(count_dict[0])
        return res



    def discrete_freq_bins(self,ori_fe,bins,fe_name):
        ori_fe = ori_fe.reshape(-1).astype(int)
        count_dict = dict(Counter(ori_fe))
        t_num = len(ori_fe)
        count_dict = {k:v/t_num  for k,v in count_dict.items()}
        count_dict = sorted(count_dict.items(),key = lambda x: x[1], reverse=True)

        total_rate = 1
        res = []
        res = self.recursion_freq_bins(count_dict,bins,total_rate,res)
        merge_dict = {idx: list(tp[:-1]) for idx,tp in enumerate(res)} # 需要存储的字典，{0:[1,2,3], 1:[4,5],...}
        map_dict = {v:k for k,v in merge_dict.items() for v in v}

        bins_fe = pd.Series(ori_fe).map(map_dict).values
        self.feature_eng_bins_dict[fe_name] = merge_dict
        return bins_fe.reshape(-1)

    def decisiontree_bins_df(self,X_df,y_arr):
        decision_tree_bin = DecisionTreeBin()
        res_X_df = deepcopy(X_df)
        for col in res_X_df.columns:
            boundary = decision_tree_bin.optimal_binning_boundary(res_X_df[col].values, y_arr)
            label = [i for i in range(len(boundary)-1)]#[i for i in range(decision_tree_bin.max_leaf_nodes)]
            binned_feature = pd.cut(x=res_X_df[col].values, bins=boundary, right=False, labels=label)
            res_X_df[col] = binned_feature
            self.feature_eng_bins_dict[col] = [boundary, label]
        return res_X_df

    def aggregation(self, ori_fe, fe_names, agg_action_list):
        '''
        ori_fe: 2d-array, array(discrete_col, continuous_col)
        fe_names: tuple, (discrete_col, continuous_col)
        agg_action_list: list, action list
        df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
        '''
        df = pd.DataFrame(ori_fe, columns=list(fe_names))
        df_group = df.groupby([fe_names[0]]).aggregate({fe_names[1] : agg_action_list}).reset_index()

        merge_df = df.merge(df_group, on=fe_names[0], how='left')
        self.feature_aggregation[(fe_names, agg_action_list)] = df_group
        res = merge_df.drop(columns= fe_names[0]).values
        return res

    def categories_to_int(self,col,name):
        '''
        :type col: list or np.array
        :rtype: np.array,shape = (len(array),1)
        '''
        col = np.array(col).reshape(-1)
        unique_count = dict(Counter(col))
        unique_count = dict(sorted(unique_count.items(), key= lambda x: x[1] , reverse= True))
        categories_map = dict(zip(unique_count.keys(), range(len(unique_count))))
        new_fe = pd.Series(col).map(categories_map).astype(int).values
        self.feature_categories2int_dict[name] = categories_map
        return new_fe.reshape(-1)

    def get_train_params(self):

        return self.onehot_enc_dict, \
               self.feature_eng_bins_dict, \
               self.feature_eng_combine_dict , \
               self.feature_normalization_dict , \
               self.feature_categories2int_dict , \
               self.feature_aggregation , \
               self.feature_filter_dict

    def clear_train_params(self):
        self.onehot_enc_dict = {}
        self.feature_eng_bins_dict = {}
        self.feature_eng_combine_dict = {}
        self.feature_normalization_dict = {}
        self.feature_categories2int_dict = {}
        self.feature_aggregation = {}
        self.feature_filter_dict = {}

