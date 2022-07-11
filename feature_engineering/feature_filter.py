# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from collections import Counter
import logging


class FeatureFilterMath(object):
    def __init__(self, operation_idx_dict):
        self.operation_idx_dict = operation_idx_dict
        self._get_continuous_discrete_idx()
        self.delete_idx_dict = {}
        self.delete_idx_list = []

    def _get_continuous_discrete_idx(self):
        self.continuous_idx = []
        self.discrete_idx = []
        self.continuous_idx.extend(self.operation_idx_dict['ori_continuous_idx'])
        self.continuous_idx.extend(self.operation_idx_dict['convert_idx'])
        self.continuous_idx.extend(self.operation_idx_dict['arithmetic_idx'])
        self.discrete_idx.extend(self.operation_idx_dict['ori_discrete_idx'])
        self.discrete_idx.extend(self.operation_idx_dict['combine_idx'])

    def _cheak_array(self):
        pass

    def var_filter(self, array, threshold=0):
        var = np.nanvar(array, axis=0)
        delete_var_idx = [idx for idx in range(len(var)) if var[idx] <= threshold]
        self.delete_idx_dict['delete_var_idx'] = delete_var_idx

    def std_filter(self, array, threshold=0):
        std = np.nanstd(array, axis=0)
        delete_std_idx = [idx for idx in range(len(std)) if std[idx] <= threshold]
        self.delete_idx_dict['delete_std_idx'] = delete_std_idx

    def chi2_filter(self, data, label, p_threshold=0.05):
        '''
        Parameters
        ----------
        data : 2darray
        label : 1darray
            must discrete.
        p_threshold : float, 0.01/ 0.05
            The threshold of the chi-square test. The default is 0.05.
        Returns
        -------
        data_array : 2darray
        global_delete_idx : list
        只针对discrete columns
        '''
        # onehot columns and idx
        continue_idx = self.continuous_idx
        discrete_idx = self.discrete_idx
        discrete_fes = np.delete(data, continue_idx, axis=1)  #
        # continuous_fes = np.delete(data, discrete_idx, axis=1) #

        zerostd_idx = [(idx, global_idx) for idx, global_idx in enumerate(discrete_idx) if
                       np.std(discrete_fes[:, idx]) == 0]
        #
        local_zerostd_idx = [tp_idx[0] for tp_idx in zerostd_idx]
        global_zerostd_idx = [tp_idx[1] for tp_idx in zerostd_idx]

        discrete_fes_nozerostd = np.delete(discrete_fes, local_zerostd_idx, axis=1)  #
        if discrete_fes_nozerostd.shape[1] == 0:
            # return data,[]
            self.delete_idx_dict['delete_chi2_idx'] = []
        else:
            chivalue, pvalues_chi = chi2(discrete_fes_nozerostd, label)
            # k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
            # X_fschi = SelectKBest(chi2, k=k).fit_transform(x_train, y_train)
            remain_discrete_idx = [idx for idx in discrete_idx if idx not in global_zerostd_idx]
            delete_p_idx = [(idx, global_idx) for idx, global_idx in enumerate(remain_discrete_idx) if
                            pvalues_chi[idx] > p_threshold]
            # local_delete_p_idx = [tp_idx[0] for tp_idx in delete_p_idx]
            global_delete_p_idx = [tp_idx[1] for tp_idx in delete_p_idx]

            # discrete_fes_nozerostd_pfilter = np.delete(discrete_fes_nozerostd, local_delete_p_idx, axis=1) #
            # data_res = np.concatenate((continuous_fes, discrete_fes_nozerostd_pfilter), axis=1)

            #
            global_delete_idx = []
            global_delete_idx.extend(global_zerostd_idx)
            global_delete_idx.extend(global_delete_p_idx)
            self.delete_idx_dict['delete_chi2_idx'] = global_delete_idx
            # return data_res,global_delete_idx

    def mic_filter(self, data, label, task_type='classifier', mic_threshold=0):
        '''

        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('please check your data type, must np.ndarray')
        if not isinstance(label, np.ndarray):
            raise TypeError('please check your data type, must np.ndarray')
        label = label.reshape(-1)
        continue_idx = self.continuous_idx  #
        discrete_idx = self.discrete_idx  #

        if task_type == 'classifier':
            discrete_fes = np.delete(data, continue_idx, axis=1)  #
            continuous_fes = np.delete(data, discrete_idx, axis=1)  #

            #
            if discrete_fes.shape[1] != 0:
                discrete_mi = mutual_info_classif(discrete_fes, label, discrete_features=False, random_state = 2)
                delete_discrete_mi_idx = [(idx, global_idx) for idx, global_idx in enumerate(discrete_idx) if
                                          discrete_mi[idx] <= mic_threshold]

                global_delete_discrete_mi_idx = [tp_idx[1] for tp_idx in delete_discrete_mi_idx]
            else:
                global_delete_discrete_mi_idx = []

            #
            if continuous_fes.shape[1] != 0:
                continuous_mi = mutual_info_regression(continuous_fes, label , random_state = 2)
                delete_continuous_mi_idx = [(idx, global_idx) for idx, global_idx in enumerate(continue_idx) if
                                            continuous_mi[idx] <= mic_threshold]
                global_delete_continuous_mi_idx = [tp_idx[1] for tp_idx in delete_continuous_mi_idx]
            else:
                global_delete_continuous_mi_idx = []

            #
            global_delete_idx = []
            global_delete_idx.extend(global_delete_discrete_mi_idx)
            global_delete_idx.extend(global_delete_continuous_mi_idx)

        elif task_type == 'regression':
            mi = mutual_info_regression(data, label, random_state = 2)

            delete_mi_idx = [idx for idx in range(len(mi)) if mi[idx] <= mic_threshold]
            #
            global_delete_idx = delete_mi_idx
        else:
            logging.error(f'task_type:{task_type} not defined, must be "classifier" or "regression"')
            raise ValueError(f'task_type:{task_type} not defined, must be "classifier" or "regression"')
        self.delete_idx_dict['delete_mic_idx'] = global_delete_idx

    def _record_delete_dict(self):
        for cate in ['delete_var_idx', 'delete_chi2_idx', 'delete_mic_idx','delete_duplicates_idx','delete_nan_idx']:
            if cate not in self.delete_idx_dict.keys():
                self.delete_idx_dict[cate] = []

    def _record_delete_list(self):
        for v in self.delete_idx_dict.values():
            self.delete_idx_list.extend(v)
        self.delete_idx_list = list(set(self.delete_idx_list))

    def update_delete_res(self):
        self._record_delete_dict()
        self._record_delete_list()

    def columns_duplicates(self, array):
        _, idx = np.unique(array, axis=1, return_index=True)
        idx = np.sort(idx)
        delete_idx = []
        for i in range(array.shape[1]):
            if i not in idx:
                delete_idx.append(i)
        self.delete_idx_dict['delete_duplicates_idx'] = delete_idx

    def columns_na(self,array):
        columns = list(np.arange(array.shape[1]))
        df = pd.DataFrame(array, columns=columns)
        nan_idx = [col for col in list(df) if np.sum(df[col].isna())>0]
        self.delete_idx_dict['delete_nan_idx'] = nan_idx
