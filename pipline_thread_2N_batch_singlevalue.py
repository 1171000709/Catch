# -*- coding: utf-8 -*-
import copy
import numpy as np
import pandas as pd
from feature_engineering.feature_generate_general_2N import log,sqrt,inverse,add,subtract,multiply,divide,square#,binning_for_discrete
from threading import Thread
from feature_engineering.feature_generate_test_2N import FeatureEngTest
from feature_engineering.feature_generate_train_2N import FeatureEngTrain
from feature_engineering.feature_generate_general_2N import power3,sigmoid,tanh,abs,max,min,round, none
import warnings
warnings.filterwarnings("ignore")

class MyThread(Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args) #
        #
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            return None

class Pipline(object):
    def __init__(self,continuous_columns, discrete_columns, do_onehot=False):
        self.continuous_columns = continuous_columns
        self.discrete_columns = discrete_columns
        self.do_onehot = do_onehot

    def single_discrete_fe_2_int_type(self,discrete_col_name, ori_col):
        int_type_col = self.fes_eng.categories_to_int(ori_col,discrete_col_name)
        return int_type_col

    def single_maxmin(self,ori_name,ori_col):
        scaled_fe = self.fes_eng.max_min(ori_col,ori_name)
        return scaled_fe

    def single_normalization(self,ori_name,ori_col):
        scaled_fe = self.fes_eng.normalization(ori_col,ori_name)
        return scaled_fe

    def single_math_convert(self,ops_tuple,ori_col):
        ori_name, operation = ops_tuple
        if operation == 'log':
            new_fe = log(ori_col)
        elif operation == 'sqrt':
            new_fe = sqrt(ori_col)
        elif operation == 'inverse':
            new_fe = inverse(ori_col)
        elif operation == 'square':
            new_fe = square(ori_col)
        elif operation == 'power3':
            new_fe = power3(ori_col)
        elif operation == 'sigmoid':
            new_fe = sigmoid(ori_col)
        elif operation == 'tanh':
            new_fe = tanh(ori_col)
        elif operation == 'abs':
            new_fe = abs(ori_col)
        elif operation == 'max':
            new_fe = max(ori_col)
        elif operation == 'min':
            new_fe = min(ori_col)
        elif operation == 'round':
            new_fe = round(ori_col)
        elif operation is None:
            new_fe = np.array(ori_col)
        else:
            raise ValueError(f'operation : {operation} is not defined')

        if len(new_fe) > 0:
            # col_op = '_'.join([ori_name , operation])
            # fe_scaled = self.fes_eng.max_min(new_fe,col_op).reshape(-1,1)
            # fe_scaled = self.fes_eng.normalization(new_fe,col_op).reshape(-1)
            return new_fe.reshape(-1,1)
        else:
            raise ValueError(f'ori_name: {ori_name} , operation : {operation} return None or len=0')

    def single_arithmetic(self,col1, col2, colidx_ops_tuple):
        # colidx_ops_tuple: tuple
        operation = colidx_ops_tuple[-1]
        if operation == 'add':
            new_fe = add(col1, col2)
        elif operation == 'subtract':
            new_fe = subtract(col1, col2)
        elif operation == 'multiply':
            new_fe = multiply(col1, col2)
        elif operation == 'divide':
            new_fe = divide(col1, col2)
        else:
            raise ValueError(f'no operation: {operation}')
        # col_op = '_'.join([str(s) for s in colidx_ops_tuple])
        # fe_scaled = self.fes_eng.max_min(new_fe[:,i],col_op).reshape(len(new_fe),1)
        # fe_scaled = self.fes_eng.normalization(new_fe,col_op).reshape(-1)
        return new_fe.reshape(-1,1)

    def arithmetic_all_tuple(self,mathops_dict):
        math_list = list(mathops_dict.values())[0]
        math_ops = list(mathops_dict.keys())[0]
        all_tuple_operations = []
        for col1_idx,col2_idx in enumerate(math_list):
            all_tuple_operations.append([(col1_idx,col2_idx),math_ops])
        return all_tuple_operations

    def single_bins(self,ori_column,bins,ori_fe):
        if self.train == True:
            if len(np.unique(ori_fe)) > bins:
                fes_bins = self.fes_eng.binning(ori_fe,bins,fe_name = ori_column,method = 'frequency')
            else:
                fes_bins = copy.deepcopy(ori_fe)
        else:
            fes_bins = self.fes_eng.binning(ori_fe, bins, fe_name=ori_column, method='frequency')
        return fes_bins.reshape(-1)


    def single_bins_discrete(self,ori_column,bins,ori_fe):
        if self.train == True:
            if len(ori_fe.unique()) > bins:
                ori_fe_bins = self.fes_eng.discrete_freq_bins(ori_fe, bins,fe_name = ori_column)
            else:
                ori_fe_bins = copy.deepcopy(ori_fe)
                ori_fe_bins = ori_fe_bins.reshape(-1)
        else:
            ori_fe_bins = self.fes_eng.discrete_freq_bins(ori_fe, bins, fe_name=ori_column)
        return ori_fe_bins


    def single_combine_tuple(self, combine_tuple,ori_cols):
        # ori_cols:
        # if self.do_onehot:
        #     combine_array = self.fes_eng.combine_onehot(ori_cols,combine_tuple,self.label, self.task_type) # combine,label,task_type
        # else:
        #     combine_array = self.fes_eng.combine_noonehot(ori_cols,combine_tuple) # combine
        combine_array = self.fes_eng.combine_noonehot(ori_cols, combine_tuple)
        return combine_array

    def combine_all_tuple(self,combsops_dict):
        combs_list = list(combsops_dict.values())[0]
        all_tuple_operations = []
        for col1_idx, col2_idx in enumerate(combs_list):
            all_tuple_operations.append((col1_idx, col2_idx))
        return all_tuple_operations

    def single_onehot_encode(self,col_name, ori_fe):
        '''
        '''
        onehot_fe = self.fes_eng.onehot_encoder(ori_fe, col_name)
        return onehot_fe

    @staticmethod
    def multi_thread(fun, args_list):
        result = []
        threads = []
        for arg in args_list:
            if isinstance(arg, str):
                t = MyThread(fun, args=(arg,))
            else:
                t = MyThread(fun, args=(*arg,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
            result.append(t.get_result())
        return result

    def noaction_baseline(self,ori_dataframe):
        con_df = ori_dataframe[self.continuous_columns]
        dis_df = ori_dataframe[self.discrete_columns]
        # discrete2int
        discrete2int_param_list = [(col_name,dis_df[col_name].values) for col_name in self.discrete_columns]
        int_col_res = self.multi_thread(self.single_discrete_fe_2_int_type, discrete2int_param_list)
        for index, array_value in enumerate(self.discrete_columns):
            dis_df[array_value] = int_col_res[index]

        dis_df[self.discrete_columns] = pd.DataFrame(int_col_res).T

        #
        # zscore_dict = {k: 'z_score' for k in self.continuous_columns}
        # zscore_param_list = [(col_name, con_df[col_name].values) for col_name in zscore_dict.keys()]
        # if len(zscore_dict) > 0:
        #     zscore_res = self.multi_thread(self.single_normalization, zscore_param_list)  # normalization
        #
        #     for index, array_value in enumerate(list(zscore_dict.keys())):
        #         con_df[array_value] = zscore_res[index]
        #
        #     con_df[list(zscore_dict.keys())] = pd.DataFrame(zscore_res).T

        if self.do_onehot:
            if len(self.discrete_columns) > 0:
                onehot_arr, onehot_col_list = self.onehot_process(dis_df,self.discrete_columns)
                ori_fes = np.concatenate((con_df.values,onehot_arr),axis=1)
                res_colname_list = self.continuous_columns + onehot_col_list  # 特征名称
                return ori_fes, res_colname_list
            else:
                return con_df.values,self.continuous_columns
        else:
            res_df = pd.concat([con_df,dis_df], axis=1)
            res_colname_list = self.continuous_columns+self.discrete_columns
            return res_df.values,res_colname_list

    def parser_single_opsdict(self,ops_list,frame,present_col_dict):
        col_idx0 = self.continuous_columns.index(ops_list[0])
        col_idx1 = self.continuous_columns.index(ops_list[1])
        ops = ops_list[2]
        ori_ops = ops_list[3]
        if isinstance(ops, str):
            #  (col1, col2, (colname,colname,ops))
            mathops_param = (frame.iloc[:, col_idx0], frame.iloc[:, col_idx1], (ops_list[0], ops_list[1], ops))
            arithmetic_res = self.single_arithmetic(*mathops_param)
            if ori_ops in ['concat', 'concat_END']:
                colname = len(list(frame))
            elif ori_ops in ['replace', 'replace_END']:
                colname = col_idx0
            else:
                raise ValueError(f'ori_ops : {ori_ops} not define')
            frame[colname] = arithmetic_res
            if colname in present_col_dict.keys():
                present_col_dict[colname].append(((ops_list[0], ops_list[1]),ops))
            else:
                present_col_dict[colname] = [((ops_list[0], ops_list[1]),ops)]
            return frame, present_col_dict

        elif isinstance(ops, tuple):
            #  [((colname,ops),array),()]
            valueops_param_list = [((ops_list[0],ops[0]), frame.iloc[:, col_idx0]), ((ops_list[1],ops[1]), frame.iloc[:, col_idx1])]
            if len(valueops_param_list) > 0:
                value_convert_res = self.multi_thread(self.single_math_convert, valueops_param_list)
                if ori_ops in ['concat', 'concat_END']:
                    colname = [len(list(frame)), len(list(frame))+1]
                elif ori_ops in ['replace', 'replace_END']:
                    colname = [col_idx0, col_idx1]
                else:
                    raise ValueError(f'ori_ops : {ori_ops} not define')
                for i,arr in enumerate(value_convert_res):
                    idx = colname[i]
                    frame[idx] = arr
                    if idx in present_col_dict.keys():
                        present_col_dict[idx].append((ops_list[i],ops[i]))
                    else:
                        present_col_dict[idx] = (ops_list[i], ops[i])
            return frame, present_col_dict
        else:
            raise ValueError(f'ops : {ops} must be str or tuple')

    def parser_single_opslist(self,ops_list,frame):
        ops = ops_list[-2]
        ori_ops = ops_list[-1]
        if len(ops_list) == 3:
            col_idx0 = self.continuous_columns.index(ops_list[0])
            # 值转换 [((colname,ops),array),()]
            valueops_param = ((ops_list[0], ops), frame.iloc[:, col_idx0])
            arr = self.single_math_convert(*valueops_param)
            if ori_ops in ['concat', 'concat_END']:
                colname = len(list(frame))
            elif ori_ops in ['replace', 'replace_END']:
                colname = col_idx0
            else:
                raise ValueError(f'ori_ops : {ori_ops} not define')
            return arr

        elif len(ops_list) == 4:
            col_idx0 = self.continuous_columns.index(ops_list[0])
            col_idx1 = self.continuous_columns.index(ops_list[1])
            #  (col1, col2, (colname,colname,ops))
            mathops_param = (frame.iloc[:, col_idx0], frame.iloc[:, col_idx1], (ops_list[0], ops_list[1], ops))
            arr = self.single_arithmetic(*mathops_param)
            if ori_ops in ['concat', 'concat_END']:
                colname = len(list(frame))
            elif ori_ops in ['replace', 'replace_END']:
                colname = col_idx0
            else:
                raise ValueError(f'ori_ops : {ori_ops} not define')
            return arr
        else:
            raise ValueError(f'ops : {ops} must be str or tuple')

    def parser_single_opslist_combine(self, ops_list, frame):
        ops = ops_list[-2]
        ori_ops = ops_list[-1]
        if len(ops_list) == 4:
            if ops == 'combine':
                col_idx0 = self.all_discrete_col.index(ops_list[0])
                col_idx1 = self.all_discrete_col.index(ops_list[1])
                # combine (col1, col2, (colname,colname,ops))
                combineops_param = ((ops_list[0], ops_list[1]), frame.iloc[:, [col_idx0, col_idx1]].values)
                arr = self.single_combine_tuple(*combineops_param)

                if ori_ops in ['concat', 'concat_END']:
                    colname = len(list(frame))
                elif ori_ops in ['replace', 'replace_END']:
                    colname = col_idx0
                else:
                    raise ValueError(f'ori_ops : {ori_ops} not define')
                return arr

            else:
                raise ValueError(f'ops : {ops} not define')

        else:
            raise ValueError(f'combine opslist length: {ops_list} must be 4')

    def onehot_process(self,dis_df, columns_list):
        if self.do_onehot:
            onehot_param_list = [(col_name, dis_df[col_name].values) for col_name in columns_list]
            onehot_res = self.multi_thread(self.single_onehot_encode, onehot_param_list)
            onehot_col_list = []
            for col, arr in zip(columns_list, onehot_res):
                col_list = [f'col_{str(idx)}' for idx in range(arr.shape[1])]
                onehot_col_list.append(col_list)
            onehot_res = np.concatenate((onehot_res[0], *onehot_res[1:]), axis=1)
            return onehot_res, onehot_col_list
        else:
            return dis_df.values, columns_list

    @staticmethod
    def get_param_list(cycle_ops_list,continuous_df,discrete_df):
        replace_param_list_combine = []
        concat_param_list_combine = []
        replace_param_list_continuous = []
        concat_param_list_continuous = []
        for ops_list in cycle_ops_list:
            if 'replace' in ops_list[-1]:
                if 'combine' in ops_list[-2]:
                    replace_param_list_combine.append((ops_list, discrete_df))
                else:
                    replace_param_list_continuous.append((ops_list, continuous_df))
            elif 'concat' in ops_list[-1]:
                if 'combine' in ops_list[-2]:
                    concat_param_list_combine.append((ops_list, discrete_df))
                else:
                    concat_param_list_continuous.append((ops_list, continuous_df))
            else:
                raise ValueError(f'ops_list[-1] must contain replace or concat, not {ops_list[-1]}')
        return replace_param_list_continuous,concat_param_list_continuous, replace_param_list_combine,concat_param_list_combine

    def process_cycle_ops_list(self,cycle_ops_list, continuous_df, discrete_df):
        replace_param_list, concat_param_list, replace_param_list_combine, concat_param_list_combine = self.get_param_list(cycle_ops_list, continuous_df, discrete_df)
        #
        if len(replace_param_list) > 0:
            replace_idx = []
            for ops, _ in replace_param_list:
                replace_idx.append(self.continuous_columns.index(ops[0]))
            replace_arr_list = self.multi_thread(self.parser_single_opslist, replace_param_list)
            if len(replace_arr_list)>1:
                replace_arr = np.concatenate((replace_arr_list[0], *replace_arr_list[1:]), axis=1)
            else:
                replace_arr = replace_arr_list[0]
            replace_arr_df = pd.DataFrame(replace_arr, columns=replace_idx)

        if len(concat_param_list) > 0:
            concat_arr_list = self.multi_thread(self.parser_single_opslist, concat_param_list)
            if len(concat_arr_list) > 1:
                concat_arr = np.concatenate((concat_arr_list[0], *concat_arr_list[1:]), axis=1)
            else:
                concat_arr = concat_arr_list[0]
            concat_arr_df = pd.DataFrame(concat_arr)
            concat_arr_df.columns = list(range(len(list(continuous_df)), len(list(continuous_df)) + len(list(concat_arr_df))))
            continuous_df = pd.concat([continuous_df, concat_arr_df], axis=1)

        if len(replace_param_list) > 0:
            continuous_df[replace_idx] = replace_arr_df

        #
        if len(replace_param_list_combine) > 0:
            replace_idx = []
            for ops, _ in replace_param_list_combine:
                replace_idx.append(self.all_discrete_col.index(ops[0]))
            replace_arr_list = self.multi_thread(self.parser_single_opslist_combine, replace_param_list_combine)
            if len(replace_arr_list)>1:
                replace_arr = np.concatenate((replace_arr_list[0], *replace_arr_list[1:]), axis=1)
            else:
                replace_arr = replace_arr_list[0]
            replace_arr_df = pd.DataFrame(replace_arr, columns=replace_idx)

        if len(concat_param_list_combine) > 0:
            concat_arr_list = self.multi_thread(self.parser_single_opslist_combine, concat_param_list_combine)
            if len(concat_arr_list) >1:
                concat_arr = np.concatenate((concat_arr_list[0], *concat_arr_list[1:]), axis=1)
            else:
                concat_arr = concat_arr_list[0]
            concat_arr_df = pd.DataFrame(concat_arr)
            concat_arr_df.columns = list(range(len(list(discrete_df)), len(list(discrete_df)) + len(list(concat_arr_df))))
            discrete_df = pd.concat([discrete_df, concat_arr_df], axis=1)

        if len(replace_param_list_combine) > 0:
            discrete_df[replace_idx] = replace_arr_df

        return continuous_df,discrete_df


    def multi_run_thread(self, ori_dataframe):
        actions_num = len(self.actions)
        if actions_num == 0:
            ori_fes,res_colname_list = self.noaction_baseline(ori_dataframe)
            return ori_fes

        con_df = ori_dataframe[self.continuous_columns]
        dis_df = ori_dataframe[self.discrete_columns]
        present_col_dict = {idx: [] for idx, col in enumerate(self.continuous_columns)}


        con_ori_df = copy.deepcopy(con_df)
        con_ori_df.columns = list(present_col_dict.keys())

        dis_ori_df_ = copy.deepcopy(dis_df)
        #discrete2int
        discrete2int_param_list = [(col_name, dis_ori_df_[col_name].values) for col_name in self.discrete_columns]
        int_col_res = self.multi_thread(self.single_discrete_fe_2_int_type, discrete2int_param_list)
        for index, array_value in enumerate(self.discrete_columns):
            dis_ori_df_[array_value] = int_col_res[index]

        if self.task_type == 'classifier':
            con_ori_df_bins = self.fes_eng.decisiontree_bins_df(con_df, self.label)
            dis_ori_df = pd.concat([con_ori_df_bins,dis_ori_df_],axis=1)
        else:
            dis_ori_df = copy.deepcopy(dis_ori_df_)
        self.all_discrete_col = list(dis_ori_df.columns)
        present_col_dict_dis = {idx: [] for idx, col in enumerate(dis_ori_df.columns)}
        dis_ori_df.columns = list(present_col_dict_dis.keys())

        # 循环处理actions
        continuous_df, discrete_df = copy.deepcopy(con_ori_df), copy.deepcopy(dis_ori_df)
        for cycle_ops_list in self.actions:
            continuous_df, discrete_df = self.process_cycle_ops_list(cycle_ops_list, continuous_df, discrete_df)

        #z-score
        # continuous_idx_list = list(continuous_df.columns)
        # zscore_dict = {k: 'z_score' for k in continuous_idx_list}
        # zscore_param_list = [(col_name, continuous_df[col_name].values) for col_name in zscore_dict.keys()]
        # if len(zscore_dict) > 0:
        #     zscore_res = self.multi_thread(self.single_normalization, zscore_param_list)  # normalization
        #     for index, array_value in enumerate(list(zscore_dict.keys())):
        #         continuous_df[array_value] = zscore_res[index]

        # onehot
        if self.do_onehot:
            if len(discrete_df.columns) > 0:
                onehot_arr, onehot_col_list = self.onehot_process(discrete_df, list(discrete_df.columns))
                res_arr = np.concatenate((continuous_df.values,onehot_arr),axis=1)
            else:
                res_arr = continuous_df.values
        else:
            concat_df = pd.concat([continuous_df, discrete_df], axis=1)
            res_arr = concat_df.values
        # self.str_col_dict = self.numdict2strdict(present_col_dict)
        self.str_col_dict = present_col_dict
        return res_arr

    def numdict2strdict(self,present_col_dict):
        str_col_dict = {}
        for k,v in present_col_dict.items():
            str_v = []
            for ops_v in v:
                col_idx = ops_v[0]
                if isinstance(col_idx,int):
                    col = self.continuous_columns[col_idx]
                    str_v.append((col, ops_v[1]))
                else:
                    col0, col1 = self.continuous_columns[col_idx[0]],self.continuous_columns[col_idx[1]]
                    str_v.append(((col0, col1), ops_v[1]))
            str_col_dict[k] = str_v
        return str_col_dict

    def create_action_fes(self, actions, ori_dataframe,task_type='classifier', target_col=None, train=True, train_params=None,test= True):

        self.train = train
        self.test = test
        if self.train == True:
            self.fes_eng = FeatureEngTrain()
        else:
            if train_params:
                self.fes_eng = FeatureEngTest(*train_params)
            else:
                raise ValueError('Please check train_params, The test task must have "train_params"')
        self.actions = actions
        self.task_type = task_type
        if train or test:
            if target_col:
                self.target_col = target_col
            else:
                self.target_col = list(ori_dataframe)[-1]

            self.label = ori_dataframe[self.target_col].values
            self.ori_columns = list(ori_dataframe)
            self.ori_columns.remove(self.target_col)
        else:
            self.ori_columns = list(ori_dataframe)
            self.label = None
        if self.task_type == 'classifier':
            self.label = self.fes_eng.categories_to_int(self.label, self.target_col)
        ori_df = ori_dataframe.drop(columns=[self.target_col])

        return self.multi_run_thread(ori_df),self.label

    def _calculate_shape_onehot(self, actions, ori_dataframe):

        operation_idx_dict = {}

        #
        concat_num = 0
        combine_concat_num = 0
        for ls_batch in actions:
            for ls in ls_batch:
                if ls[-1] in ['concat', 'concat_END']:
                    if ls[-2] != 'combine':
                        if isinstance(ls[2], tuple):
                            concat_num += 2
                        if isinstance(ls[2], str):
                            concat_num += 1
                    else:
                        combine_concat_num += 1
        ori_continuous_idx = [i for i in range(int(len(self.continuous_columns) + concat_num))]

        if len(ori_continuous_idx) == 0:
            max_idx = -1
        else:
            max_idx = np.max(ori_continuous_idx)

        # onehot
        onehot_num = np.sum([len(ori_dataframe[col].unique()) for col in self.discrete_columns])
        if onehot_num > 0:
            onehot_idx = list(range(max_idx + 1, int(max_idx + 1 + onehot_num)))
            max_idx = np.max(onehot_idx)
        else:
            onehot_idx = []

        operation_idx_dict['ori_continuous_idx'] = ori_continuous_idx
        operation_idx_dict[ 'ori_discrete_idx'] = onehot_idx  # + continuous_bins_onehot_idx #
        operation_idx_dict['convert_idx'] = []
        operation_idx_dict['arithmetic_idx'] = []
        operation_idx_dict['combine_idx'] = []
        return max_idx + 1,operation_idx_dict

    def _calculate_shape_noonehot(self, actions, ori_dataframe):

        operation_idx_dict = {}

        # 索引字典
        concat_num = 0
        combine_concat_num = 0
        for ls_batch in actions:
            for ls in ls_batch:
                if ls[-1] in ['concat', 'concat_END']:
                    if ls[-2] != 'combine':
                        if isinstance(ls[2], tuple):
                            concat_num += 2
                        if isinstance(ls[2], str):
                            concat_num += 1
                    else:
                        combine_concat_num += 1

        ori_continuous_idx = [i for i in range(int(len(self.continuous_columns) + concat_num))]
        if len(ori_continuous_idx) == 0:
            max_idx = -1
        else:
            max_idx = np.max(ori_continuous_idx)

        #max_idx = np.max(ori_continuous_idx)
        # 离散 索引, 按离散分箱前
        if len(actions) == 0:
            all_discrete = len(self.discrete_columns)
        else:
            all_discrete = len(self.continuous_columns) + len(self.discrete_columns) + int(combine_concat_num)
        if all_discrete > 0:
            discrete_idx = list(range(max_idx + 1, int(max_idx + 1 + all_discrete)))
            max_idx = np.max(discrete_idx)
        else:
            discrete_idx = []

        operation_idx_dict['ori_continuous_idx'] = ori_continuous_idx
        operation_idx_dict[ 'ori_discrete_idx'] = discrete_idx  # + continuous_bins_onehot_idx # 原始离散索引为，onehot之后的 / 连续分箱onehot
        operation_idx_dict['convert_idx'] = []
        operation_idx_dict['arithmetic_idx'] = []
        operation_idx_dict['combine_idx'] = []
        return max_idx + 1,operation_idx_dict

    def calculate_shape(self, actions, ori_dataframe, target_col=None):
        if target_col:
            self.target_col = target_col
            calculate_shape_df = ori_dataframe.drop(columns = [self.target_col], axis = 1)
        else:
            calculate_shape_df = ori_dataframe.copy()
        if self.do_onehot:
            return self._calculate_shape_onehot(actions, calculate_shape_df)
        else:
            return self._calculate_shape_noonehot(actions, calculate_shape_df)

if __name__ == '__main__':
    df = pd.DataFrame([[1,2,3,6,7],[3,5,3,6,4],[1,5,9,8,7],[3,3,2,5,8],[4,5,6,7,8],[9,5,2,1,6],[6,4,2,7,3]],columns=['A','b','C','d','label'])
    continuous_columns = ['A','b']
    discrete_columns = ['C','d']
    do_onehot = False
    pipline_ins = Pipline(continuous_columns, discrete_columns,do_onehot)
    actions = [
        # [['A', 'b', ('tanh', 'tanh'), 'replace'] ,['b', 'd', ('log', 'sigmoid'), 'concat']],
        [['A', 'C', 'combine', 'concat'] ,['A', 'b', 'multiply', 'replace']],
        [['A', 'b', 'add', 'concat'] , ['C', 'd', 'combine', 'concat']]
        ]
    # actions = [
    #     [['C', 'd', 'combine', 'concat']]
    # ]
    # actions = []
    res, label = pipline_ins.create_action_fes(actions, df, task_type='classifier', target_col=None, train=True, train_params=None, test=True)
    # print(pipline_ins.str_col_dict)
    # res_math, label = pipline_ins.create_action_fes(actions, df, task_type='classifier', target_col=None, train=True, train_params=None, test=True)
    # train_p = pipline_ins.fes_eng.get_train_params()
    # test_res, test_label = pipline_ins.create_action_fes(actions, df.tail(5).reset_index(drop=True), task_type='classifier', target_col=None, train=False,
    #                                            train_params=train_p, test=True)