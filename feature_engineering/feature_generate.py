"""
Created on 2021.4.9 10:32
@author: zly
"""
import gc
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
# from feature_engineering.global_params import feature_eng_bins_dict,feature_eng_combine_dict,feature_normalization_dict
from feature_engineering.utils import replace_abnormal
from feature_engineering.utils import ff,categories_to_int,calculate_chi2
from collections import Counter


#
def sqrt(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #
    try:
        # sqrt_col = np.sqrt(np.array(col))
        sqrt_col = [np.sqrt(x) if x>=0 else -np.sqrt(np.abs(x)) for x in col]
        sqrt_col = np.array(sqrt_col).reshape(len(sqrt_col),1)
        return sqrt_col
    except:
        raise ValueError('Value type error,check feature type')

def inverse_old(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    try:
        col = np.array(col)

        replace_col = np.array([float(np.where(x == 0., random.choice([1e-5,-1e-5]), x)) for x in col])
        new_col = replace_abnormal(1/replace_col)
        return new_col.reshape(len(new_col),1)
    except:
        raise ValueError('Value type error,check feature type')

def inverse(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    try:
        col = np.array(col)       
        # replace_num = np.mean(1 / np.array(list(filter(lambda x: x!=0 ,col ))) )
        new_col = np.array([1/x if x!=0 else 0 for x in col])

        return new_col.reshape(len(new_col),1)
    except:
        raise ValueError('Value type error,check feature type')


def log(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #
    try:
        log_col = [np.log(np.abs(x)) if x!=0 else 0 for x in col]
        log_col = np.array(log_col).reshape(len(log_col),1)
        
        return log_col
    except:
        raise ValueError('Value type error,check feature type')
        

"""
def normalization(col,col_index):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 特征z-core标准化
    # try:
    col = np.array(col)
    if col_index in feature_normalization_dict:
        mean,sigma = feature_normalization_dict[col_index]
    else:
        mean = np.mean(col)
        sigma = np.std(col)
        feature_normalization_dict[col_index] = (mean,sigma)
    new_col = []
    for x in col:
        scaled_x = (x-mean)/sigma
        new_col.append(scaled_x)
    return np.array(new_col).reshape(len(new_col),1)
    # except:
    #     raise ValueError('Value type error,check feature type')


    # try:
    #     scaled_col = preprocessing.scale(np.array(col))
    #     return scaled_col.reshape(len(scaled_col),1)
    # except:
    #     raise ValueError('Value type error,check feature type')


def max_min_old(col,col_index):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 特征min_max归一化,需要考虑多列情况

    col = np.array(col)
    if col_index in feature_normalization_dict:
        max,min = feature_normalization_dict[col_index]
        # print(col_index)
    else:
        max = np.max(col)
        min = np.min(col)
        feature_normalization_dict[col_index] = (max,min)
    new_col = []
    for x in col:
        x_scaled = float((x - min) / (max - min))
        new_col.append(x_scaled)
        # print(x_scaled)
    return np.array(new_col)
"""
def max_min(col,col_op):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #

    col = np.array(col).reshape(len(col),1)
    # print('max_min : -----', col)
    max = np.max(col)
    min = np.min(col)
    # feature_normalization_dict[str(col_op)] = (max,min)

    # new_col = np.apply_along_axis(lambda x :float((x - min) / (max - min)),axis=1,arr=col)
    new_col = [float((x - min) / (max - min))  if (max - min)!=0 else 0 for x in col ]

    return np.array(new_col).reshape(len(new_col))



def add(col1,col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 + col2).reshape(len(col1),1)
    except:
        raise ValueError('Value type error,check feature type')

def multiply(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 * col2).reshape(len(col1),1)
    except:
        raise ValueError('Value type error,check feature type')

def subtract(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    #
    try:
        col1 = np.array(col1).reshape(len(col1),1)
        col2 = np.array(col2).reshape(len(col2),1)
        return np.concatenate((col1 - col2,col2 - col1),axis = 1)
    except:
        raise ValueError('Value type error,check feature type')

def divide_old(col1,col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        replace_col1 = np.array([float(np.where(x == 0., random.choice([1e-5,-1e-5]), x)) for x in col1])
        replace_col2 = np.array([float(np.where(x == 0., random.choice([1e-5,-1e-5]), x)) for x in col2])
        col_d1,col_d2 = replace_col1/replace_col2,replace_col2/replace_col1
        col_d1 = replace_abnormal(col_d1)
        col_d2 = replace_abnormal(col_d2)
        return np.concatenate((col_d1,col_d2),axis = 1)
    except:
        raise ValueError('Value type error,check feature type')


def divide(col1,col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        col_d1 = np.array([col1[idx]/col2[idx] if col2[idx]!=0 else 0 for idx in range(len(col1))])
        col_d2 = np.array([col2[idx]/col1[idx] if col1[idx]!=0 else 0 for idx in range(len(col1))]) 
        col_d1 = replace_abnormal(col_d1)
        col_d2 = replace_abnormal(col_d2)
        col_d1 = col_d1.reshape(len(col_d1),1)
        col_d2 = col_d2.reshape(len(col_d2),1)
        return np.concatenate((col_d1,col_d2),axis = 1)
    except:
        raise ValueError('Value type error,check feature type')

def onehot_encoder(ori_fe):
    '''

    '''
    ori_fe = np.array(ori_fe).reshape(len(ori_fe),1)
    encoder = OneHotEncoder()
    enc = encoder.fit(ori_fe)
    onehot_fe = enc.transform(ori_fe).toarray()
    return onehot_fe

# 分箱操作
def merge(ori_fe,bins,fe_name = None):
    ''''''
    ori_fe = np.array(ori_fe)



"""

"""

def binning(ori_fe,bins,fe_name = None,method = 'frequency'):
    '''

    '''
    ori_fe = np.array(ori_fe)

    # k = bins - 1
    if method == 'frequency':

        fre_list = [np.percentile(ori_fe, 100 / bins * i) for i in range(1,bins)]
        fre_list = sorted(list(set(fre_list)))

        new_fe = np.array([ff(x,fre_list) for x in ori_fe])
        new_fe_encode = onehot_encoder(new_fe)
        return new_fe.reshape(len(new_fe),1),fre_list,new_fe_encode
    # 等距分箱
    elif method == 'distance':
        umax = np.percentile(ori_fe, 99.99)
        umin = np.percentile(ori_fe, 0.01)
        step = (umax - umin) / bins
        fre_list = [umin + i * step for i in range(bins)]

        new_fe = np.array([ff(x,fre_list) for x in ori_fe])
        new_fe_encode = onehot_encoder(new_fe)
        return new_fe.reshape(len(new_fe), 1),fre_list,new_fe_encode

def reset_value(ori_fe,c, merged_values, k):
    for merged_value in merged_values:
        indexs = np.argwhere(ori_fe == merged_value).reshape(-1)
        # 将原始ori_fe的低频率值今进行修改查找
        new_value = k + c # 这样基本能保证不会跟原始值重复
        ori_fe[indexs] = new_value


def recur_merge_regression(bins, frequency_list, value_types, residual_f , ori_fe):
    # 递归合并频概率变量，这里默认变量频率已经排序
    # 针对回归问题的版本
    k = len(ori_fe)
    if bins == 1:
        merged_values = value_types
        reset_value(ori_fe,len(value_types),merged_values, k)
        return
    target_frequency = residual_f / bins
    merged_f,merged_values,ptr = 0,[],0
    for i,f in enumerate(frequency_list):
        residual_f -= f
        ptr = i + 1
        if f < target_frequency:
            merged_f += f
            merged_values.append(value_types[i])
            if merged_f >= target_frequency:
                bins -= 1
                break
        else:
            bins -= 1
            break
    reset_value(ori_fe,len(value_types), merged_values, k)
    frequency_list,value_types = frequency_list[ptr:],value_types[ptr:]

    recur_merge_regression(bins,frequency_list,value_types,residual_f, ori_fe)



def recur_merge_classify(chi2_dict,bins,ori_fe):


    def merge_value_type(chi2_value_tuple, chi2_dict, c):
        chi2_1, chi2_2 = chi2_value_tuple
        if chi2_1 == chi2_2:
            index1 = list(chi2_dict.values()).index(chi2_1)
            index2 = index1 + 1
        else:
            index1 = list(chi2_dict.values()).index(chi2_1)
            index2 = list(chi2_dict.values()).index(chi2_2)
        value_type_of_chi2_1 = list(chi2_dict.keys())[index1]
        value_type_of_chi2_2 = list(chi2_dict.keys())[index2]
        new_chi2_value = chi2_1 + chi2_2
        k = len(ori_fe) + value_type_of_chi2_2 + value_type_of_chi2_1
        merged_values = [value_type_of_chi2_1, value_type_of_chi2_2]
        reset_value(ori_fe, c, merged_values, k)
        new_value_type = k + c
        chi2_dict[new_value_type] = new_chi2_value
        del chi2_dict[value_type_of_chi2_1]
        del chi2_dict[value_type_of_chi2_2]
        chi2_dict = dict(sorted(chi2_dict.items(), key=lambda x: x[1], reverse=True))
        return chi2_dict

    c = len(np.unique(ori_fe))
    while c > bins:
        chi2_value_list = np.array(list(chi2_dict.values()))
        chi2_value_tuple = (chi2_value_list[-1],chi2_value_list[-2])
        chi2_dict = merge_value_type(chi2_value_tuple,chi2_dict,c)
        c = len(list(chi2_dict.values()))


def binning_for_discrete(ori_fe, bins, mode, label,fe_name = None):


    ori_fe = categories_to_int(ori_fe)
    unique_value = list(set(list(ori_fe)))
    k = len(unique_value)
    if k <= bins:
        return np.array(ori_fe).reshape(-1,1)

    if mode == 'regression':
        n = len(ori_fe)
        # 1.先计算每个分类变量的frequency
        frequency = dict(Counter(ori_fe))
        sorted_frequency = dict(sorted(frequency.items(),key = lambda x:x[1], reverse = True))
        for key in sorted_frequency.keys():
            sorted_frequency[key] /= n
        frequency_list = list(sorted_frequency.values())
        value_types = list(sorted_frequency.keys())
        recur_merge_regression(bins,frequency_list,value_types,residual_f=1.0,ori_fe = ori_fe)

    else:

        sorted_chi2_dict = calculate_chi2(ori_fe,label)
        recur_merge_classify(sorted_chi2_dict,bins,ori_fe)
    ori_fe = categories_to_int(ori_fe)
    return ori_fe.reshape(len(ori_fe))

def cal_woe_iv(X,y,bins):
    '''
    '''
    if len(X) != len(y):
        raise KeyError('Feature length not equal to target length')
    bins_x,bond_list,encode_x = binning(X,bins)
    del encode_x #
    gc.collect()
    bins_x = np.array([x[0] for x in bins_x])
    bins_unique = np.unique(bins_x)
    y_positive_sum,y_negtive_sum = sum(y),len(y) - sum(y)

    bins_positive_negtive_dic = {}
    for bin_unique in bins_unique:
        bins_positive_negtive_dic[bin_unique] = [0,0] #
    for bin_x,t in zip(bins_x,y):
        if not t:
            bins_positive_negtive_dic[bin_x][1] += 1
        else:
            bins_positive_negtive_dic[bin_x][0] += 1
    # woe = np.log((yi/yt)/(ni/nt))
    woe,iv = [],[]
    for bin_x in bins_x:
        yi,ni = bins_positive_negtive_dic[bin_x]
        woe_i = np.log((yi/y_positive_sum)/(ni/y_negtive_sum))
        iv_i = ((yi/y_positive_sum) - (ni/y_negtive_sum)) * woe_i
        woe.append(woe_i)
        iv.append(iv_i)
    return np.array(woe).reshape(len(woe),1), iv




def col_names_maping(col_names,ori_cols,combine_features_name_list):
    '''
    '''
    hash_map = {}
    for i,col_name in enumerate(col_names):
        if col_name not in hash_map:
            hash_map[col_name] = i
        else:
            raise ValueError('Duplicate col_name in original features')
    selected_cols = []
    for feature_name in combine_features_name_list:
        selected_cols.append(hash_map[feature_name])
    extracted_features = ori_cols[:,selected_cols]
    return extracted_features


def combine_onehot(ori_fes,fe_names):

    col_unique_list = []
    col_unique_dict = {}
    for i,name in enumerate(fe_names):
        unique_value = np.unique(ori_fes[:,i])
        unique_dict = {value:str(idx) for idx,value in enumerate(unique_value)}
        col_unique_dict[name] = unique_dict
        col_unique_list.append(list(unique_dict.values()))
    from itertools import product
    composite_idx = list(product(*col_unique_list))
    composite_str = [''.join(list(tp)) for tp in composite_idx] # 唯一组合['00', '01', '10', '11']
    zero_array = np.zeros(shape=(len(ori_fes) , len(composite_str) + 1))
    new_col_composite_str = [[str(int(imp)) for imp in row] for row in ori_fes]
    new_col_composite_str = [''.join(lt) for lt in new_col_composite_str]
    
    for row,cp_str in enumerate(new_col_composite_str):
        if cp_str in composite_str:
            col = composite_str.index(cp_str)
            zero_array[row,col] = 1
        else:
            zero_array[row,-1] = 1
    # print(fe_names , zero_array.shape)
    return zero_array
    

def check_is_continuous(ori_fes,fe_names,continuous_columns,continuous_bins):
    for idx,name in enumerate(fe_names):
        if name in continuous_columns:
            bins = continuous_bins[name]
            if len(np.unique(ori_fes[:,idx])) < bins:
                raise ValueError(f'{name} unique value is {len(np.unique(ori_fes[:,idx]))} , but bins {bins}')
            fes_bins,_,_ = binning(ori_fes[:,idx],bins,fe_name = name,method = 'frequency')
            # print(fes_bins)
            # print(fes_bins.reshape(len(fes_bins),1))
            ori_fes[:,idx] = fes_bins.reshape(len(fes_bins))
    return ori_fes

def combine_noonehot(ori_fes,fe_names):
    #
    col_unique_list = []
    col_unique_dict = {}
    for i,name in enumerate(fe_names):
        unique_value = np.unique(ori_fes[:,i])
        unique_dict = {value:str(idx) for idx,value in enumerate(unique_value)}
        col_unique_dict[name] = unique_dict
        col_unique_list.append(list(unique_dict.values()))
    from itertools import product
    composite_idx = list(product(*col_unique_list))
    composite_str = [''.join(list(tp)) for tp in composite_idx] # 唯一组合['00', '01', '10', '11']
    new_col_composite_str = [[str(int(imp)) for imp in row] for row in ori_fes]
    new_col_composite_str = [''.join(lt) for lt in new_col_composite_str]
    combine_col = np.array([composite_str.index(cp_str) for cp_str in new_col_composite_str])
    
    return combine_col.reshape(-1,1)


