"""
Created on 2021.4.25 11:32
@author: zly
"""

import numpy as np
from itertools import combinations

#


def categories_to_int(col):
    """
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    """
    unique_type = np.unique(np.array(col))
    categories_map = {}
    for i, type in enumerate(unique_type):
        categories_map[type] = i
    new_fe = np.array([categories_map[x] for x in col])
    return new_fe


def replace_abnormal(col):
    ''''''
    #
    # mean,std = np.mean(col),np.std(col)
    # floor,upper = mean - 3 * std, mean + 3 * std
    # col_replaced = [float(np.where(((x<floor)|(x>upper)), mean, x)) for x in col]
    #
    percent_25,percent_50,percent_75 = np.percentile(col,(25,50,75))
    #
    IQR = percent_75 - percent_25
    floor,upper = percent_25 - 1.5 * IQR,percent_75 + 1.5 * IQR
    #
    col_replaced = [float(np.where((x < floor), floor, x)) for x in col]
    col_replaced = [float(np.where((x > upper), upper, x)) for x in col_replaced]
    return np.array(col_replaced).reshape(len(col_replaced),1)

def combine_feature_tuples(feature_list,combine_type):
    '''
    tuple
    :type feature_list: list
    :type combine_type: int
    :rtype: list of tuples like[(A,B),(B,C)]
    '''
    return list(combinations(feature_list,combine_type))

def ff(x,fre_list):
    '''
    #
    :type x: float,
    :type fre_list: list of floats,
    '''
    if x<=fre_list[0]:
        return 0
    elif x>fre_list[-1]:
        return len(fre_list)
    else :
        for i in range(len(fre_list)-1):
            if x>fre_list[i] and x<=fre_list[i+1]:
                return i+1
            
            
def calculate_chi2(col,label):
    '''
    #
    :type col: list or np.array,
    :type label: list or np.array
    '''
    if not isinstance(label[0],(int,float)):
        label = categories_to_int(label)
    col = np.array(col)
    target_total = np.sum(label)
    target_len = len(label)
    #
    expect_ratio = target_total / target_len
    feature_unique_values = list(set(list(col)))
    chi2_dict = {}
    for value in feature_unique_values:
        #
        indexs = np.argwhere(col == value).reshape(-1)
        target_of_value = label[indexs]
        target_of_value_sum = np.sum(target_of_value)
        target_of_value_len = len(target_of_value)
        expected_target_sum = target_of_value_len * expect_ratio
        chi2 = (target_of_value_sum - expected_target_sum)**2 / expected_target_sum
        chi2_dict[value] = chi2
    chi2_dict_sorted = dict(sorted(chi2_dict.items(),key = lambda x:x[1],reverse=True))
    return chi2_dict_sorted





