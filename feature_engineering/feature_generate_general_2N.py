# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import Counter
from feature_engineering.utils import ff,calculate_chi2 ,categories_to_int,replace_abnormal


def sqrt(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #
    try:
        # sqrt_col = np.sqrt(np.array(col))
        sqrt_col = [np.sqrt(x) if x>=0 else -np.sqrt(np.abs(x)) for x in col]
        sqrt_col = np.array(sqrt_col).reshape(-1,1)
        return sqrt_col
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

        return new_col.reshape(-1,1)
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
        log_col = np.array(log_col).reshape(-1,1)

        return log_col
    except:
        raise ValueError('Value type error,check feature type')




def add(col1,col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    #
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 + col2).reshape(-1,1)
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
        return (col1 * col2).reshape(-1,1)
    except:
        raise ValueError('Value type error,check feature type')

def subtract(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    #
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 - col2).reshape(-1,1)
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
        col_d1 = replace_abnormal(col_d1)
        return col_d1.reshape(-1,1)
    except:
        raise ValueError('Value type error,check feature type')

def square(col):
    try:
        col = np.array(col).reshape(-1)
        square_col = np.square(col).reshape(-1, 1)
        return square_col
    except:
        raise ValueError('Value type error,check feature type')

def power3(col):
    try:
        col = np.array(col).reshape(-1)
        res = np.power(col,3).reshape(-1, 1)
        return res
    except:
        raise ValueError('Value type error,check feature type')

def sigmoid(col):
    try:
        col = np.array(col).reshape(-1)
        res = 1 / (1 + np.exp(-col))
        return res.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')

def tanh(col):
    try:
        col = np.array(col).reshape(-1)
        res = 2 * sigmoid(2*col) - 1
        return res.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')

def abs(col):
    try:
        col = np.array(col).reshape(-1)
        res = np.abs(col).reshape(-1, 1)
        return res
    except:
        raise ValueError('Value type error,check feature type')

def max(col):
    try:
        col = np.array(col).reshape(-1)
        res = np.ones(shape= col.shape) * np.max(col)
        return res.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')

def min(col):
    try:
        col = np.array(col).reshape(-1)
        res = np.ones(shape= col.shape) * np.min(col)
        return res.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')

def round(col):
    try:
        col = np.array(col).reshape(-1)
        res = np.round(col,2).reshape(-1, 1)
        return res
    except:
        raise ValueError('Value type error,check feature type')

def none(col):
    try:
        col = np.array(col).reshape(-1)
        res = col.reshape(-1, 1)
        return res
    except:
        raise ValueError('Value type error,check feature type')
