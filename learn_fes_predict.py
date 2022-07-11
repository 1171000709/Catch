# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:25:11 2021

@author: yj
"""

import os
import random
import time
import pandas as pd
from pipline_nas import pipline as Pipline
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score

def get_sample_data(ori_dataframe,rate,is_balance=True,pos_target=1):
    x_train, x_test, y_train, y_test = train_test_split(ori_dataframe.iloc[:,:-1], ori_dataframe.iloc[:,-1], test_size = 0.2, shuffle= True,stratify=ori_dataframe.iloc[:,-1])
    test_data = x_test.merge(pd.DataFrame(y_test),left_index=True,right_index=True,how='left')
    res_data = x_train.merge(pd.DataFrame(y_train),left_index=True,right_index=True,how='left')
    
    if is_balance == False:
        neg_mut = int(rate.split(':')[0])
        pos_data_idx = res_data[res_data.iloc[:,-1] == pos_target].index.tolist()
        neg_data_idx = res_data[res_data.iloc[:,-1] != pos_target].index.tolist()
        random.shuffle(neg_data_idx)
        neg_data_idx = random.sample(neg_data_idx, neg_mut* len(pos_data_idx))
        train_data = res_data[res_data.index.isin(pos_data_idx + neg_data_idx)]
        return train_data,test_data

def save_data(train_data,test_data):
    
    save_path = os.path.join(r'data\sample' , time.strftime("%Y%m%d-%H%M%S"))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_data.to_csv(os.path.join(save_path , 'train_data.csv'),index=False )
    test_data.to_csv(os.path.join(save_path , 'test_data.csv'),index=False )
    return save_path.split('\\')[-1]
    
def resample():
    #
    file_name =r'data\credit_dataset.csv'
    ori_dataframe = pd.DataFrame(pd.read_csv(file_name))
    train_data,test_data = get_sample_data(ori_dataframe,rate='13:1',is_balance=False,pos_target=1)
    data_dir = save_data(train_data,test_data)
    return data_dir
 
def load_data(data_dir):
    data_path = f'data\sample\{data_dir}'
    train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    
    return train_data,test_data


def clear_traindict():
    with open(r'global_params.py','w') as f:
        f.write('feature_eng_bins_dict = {}' 
                + '\n feature_eng_combine_dict= {}'
                + '\n feature_normalization_dict= {}'
                )
        
        f.close()

# def write_traindict():
#     with open(r'global_params.py','w') as f:
#         f.write('feature_eng_bins_dict = ' +str(feature_eng_bins_dict) 
#                 + '\n feature_eng_combine_dict=' + str(feature_eng_combine_dict)
#                 + '\n feature_normalization_dict=' + str(feature_normalization_dict)
#                 )
        
#         f.close()


def predict_fe(data_dir,fe_pipline, actions):
    train_data,test_data = load_data(data_dir)
    train_num = len(train_data)
    all_data = train_data.append(test_data)

    new_fes,label = fe_pipline.create_action_fes(actions, all_data)
    
    new_train_fes = new_fes[:train_num,:]
    train_label = label[:train_num]
    
    new_test_fes = new_fes[train_num:,:]
    test_label = label[train_num:]
    
    model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False).fit(new_train_fes,train_label)

    y_pred = model.predict(new_test_fes)
    f1 = f1_score(test_label,y_pred)
    print(f'f1 score: {f1}')
    return f1
    

if __name__ == '__main__':
    start = time.time()
    # global data_dir
    data_dir = '20210508-180819'
    # data_dir = resample()
    continuous_columns = ['YEARS_EMPLOYED', 'BEGIN_MONTH', 'AGE', 'INCOME']
    discrete_columns = ['GENDER', 'CAR', 'REALITY', 'NO_OF_CHILD', 'INCOME_TYPE', 'EDUCATION_TYPE', 'FAMILY_TYPE',
                        'HOUSE_TYPE', 'WORK_PHONE', 'PHONE', 'E_MAIL', 'FAMILY SIZE']
    fe_pipline = Pipline(continuous_columns, discrete_columns)
    
    actions = {'convert_col': {'YEARS_EMPLOYED': 'log', 'BEGIN_MONTH': 'inverse', 'AGE': 'log', 'INCOME' :'sqrt'}
            , 'add_col': ['YEARS_EMPLOYED']
            , 'subtract_col': []
            , 'multiply_col': ['AGE']
            , 'divide_col': ['BEGIN_MONTH']
            , 'continuous_bins': {'YEARS_EMPLOYED': 10, 'BEGIN_MONTH': 2, 'AGE': 10, 'INCOME': 5}
            , 'discrete_bins': {'GENDER': 5, 'CAR': 5, 'REALITY': 7, 'NO_OF_CHILD': 5, 'INCOME_TYPE': 5, 'EDUCATION_TYPE': 3, 'FAMILY_TYPE': 5, 'HOUSE_TYPE': 5, 'WORK_PHONE': 7, 'PHONE': 5, 'E_MAIL': 5, 'FAMILY SIZE': 3}
            , 'combine_2': ['YEARS_EMPLOYED', 'INCOME_TYPE']
            , 'combine_3': ['BEGIN_MONTH', 'AGE', 'INCOME', 'FAMILY SIZE']
            , 'combine_4': ['GENDER', 'REALITY', 'EDUCATION_TYPE']}
    f1 = predict_fe(data_dir,fe_pipline, actions)
    end = time.time()
    print( end - start)
