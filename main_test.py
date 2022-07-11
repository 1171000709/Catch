# -*- coding: utf-8 -*-
import copy
import logging
import time
import os
import sys
import argparse
import random

import arff
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

import reduce_scale
from config import dataset_config
from get_reward import GetReward
from ppo_ori import PPO_ori
import feature_type_recognition

# def set_cuda(cuda):
#     os.environ["CUDA_VISIBLE_DEVICES"] = cuda
#     global is_cuda, device
#     is_cuda = torch.cuda.is_available()
#     torch.device('cuda', args.cuda) if torch.cuda.is_available() and args.cuda != -1  else torch.device('cpu')
from utility.model_utility import sub_rae


def log_config(args):
    """
    log 配置信息，指定输出日志文件保存路径等
    :return: None
    """
    dataset_path = Path(args.dataset_path)
    dataset_name = dataset_path.stem
    exp_dir = \
        'search_{}_{}'.format(dataset_name, time.strftime("%Y%m%d-%H%M%S"))
    exp_log_dir = Path('Catch_log') / exp_dir
    #  args
    setattr(args, 'exp_log_dir', exp_log_dir)

    if not os.path.exists(exp_log_dir):
        os.mkdir(exp_log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(exp_log_dir / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def parse_args():
    parser = argparse.ArgumentParser(description='cfs')
    parser.add_argument('--data', type=str, default="Housing_Boston", help='dataset name')
    parser.add_argument('--cuda', type=int, default=6, help='which gpu to use')#-1 represent cpu-only
    parser.add_argument('--coreset', type=int, default=0, help='whether to use coreset')# 1 represent work with coreset
    parser.add_argument('--core_size', type=int, default=10000, help='size of coreset')
    args = parser.parse_args()

    # 把 config.py
    search_dataset_info = dataset_config[args.data]
    for key, value in search_dataset_info.items():
        setattr(args, key, value)
    return args

def main():
    #
    _args = parse_args()
    # setup environments
    #set_cuda(_args.cuda)

    #read data
    # csv
    all_data = pd.read_csv(_args.dataset_path)
    # all_data.replace('nan', np.nan)
    # all_data.fillna(0, inplace=True)
    #automatic recognition
    if (not _args.continuous_col) and (not _args.discrete_col):
        T = feature_type_recognition.Feature_type_recognition()
        T.fit(all_data)
        T.num.remove(_args.target_col)
        _args.continuous_col = T.num
        _args.discrete_col = T.cat

    features = _args.continuous_col + _args.discrete_col
    label = _args.target_col
    all_data = all_data[features + [label]]
    # log
    log_config(_args)
    logging.info(f'args : {_args}')
    # best_trans = [[['atemp', 'sigmoid', 'replace'], ['humidity', 'inverse', 'replace'], ['windspeed', 'sqrt', 'replace'],
    #               ['casual', 'windspeed', 'add', 'concat'], ['registered', 'casual', 'add', 'replace'],
    #               ['holiday', 'season', 'combine', 'concat'], ['workingday', 'weather', 'combine', 'replace'],
    #               ['weather', 'season', 'combine', 'concat']],
    #              [['temp', 'casual', 'subtract', 'replace'], ['atemp', 'registered', 'add', 'concat'],
    #               ['humidity', 'registered', 'add', 'concat'], ['windspeed', 'temp', 'subtract', 'concat'],
    #               ['registered', 'inverse', 'concat'], ['season', 'holiday', 'combine', 'concat'],
    #               ['holiday', 'season', 'combine', 'replace'], ['weather', 'season', 'combine', 'replace']]]
    # best_trans = [
    #     [['oz1', 'oz2', 'add', 'replace'], ['oz2', 'oz3', 'add', 'concat'], ['oz3', 'oz4', 'subtract', 'concat'],
    #      ['oz4', 'sqrt', 'concat'], ['oz5', 'oz2', 'multiply', 'concat'], ['oz6', 'oz19', 'add', 'replace'],
    #      ['oz7', 'oz11', 'multiply', 'replace'], ['oz8', 'oz23', 'subtract', 'replace'],
    #      ['oz10', 'oz12', 'subtract', 'replace'], ['oz11', 'oz3', 'divide', 'concat'],
    #      ['oz12', 'oz2', 'subtract', 'replace'], ['oz13', 'oz12', 'divide', 'replace'],
    #      ['oz14', 'oz3', 'divide', 'replace'], ['oz15', 'oz18', 'add', 'replace'], ['oz16', 'oz25', 'add', 'replace'],
    #      ['oz18', 'oz7', 'add', 'concat'], ['oz20', 'oz9', 'add', 'replace'], ['oz21', 'square', 'replace'],
    #      ['oz22', 'oz5', 'multiply', 'concat'], ['oz23', 'sigmoid', 'replace'], ['oz24', 'inverse', 'replace'],
    #      ['oz25', 'oz24', 'multiply', 'replace']],
    #     [['oz1', 'oz2', 'add', 'concat'], ['oz2', 'oz3', 'add', 'replace'], ['oz4', 'oz5', 'add', 'replace'],
    #      ['oz5', 'oz1', 'multiply', 'replace'], ['oz6', 'oz1', 'add', 'replace'], ['oz7', 'oz6', 'add', 'concat'],
    #      ['oz8', 'oz12', 'add', 'replace'], ['oz9', 'oz6', 'add', 'replace'], ['oz10', 'oz12', 'subtract', 'replace'],
    #      ['oz11', 'oz23', 'add', 'replace'], ['oz12', 'oz8', 'add', 'replace'], ['oz13', 'oz15', 'subtract', 'replace'],
    #      ['oz15', 'oz13', 'divide', 'replace'], ['oz16', 'oz21', 'divide', 'replace'], ['oz18', 'inverse', 'replace'],
    #      ['oz21', 'oz7', 'add', 'replace'], ['oz22', 'oz18', 'subtract', 'replace'],
    #      ['oz23', 'oz14', 'subtract', 'replace'], ['oz24', 'oz17', 'add', 'replace'],
    #      ['oz25', 'oz9', 'divide', 'concat']]]
    # best_trans = [
    #     [['RESOURCE', 'ROLE_FAMILY_DESC', 'combine', 'concat'], ['MGR_ID', 'ROLE_DEPTNAME', 'combine', 'concat'],
    #      ['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'combine', 'concat'],
    #      ['ROLE_ROLLUP_2', 'ROLE_ROLLUP_1', 'combine', 'concat'], ['ROLE_TITLE', 'ROLE_FAMILY', 'combine', 'replace'],
    #      ['ROLE_FAMILY_DESC', 'MGR_ID', 'combine', 'concat'], ['ROLE_FAMILY', 'ROLE_ROLLUP_1', 'combine', 'replace'],
    #      ['ROLE_CODE', 'ROLE_TITLE', 'combine', 'replace']],
    #     [['MGR_ID', 'ROLE_DEPTNAME', 'combine', 'concat'], ['ROLE_ROLLUP_1', 'ROLE_FAMILY_DESC', 'combine', 'replace'],
    #      ['ROLE_ROLLUP_2', 'ROLE_FAMILY', 'combine', 'concat'], ['ROLE_TITLE', 'ROLE_ROLLUP_1', 'combine', 'replace'],
    #      ['ROLE_FAMILY', 'MGR_ID', 'combine', 'concat'], ['ROLE_CODE', 'ROLE_TITLE', 'combine', 'replace']]]
    best_trans = [
        [['V0', 'V5', 'add', 'replace'], ['V1', 'V12', 'subtract', 'concat'], ['V2', 'V6', 'subtract', 'concat'],
         ['V4', 'V6', 'add', 'replace'], ['V5', 'V12', 'subtract', 'concat'], ['V6', 'V2', 'divide', 'concat'],
         ['V8', 'V4', 'multiply', 'concat'], ['V9', 'V4', 'multiply', 'concat'], ['V10', 'V4', 'subtract', 'concat'],
         ['V11', 'V7', 'divide', 'concat'], ['V12', 'V1', 'subtract', 'concat']],
        [['V0', 'V9', 'add', 'concat'], ['V1', 'V4', 'multiply', 'replace'], ['V2', 'V7', 'multiply', 'replace'],
         ['V4', 'V2', 'divide', 'replace'], ['V5', 'log', 'concat'], ['V6', 'V7', 'multiply', 'replace'],
         ['V7', 'V2', 'multiply', 'replace'], ['V8', 'V6', 'divide', 'concat'], ['V9', 'square', 'replace'],
         ['V10', 'V12', 'subtract', 'replace'], ['V11', 'log', 'replace'], ['V12', 'inverse', 'replace']]]

    pre = all_data.shape[1] - 1
    logging.info(f'n:{pre}')
    get_reward_ins = GetReward(_args)
    res_tuple =get_reward_ins.feature_pipline_train(best_trans, all_data)
    if res_tuple is None:
        return None
    search_fes, search_label, fe_params, operation_idx_dict = res_tuple
    search_fes = pd.DataFrame(search_fes)
    now = search_fes.shape[1] - pre
    logging.info(f'new_n:{now}')
    for k in range(0, now):
        ok = copy.deepcopy(search_fes)
        y = []
        for j in range(0, now):
            if j!=k:
                y.append(ok.columns[j])
        ok.drop(y, axis = 1 ,inplace= True)
        model = get_reward_ins.get_rf_model(None)
        if _args.task_type == 'classifier':
            score_list = cross_val_score(model, ok, search_label, scoring="f1_micro", cv=5)
        else:
            subrae = make_scorer(sub_rae, greater_is_better=True)
            score_list = cross_val_score(model, ok, search_label, scoring=subrae, cv=5)
        logging.info(f'k:{k},score:{score_list.mean()}')

    for k in range(1, now-1):
        ok = copy.deepcopy(search_fes)
        z = pre-1
        y = []
        yy = []
        l = []
        for j in range(0, k):
            z = random.randint(z+1, ok.shape[1]-1-(k-j-1))
            y.append(ok.columns[z])
            yy.append(z-pre)
        for x in range(0, ok.shape[1]-pre):
            if (x in yy):
                pass
            else:
                l.append(x)
        logging.info(f'y:{l}')
        ok.drop(y, axis = 1, inplace = True)
        model = get_reward_ins.get_rf_model(None)
        if _args.task_type == 'classifier':
            score_list = cross_val_score(model, ok, search_label, scoring="f1_micro", cv=5)
        else:
            subrae = make_scorer(sub_rae, greater_is_better=True)
            score_list = cross_val_score(model, ok, search_label, scoring=subrae, cv=5)
        logging.info(f'k:{now-k},score:{score_list.mean()}')

if __name__ == '__main__':
    old_time = time.time()

    main()



