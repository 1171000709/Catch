import random
from constant import Operation
import numpy as np
import pandas as pd
from pathlib import Path
from get_reward import GetReward
from utility.base_utility import BaseUtility
from multiprocessing import Process


class Worker(object):
    def __init__(self, args, history):
        self.args = args
        self.history = history
        self.continuous_col = args.continuous_col
        self.discrete_col = args.discrete_col

        self.fe_history = history['feature_engineering']
        self.hp_history = history['hyper_parameter']

        #
        self.actions_trans = self.get_actions_trans(self.fe_history)
        self.actions_prob = [sample['prob_vec'] for sample in self.fe_history]
        self.ops_logits = [sample['ops_logits'] for sample in self.fe_history]
        self.otp_logits = [sample['otp_logits'] for sample in self.fe_history]

        #
        if self.hp_history is not None:
            self.hp_action = self.hp_history['action_name']
        else:
            self.hp_action = None

        self.params_size = None
        self.score = None
        self.score_list = []
        self.index_num = None
        self.rep_num = None
        self.fe_num = None

    @staticmethod
    def get_actions_trans(history):
        actions_trans = []
        end_feature = []
        for sample in history:
            actions = sample['trans_actions']

            batch_actions = []
            for action in actions:
                if action[0] not in end_feature:
                    batch_actions.append(action)

                if action[-1] in ['concat_END', 'replace_END']:
                    end_feature.append(action[0])

            actions_trans.append(batch_actions)

        return actions_trans

    def get_k_fold_score(self, search_data):
        get_reward_ins = GetReward(self.args)
        score_list, fe_num = get_reward_ins.k_fold_score(
            search_data, self.actions_trans, self.hp_action)
        self.score_list = score_list
        self.rep_num = get_reward_ins.rep_num
        self.fe_num = fe_num


