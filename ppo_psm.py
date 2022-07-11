import copy
import math

import pandas as pd
import torch.nn.functional as F

from cfs_trans import Multi_p
from controller_alpha import Controller
import torch.optim as optim
from worker_alpha import Worker
import numpy as np
import torch
import logging
from get_reward import GetReward
from dataset_split import DatasetSplit
from search_business import SearchBusiness
from constant import NO_ACTION
from multiprocessing import Pool
from pathlib import Path
import pickle


class PPO_psm(object):
    def __init__(self, args):
        self.args = args
        self.process_pool_num = 24
        self.arch_epochs = 400
        self.arch_lr = 0.001
        self.episodes = 24
        self.entropy_weight = 1e-3
        self.excited = 0.05
        self.policy_nums = 3
        self.task_type = args.task_type
        self.eval_method = args.eval_method
        self.final_action_ori = None
        self.ppo_epochs = 5
        device = torch.device('cuda', args.cuda) if torch.cuda.is_available() and args.cuda != -1 else torch.device(
            'cpu')
        self.controller = Controller(self.args).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(),
                               lr=self.arch_lr)
        self.baseline = None
        self.final_action = None
        self.baseline_weight = 0.9
        self.clip_epsilon = 0.2
        self.Ctrains = []
        self.base_score_list = []
        self.base_ori_score_list = []
        self.iterater = None
        # =========================================================
        self.base_ori_score_list = None
        # =========================================================
    def get_ori_base_score(self):
        get_reward_ins = GetReward(self.args)
        score_list = get_reward_ins.k_fold_score(self.search_data, NO_ACTION, is_base=True)
        return score_list

    def get_base_score(self, it):
        get_reward_ins = GetReward(self.args)
        score_list = get_reward_ins.k_fold_score(self.Ctrains[it], NO_ACTION, is_base=True)
        return score_list

    def pool_produce(self, worker):
        worker.get_k_fold_score(self.Ctrains[self.iterater])#self.search_data----->self.Ctrains[self.iterater])
        return worker

    def link_and_get(self, otps, opss):
        #link

        #softmax
        for l in range(2):
            for i in range(len(otps)):
                otps[i][l] = F.softmax(otps[i][l], dim=1)
                opss[i][l] = F.softmax(opss[i][l], dim=1)
        m = otps[0][0].shape[1]
        n = otps[0][0].shape[0]

        trans_actions = []
        for l in range(2):
            for lie in range(m):
                for han in range(n):
                    m1 = -float("inf")
                    m2 = -float("inf")
                    for i in range(len(otps)):
                        m1 = max(m1, otps[i][l][han][lie])
                        m2 = max(m2, opss[i][l][han][lie])
                    otps[0][l][han][lie] = m1
                    opss[0][l][han][lie] = m2
            #trans2action
            # M choose
            opt_info = self.controller.infer_trans(otps[0][l], 'ops_type')
            # T choose
            ops_info = self.controller.infer_trans(opss[0][l], 'ops')
            # T-de
            ops_actions_name = ops_info['action_name']
            opt_actions_name = opt_info['action_name']
            # action-de
            trans_action = self.controller.actions_trans(
                ops_actions_name, opt_actions_name)
            trans_actions.append(trans_action)
        return trans_actions

    def feature_search(self):
        self.base_ori_score_list = self.get_ori_base_score()
        # subsampling
        dataset_split = DatasetSplit(self.args)
        for i in range(self.policy_nums):
            train_i, test_i = dataset_split.split_dataset_with_ratio(self.search_data, 0.2, random_state=i)
            train_i.index = range(0, len(train_i))
            self.Ctrains.append(train_i)
            self.base_score_list.append(self.get_base_score(i))

        best_otps = []
        best_opss = []
        flag = 1
        if self.eval_method == 'mse':
            flag = -1
        elif self.eval_method == 'f1_score' or \
                self.eval_method == 'r_squared' or self.eval_method == 'acc' \
                or self.eval_method == 'auc' or self.eval_method == 'ks' or \
                self.eval_method == 'sub_rae':
            flag = 1
        logging.info(f'ori_base_score: {self.base_ori_score_list}, '
              f'ori_base_score_mean: {np.mean(self.base_ori_score_list)}, '
              f'ori_search_data_shape: {self.search_data.shape}')

        for k in range(self.policy_nums):
            self.iterater = k
            ma = -float("inf")
            best_ops = None
            best_otp = None

            logging.info(f'base_score: {self.base_score_list[k]}, '
                         f'base_score_mean: {np.mean(self.base_score_list[k])}, '
                         f'search_data_shape: {self.Ctrains[k].shape}')
            # print(f'base_score: {self.base_score_list[k]}, '
            #       f'base_score_mean: {np.mean(self.base_score_list[k])}, '
            #       f'search_data_shape: {self.Ctrains[k].shape}')

            for arch_epoch in range(self.arch_epochs):
                #print(arch_epoch)
                logging.info(
                    f'arch_epoch: {arch_epoch}, '
                    f'baseline: {self.baseline}, \r'
                )

                sample_workers = []
                ids = []
                for try_num in range(self.episodes):
                    # Sample from network
                    tmp = []
                    sample_history = self.controller.sample()
                    # Instantiate worker
                    worker = Worker(self.args, sample_history)
                    tmp.append(worker.actions_trans)
                    tmp.append(try_num)
                    ids.append(tmp)
                    sample_workers.append(worker)
                    # print('test_point_worker', worker.actions_trans)
                multipro = Multi_p(self.args, self.Ctrains[self.iterater])
                ids = multipro.multi_c(self.process_pool_num, ids)
                for i in range(self.episodes):
                    tmp = ids[i]
                    worker = sample_workers[tmp[1]]
                    worker.score_list = tmp[0]
                    worker.rep_num = tmp[2]
                    worker.fe_num = tmp[3]


                # =========================================================
                for worker in sample_workers:
                    rep_score_list = []
                    for idx, single_score in enumerate(worker.score_list):
                        # 减去没有动作的得分
                        score = single_score - self.base_score_list[k][idx]
                        rep_score_list.append(score)
                    worker.score = (np.mean(rep_score_list) - 0.01 * worker.rep_num)
                    if worker.score > ma:
                        ma = worker.score
                        best_ops = worker.ops_logits
                        best_otp = worker.otp_logits
                # =========================================================

                #print
                #=========================================================
                for worker in sample_workers:
                    logging.info(
                        f'sample_worker_score: {worker.score}, '
                        f'score_list: {worker.score_list}, '
                        f'score_mean: {np.mean(worker.score_list)}, '
                        f'worker_actions: {worker.actions_trans}, '
                        f'hp_action: {worker.hp_action}'
                    )
                #=========================================================

                # update baseline
                # =========================================================
                sample_score = [worker.score for worker in sample_workers]
                if self.baseline is None:
                    self.baseline = np.mean(sample_score)
                else:
                    for worker in sample_workers:
                        self.baseline = self.baseline * self.baseline_weight + \
                                        worker.score * (1-self.baseline_weight)
                # =========================================================

                # =========================================================
                score_lists = [worker.score for worker in sample_workers]
                score_reward = \
                    [score_list - self.baseline for score_list in score_lists]

                # ***************************************************
                score_reward_abs = [abs(reward) for reward in score_reward]
                min_reward = min(score_reward_abs)
                max_reward = max(score_reward_abs)
                score_reward_minmax = []
                for reward in score_reward:
                    if reward < 0:
                        min_max_reward = -(abs(reward) - min_reward) / \
                                         (max_reward - min_reward + 1e-6)
                    elif reward >= 0:
                        min_max_reward = (abs(reward) - min_reward) / \
                                         (max_reward - min_reward + 1e-6)
                    else:
                        min_max_reward = None
                        print('error')

                    score_reward_minmax.append(min_max_reward)
                score_reward_minmax = [i/2 for i in score_reward_minmax]
                # ***************************************************

                normal_reward = score_reward_minmax
                print('normal_reward', normal_reward)

                # PPO network update
                # =========================================================
                # Cumulative gradient update method
                for ppo_epoch in range(self.ppo_epochs):
                    loss = 0
                    for episode, worker in enumerate(sample_workers):
                        prod_history = self.controller.get_prod(worker.history)
                        #
                        loss += self.call_worker_loss(
                            prod_history, worker.history, normal_reward[episode])
                    loss /= len(sample_workers)
                    logging.info(f'epoch: {ppo_epoch}, loss: {loss}')
                    # # cal. loss
                    loss.backward()
                    # update
                    self.adam.step()
                    # ->zero
                    self.adam.zero_grad()
                # =========================================================
            best_opss.append(best_ops)
            best_otps.append(best_otp)
            logging.info(f'best_reward: {ma}, k-th policy: {k}\r')
        self.final_action = self.link_and_get(best_otps, best_opss)

    def call_worker_loss(self, prod_history, sample_history, reward):
        """
        calulate loss
        """
        policy_loss_list = []
        prod_actions_p = []
        sample_actions_p = []
        actions_entropy = []

        for ph in prod_history['feature_engineering']:
            prod_actions_p.append(ph['prob_vec'])
            actions_entropy.append(ph['action_entropy'])
        try:
            prod_actions_p.append(prod_history['hyper_parameter']['prob_vec'])
            actions_entropy.append(
                prod_history['hyper_parameter']['action_entropy'])
        except Exception as e:
            pass

        for sh in sample_history['feature_engineering']:
            sample_actions_p.append(sh['prob_vec'])
        try:
            sample_actions_p.append(
                sample_history['hyper_parameter']['prob_vec'])
        except Exception as e:
            pass

        actions_entropy = torch.cat(actions_entropy, dim=0)
        all_entropy = torch.sum(actions_entropy)

        for index, action_p in enumerate(prod_actions_p):
            action_importance = action_p / sample_actions_p[index]
            clipped_action_importance = self.clip(action_importance)

            action_reward = action_importance * reward
            clipped_action_reward = clipped_action_importance * reward

            action_reward, _ = torch.min(
                torch.cat([action_reward.unsqueeze(0),
                           clipped_action_reward.unsqueeze(0)], dim=0), dim=0)
            policy_loss_list.append(torch.sum(action_reward))

        if self.eval_method == 'mse':
            all_policy_loss = sum(policy_loss_list)
        elif self.eval_method == 'f1_score' or \
                self.eval_method == 'r_squared' or self.eval_method == 'acc' \
                or self.eval_method == 'auc' or self.eval_method == 'ks' or \
                self.eval_method == 'sub_rae':
            all_policy_loss = -1 * sum(policy_loss_list)
        else:
            logging.info(f'incre defaut')
            all_policy_loss = -1 * sum(policy_loss_list)

        entropy_bonus = -1 * all_entropy * self.entropy_weight
        # print('loss: ', all_policy_loss, entropy_bonus)

        return all_policy_loss + entropy_bonus

    def clip(self, actions_importance):
        lower = (torch.ones_like(actions_importance) * (1 - self.clip_epsilon))
        upper = (torch.ones_like(actions_importance) * (1 + self.clip_epsilon))

        actions_importance, _ = torch.min(torch.cat(
            [actions_importance.unsqueeze(0), upper.unsqueeze(0)],
            dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat(
            [actions_importance.unsqueeze(0), lower.unsqueeze(0)],
            dim=0), dim=0)

        return actions_importance


if __name__ == '__main__':
    load_args = 'hzd_amend'
    # from main import args, dataset_config
    # search_dataset_info = dataset_config[load_args]
    # for key, value in search_dataset_info.items():
    #     setattr(args, key, value)
    #
    # ppo = PPO(args)
    # ppo.feature_search()



