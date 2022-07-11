import copy
import math

import pandas as pd
import torch.multiprocessing
import time

from sklearn.model_selection import cross_val_score

from cfs_trans import Multi_p
import cfs_trans
from controller_alpha import Controller
import torch.optim as optim

from feature_engineering.model_base import ModelBase
from utility.model_utility import sub_rae
from worker_alpha import Worker
import numpy as np
import torch
import logging
from get_reward import GetReward
from dataset_split import DatasetSplit
from search_business import SearchBusiness
from constant import NO_ACTION
import multiprocessing
from sklearn.metrics import make_scorer

class PPO_ori(object):
    def __init__(self, args):
        self.args = args
        self.process_pool_num = 24
        self.arch_epochs = 400
        self.arch_lr = 0.001
        self.episodes = 24
        self.entropy_weight = 1e-3

        self.task_type = args.task_type
        self.eval_method = args.eval_method

        self.ppo_epochs = 15
        device = torch.device('cuda', args.cuda) if torch.cuda.is_available() and args.cuda != -1  else torch.device('cpu')
        self.controller = Controller(self.args).to(device)
        #self.controller = Controller(self.args)
        self.adam = optim.Adam(params=self.controller.parameters(),
                               lr=self.arch_lr)
        self.baseline = None
        self.baseline_weight = 0.9
        self.clip_epsilon = 0.2
        self.search_data = None
        self.best_trans = None

    def get_base_score(self):
        # x = self.search_data.drop([self.args.target_col], axis = 1)
        # x = self.search_data[self.args.continuous_col + self.args.discrete_col]
        # y = self.search_data[self.args.target_col]
        # if self.task_type == 'classifier':
        #     model = ModelBase.rf_classify()
        #     score_list = cross_val_score(model, x, y, scoring="f1_micro", cv=5)
        # else:
        #     model = ModelBase.rf_regeression()
        #     subrae = make_scorer(sub_rae, greater_is_better=True)
        #     score_list = cross_val_score(model, x, y, scoring=subrae, cv=5)
        get_reward_ins = GetReward(self.args)
        score_list = get_reward_ins.k_fold_score(self.search_data, NO_ACTION, is_base=True)
        return score_list

    def feature_search(self):
        # Initial score
        self.base_score = self.get_base_score()
        best_score = 0
        improve = []
        # =========================================================

        logging.info(f'base_score: {self.base_score}, '
                     f'base_score_mean: {np.mean(self.base_score)}, '
                     f'search_data_shape: {self.search_data.shape}')
        logging.info(f'controller structure：{self.controller}')
        for arch_epoch in range(self.arch_epochs):
            last_time = time.time()

            # Sample actions and calculate the corresponding score
            # =========================================================
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
                #print('test_point_worker', worker.actions_trans)
            multipro = Multi_p(self.args, self.search_data)
            ids = multipro.multi_c(self.process_pool_num, ids)
            for i in range(self.episodes):
                tmp = ids[i]
                worker = sample_workers[tmp[1]]
                worker.score_list = tmp[0]
                worker.rep_num = tmp[2]
                worker.fe_num = tmp[3]
            # =========================================================

            # Network inference action
            # =========================================================
            # infer_history = self.controller.inference()
            # infer_worker = Worker(self.args, infer_history)
            # infer_actions_trans = infer_worker.actions_trans
            # infer_actions_prob = infer_worker.actions_prob
            # infer_hp_actions = infer_worker.hp_action
            # =========================================================

            # Take the k-fold reward and average it into one. Each reward takes the current score - baseline
            # =========================================================
            for worker in sample_workers:
                rep_score_list = []
                for idx, single_score in enumerate(worker.score_list):
                    #
                    score = single_score - self.base_score[idx]
                    rep_score_list.append(score)
                #worker.score = (np.mean(rep_score_list) - 0.01*worker.rep_num)
                worker.score = (np.mean(rep_score_list))
            # =========================================================

            # Printout
            # =========================================================
            for worker in sample_workers:
                mean_score = np.mean(worker.score_list)
                if best_score < mean_score:
                    best_score = mean_score
                    self.best_trans = worker.actions_trans
                logging.info(
                    f'rnn_cycle_num: {len(worker.actions_trans)}, '
                    f'sample_worker_score: {worker.score}, '
                    f'score_list: {worker.score_list}, '
                    f'score_list_mean: {np.mean(worker.score_list)}, '
                    #f'worker_actions: {worker.actions_trans}, '
                    f'fe_num:{worker.fe_num},'
                    f'hp_action: {worker.hp_action}'
                )
            # =========================================================

            # update baseline
            # =========================================================
            sample_score = [worker.score for worker in sample_workers]
            if self.baseline is None:
                self.baseline = np.mean(sample_score)
            else:
                for worker in sample_workers:
                    self.baseline = self.baseline * self.baseline_weight + \
                                    worker.score * (1-self.baseline_weight)
            improve.append(max(0, self.baseline) / self.base_score.mean())
            print(max(0, self.baseline) / self.base_score.mean())
            # =========================================================

            # Output infer related information
            # =========================================================
            logging.info(
                f'\n ************************************************* \n'
                f'data: {self.args.data} \n'
                f'arch_epoch: {arch_epoch} \n'
                f'baseline: {self.baseline} \n'
                # f'infer_actions_len: {len(infer_actions_trans)} \n'
                # f'infer_actions_trans: {infer_actions_trans} \n'
                # f'infer_hp_actions: {infer_hp_actions} \n'
                # f'infer_actions_prob: {infer_actions_prob} \n'
                f' **************************************************')
            # =========================================================

            # Save the workers locally to facilitate the subsequent analysis and search process
            # =========================================================
            # sample_workers_info = []
            # for worker in sample_workers:
            #     sample_workers_info.append({
            #         'score': worker.score,
            #         'score_list': worker.score_list,
            #         'actions_trans': worker.actions_trans,
            #         'hp_actions': worker.hp_action
            #     })
            # total_workers.append({
            #     'sample_workers_info': sample_workers_info,
            #     'infer_actions_trans': infer_actions_trans,
            #     'infer_hp_actions': infer_hp_actions,
            #     'baseline': self.baseline,
            #     'base_score': self.base_score,
            #     'config_key': self.args.config_key,
            #     'with_combine_search': self.controller.with_combine_search,
            #     'with_hyper_param_search': self.controller.with_hyper_param_search
            # })
            # if self.args.exp_log_dir:
            #     save_path = \
            #         self.args.exp_log_dir / Path(str(self.args.config_key)+'.pkl')
            # else:
            #     save_path = self.args.exp_log_dir / Path('total_workers.pkl')
            # with open(save_path, 'wb') as f:
            #     pickle.dump(total_workers, f)
            # =========================================================

            # reward standardization
            # =========================================================
            score_lists = [worker.score for worker in sample_workers]
            score_reward = [score_list - self.baseline for score_list in score_lists]

            # Standardization of division standard deviation
            # ***************************************************
            # score_reward_std = np.std(score_reward)
            # score_reward_mean = np.mean(score_reward)
            # score_reward_normal = \
            #     [(reward - score_reward_mean) / (score_reward_std + 1e-6)
            #      for reward in score_reward]
            # ***************************************************

            # Maximum and minimum normalization without changing sign
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

            # Normalization mode selection
            normal_reward = score_reward_minmax
            # normal_reward = score_reward
            #normal_reward = score_reward_normal
            #print('normal_reward', normal_reward)

            # PPO network update
            # =========================================================
            # Cumulative gradient update method
            for ppo_epoch in range(self.ppo_epochs):
                loss = 0
                for episode, worker in enumerate(sample_workers):
                    prod_history = self.controller.get_prod(worker.history)
                    loss += self.call_worker_loss(
                        prod_history, worker.history, normal_reward[episode])
                loss /= len(sample_workers)
                logging.info(f'training_rid: {ppo_epoch}, loss: {loss}')
                # #
                loss.backward()
                #
                self.adam.step()
                #
                self.adam.zero_grad()
            now_time = time.time()
            logging.info(f'arch_epoch_time:{now_time-last_time}\n'f' **************************************************')
            # =========================================================
        logging.info(f'baseline_improve:{improve}\n')
        logging.info(f'best_score:{best_score}\n' f'best_trans:{self.best_trans}\n')

    def call_worker_loss(self, prod_history, sample_history, reward):
        """
        Calculate loss function
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
            logging.info(f'The evaluation index does not exist, it will be increased by default')
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
    from main import args, dataset_config
    search_dataset_info = dataset_config[load_args]
    for key, value in search_dataset_info.items():
        setattr(args, key, value)

    ppo = PPO_ori(args)
    ppo.feature_search()



