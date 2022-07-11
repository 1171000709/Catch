import pickle
from pathlib import Path
import numpy as np
import os
import glob


class WorkerHistoryBusiness(object):
    def __init__(self):
        self.log_path = 'result_log'

    def get_file_name(self):
        file_path = os.listdir(self.log_path)
        return file_path

    def load_pickle_info(self):
        file_path_list = self.get_file_name()
        for file_path in file_path_list:
            path = Path(self.log_path) / file_path
            pkl_path = glob.glob(os.path.join(path, '*.pkl'))
            #
            if pkl_path:
                # print('pkl_path', pkl_path)
                total_workers_file = open(pkl_path[0], 'rb')
                total_workers = pickle.load(total_workers_file)
                print(total_workers[0]['config_key'])
                print('base_score', np.mean(total_workers[0]['base_score']))
                res_workers = []
                for workers in total_workers:
                    res_workers = res_workers + workers['sample_workers_info']
                for worker in res_workers:
                    worker['mean_score'] = np.mean(worker['score_list'])
                #
                res_workers.sort(
                    key=lambda item: item['mean_score'], reverse=True)
                print('top_one_mean_score: ', res_workers[0]['mean_score'])
                print('top_one_actions_trans: ',
                      len(res_workers[0]['actions_trans']),
                      res_workers[0]['actions_trans'],
                      res_workers[0]['hp_actions'])
            else:
                print('文件夹为空')

    # def load_pickle_workers(self):
    #     workers_path = Path('train_log') / self.pickle_path_name / \
    #                    Path(str(self.dataset_name) + '.pkl')
    #
    #     total_workers_file = open(workers_path, 'rb')
    #     total_workers = pickle.load(total_workers_file)
    #     return total_workers
    #
    def get_top_k_limit_epoch(self, top_k, limit_epoch=0):
        workers_list = self.total_workers[limit_epoch:]
        res_workers = []
        print(workers_list[0]['config_key'])
        print('base_score', np.mean(workers_list[0]['base_score']))
        for workers in workers_list:
            res_workers = res_workers + workers['sample_workers_info']
        # print('res_workers', res_workers)
        # mean_score_list = []
        for worker in res_workers:
            worker['mean_score'] = np.mean(worker['score_list'])
        # print('res_workers', res_workers)
        # 排序
        res_workers.sort(key=lambda item: item['mean_score'], reverse=True)
        return res_workers[0:top_k]
    #
    # def get_top_one_info(self):
    #     res_info = self.get_top_k_limit_epoch(1)
    #     print('top1_mean_score: ', res_info[0]['mean_score'])
    #     print('top1_actions: ', res_info[0]['actions_trans'])


if __name__ == '__main__':
    worker_history = WorkerHistoryBusiness()

    worker_history.load_pickle_info()
