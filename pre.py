from feature_engineering.model_base import ModelBase
import pandas as pd
import numpy as np

from get_reward import GetReward


def mk_features(args, num):
    if args.task_type == 'regression':
        model = ModelBase.rf_regeression()
    else:
        model = ModelBase.rf_classify()
    #dataset_path = 'data/AP_Omentum_Ovary.csv'
    dataset_path = args.dataset_path
    raw_df = pd.DataFrame(pd.read_csv(dataset_path))
    raw_df.replace('nan', np.nan)
    raw_df.fillna(0, inplace=True)
    N = raw_df.shape[1]
    print(N)
    # n = int(N * 0.2)
    get_reward_ins = GetReward(args)
    res_tuple = get_reward_ins.feature_pipline_train([], raw_df)
    train, label, fe_params, operation_idx_dict = res_tuple


    features = args.continuous_col + args.discrete_col
    # train = raw_df[features]
    # label = raw_df[args.target_col]
    model.fit(train, label)

    entroys = model.feature_importances_
    rank = [index for index, value in sorted(list(enumerate(entroys)), key=lambda x: x[1], reverse=True)]
    print(rank[0:num])
    ans = []
    for x in rank[0:num]:
        print(entroys[x])
        ans.append(features[x])
    print(ans)
    return ans

if __name__ == '__main__':
    model1 = ModelBase.rf_classify()
    model2 = ModelBase.rf_regeression()
    dataset_path = 'data/AP_Omentum_Ovary.csv'
    raw_df = pd.DataFrame(pd.read_csv(dataset_path))
    N = raw_df.shape[1]
    print(N)
    # n = int(N * 0.2)
    train = raw_df.iloc[:, 0:N-1]
    label = raw_df.iloc[:, [N-1]]
    model1.fit(train, label)
    entroys = model1.feature_importances_
    rank = [index for index, value in sorted(list(enumerate(entroys)), key=lambda x: x[1], reverse=True)]
    print(rank[0:50])
    ans = []
    for x in rank[0:50]:
        print(entroys[x])
        ans.append(raw_df.columns[x])
    print(ans)
    z = ['205913_at', '207175_at', '1555778_a_at', '213247_at', '220988_s_at', '212344_at', '229849_at', '216442_x_at', '212419_at', '219873_at', '229479_at', '204589_at', '201744_s_at', '209763_at', '203824_at', '227566_at', '209612_s_at', '201125_s_at', '200788_s_at', '218468_s_at', '235978_at', '209242_at', '201149_s_at', '203980_at', '214505_s_at', '204548_at', '209581_at', '220102_at', '225242_s_at', '209090_s_at', '227061_at', '235733_at', '201117_s_at', '223122_s_at', '225241_at', '201150_s_at', '213125_at', '225424_at', '219087_at', '212354_at', '225987_at', '240135_x_at', '37892_at', '212587_s_at', '205941_s_at', '221730_at', '212488_at', '225681_at', '210072_at', '202273_at']
    #z = ['222281_s_at', '205778_at', '37004_at', '228462_at', '223806_s_at', '219612_s_at', '211657_at', '223495_at', '206067_s_at', '215813_s_at', '228377_at', '227195_at', '231007_at', '207717_s_at', '227848_at', '231315_at', '231382_at', '239381_at', '219795_at', '219747_at', '228708_at', '218835_at', '206799_at', '33322_i_at', '238669_at', '209810_at', '205128_x_at','213936_x_at', '213917_at', '220177_s_at', '203423_at', '225016_at', '229242_at', '232531_at', '212622_at', '223843_at', '40665_at', '210906_x_at', '230378_at', '229281_at', '229177_at', '222996_s_at', '219778_at', '209437_s_at', '212909_at', '209835_x_at', '206754_s_at', '227314_at', '206496_at', '212151_at']