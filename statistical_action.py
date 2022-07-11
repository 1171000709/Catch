# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def trans_sta():
    path = r'D:\work\project\Gitee\Auto-ML\double_rnn\res.xlsx'

    # df = pd.read_excel(path,sheet_name='_RF')
    df = pd.read_excel(path,sheet_name='_LR')
    # ['Dataset', 'Source', 'C/R', 'rf_action_600epoch']
    all_concat_action = []

    # ['square', 'inverse', 'log', 'sqrt', 'sigmoid', 'tanh']

    # ['add', 'subtract', 'multiply', 'divide']

    for i in range(df.shape[0]):
        # action_str = df.loc[i,'rf_action_600epoch']
        action_str = df.loc[i, 'lr_action_600epoch']
        dataname = df.loc[i,'Dataset']
        all_concat = []
        all_action = []
        square_list = []
        inverse_list = []
        log_list = []
        sqrt_list = []
        sigmoid_list = []
        tanh_list = []
        add_list = []
        subtract_list = []
        multiply_list = []
        divide_list = []
        action_list = eval(action_str)
        for order_action in action_list:
            for action in order_action:
                all_action.append(action)
                if action[-1] == 'concat':
                    all_concat.append(action)
                if action[-2] == 'inverse':
                    inverse_list.append(action)
                if action[-2] == 'log':
                    log_list.append(action)
                if action[-2] == 'sqrt':
                    sqrt_list.append(action)
                if action[-2] == 'sigmoid':
                    sigmoid_list.append(action)
                if action[-2] == 'tanh':
                    tanh_list.append(action)
                if action[-2] == 'add':
                    add_list.append(action)
                if action[-2] == 'subtract':
                    subtract_list.append(action)
                if action[-2] == 'multiply':
                    multiply_list.append(action)
                if action[-2] == 'divide':
                    divide_list.append(action)
        all_concat_action.append(all_concat)
        df.loc[df.Dataset == dataname, 'allaction_num'] = len(all_action)
        df.loc[df.Dataset==dataname,'concat_num'] = len(all_concat)
        df.loc[df.Dataset==dataname, 'square_num'] = len(square_list)
        df.loc[df.Dataset==dataname, 'inverse_num'] = len(inverse_list)
        df.loc[df.Dataset==dataname, 'log_num'] = len(log_list)
        df.loc[df.Dataset==dataname, 'sqrt_num'] = len(sqrt_list)
        df.loc[df.Dataset==dataname, 'sigmoid_num'] = len(sigmoid_list)
        df.loc[df.Dataset==dataname, 'tanh_num'] = len(tanh_list)
        df.loc[df.Dataset==dataname, 'add_num'] = len(add_list)
        df.loc[df.Dataset==dataname, 'subtract_num'] = len(subtract_list)
        df.loc[df.Dataset==dataname, 'multiply_num'] = len(multiply_list)
        df.loc[df.Dataset==dataname, 'divide_num'] = len(divide_list)


    # for ls in [all_concat_action,
    # square_list
    # ,inverse_list
    # ,log_list
    # ,sqrt_list
    # ,sigmoid_list
    # ,tanh_list
    # ,add_list
    # ,subtract_list
    # ,multiply_list
    # ,divide_list]:
    #     print(len(ls))

    # df.to_excel(r'D:\work\project\Gitee\Auto-ML\double_rnn\res_RF.xlsx',index=False)
    df.to_excel(r'D:\work\project\Gitee\Auto-ML\double_rnn\res_LR.xlsx',index=False)
def replace_concat_sta():
    path = r'D:\work\project\Gitee\Auto-ML\double_rnn\res.xlsx'

    df = pd.read_excel(path,sheet_name='action_num_rate')
    # df = pd.read_excel(path,sheet_name='_LR')
    # ['Dataset', 'Source', 'C/R', 'rf_action_600epoch']
    all_concat_action = []

    # ['square', 'inverse', 'log', 'sqrt', 'sigmoid', 'tanh']

    # ['add', 'subtract', 'multiply', 'divide']
    res = pd.DataFrame()
    res['Features'] = df.Features
    for col in list(df):
        if 'cycle' in col:
            for i in range(df.shape[0]):
                action_str = df.loc[i,col]
                # action_str = df.loc[i, 'lr_action_600epoch']
                dataname = df.loc[i,'Dataset']
                all_concat = []
                all_action = []
                all_replace = []

                action_list = eval(action_str)
                for order_action in action_list:
                    for action in order_action:
                        all_action.append(action)
                        if action[-1] == 'concat':
                            all_concat.append(action)
                        if action[-1] == 'replace':
                            all_replace.append(action)
                # print(i, dataname, len(all_concat))
                df.loc[df.Dataset == dataname, 'allaction_num'] = len(all_action)
                df.loc[df.Dataset == dataname, 'allconcat_num'] = len(all_concat)
                df.loc[df.Dataset == dataname, 'allreplace_num'] = len(all_replace)
                np.sum(df.allconcat_num / df.Features)
            res[col + 'allaction_num'] = df.allaction_num
            res[col + 'allconcat_num'] = df.allconcat_num
            res[col + 'allreplace_num'] = df.allreplace_num
    return res



def cycle_order_plot():
    path = r'D:\work\project\Gitee\Auto-ML\double_rnn\结果对比.xlsx'
    df = pd.read_excel(path, sheet_name='cycle1-6')
    order_mean = df.groupby('C/R').mean().reset_index()
    all_mean = pd.DataFrame(['All data sets'] + list(df.mean(axis=0)),index=list(order_mean)).T
    res = order_mean.append(all_mean).reset_index(drop=True)
    for col in list(res):
        if 'cycle' in col:
            res[col] = res[col] - res['rf_Base']
    res['cycle0'] = 0
    for col in list(res):
        if 'cycle' in col:
            res[col] = res[col]*100

    import matplotlib.pyplot as plt
    # res.iloc[:,1:-1].plot()
    # res.show()
    res = res.drop(columns=['cycle6'])
    res = res.set_index(keys =res['C/R'],drop=True ).drop(columns=['C/R']).T
    res = res.sort_index()
    res = res.drop('rf_Base',axis=0)
    plt.figure(figsize=(8, 6))
    # plt.subplot(2,1,1)
    marker_list= ['o','x','D']
    for i,col in enumerate(list(res)):
        plt.plot(range(len(res[col])),res[col],linewidth=1.5,alpha=0.7,marker=marker_list[i])
    # plt.title(re.findall(r'search_(.*)', log_path_name)[0])
    plt.legend(['regression','classification','All data sets'],loc=0)
    plt.xlabel('Order of Transformations')
    plt.ylabel('Average Improvement of F1-score/1-rae(%)')
    plt.ylim()
    plt.xticks()

    # plt.subplot(2, 1, 2)
    # for col in list(infer):
    #     plt.plot(infer.index.tolist(),infer[col],linewidth=3)
    # plt.legend(list(infer), loc=0)
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\comp-hight-order.png')

    # plt.imsave(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\comp-hight-order.png')

    return df

def concat_rate_plot():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    df = pd.read_excel(path, sheet_name='concat_rate')
    df = df.set_index('cate')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    # plt.subplot(2,1,1)
    marker_list= ['*','^','o','x','D','+']
    for i,col in enumerate(list(df)):
        plt.plot(range(len(df[col])),df[col],linewidth=1.5,alpha=0.7,marker=marker_list[i],linestyle='--')
    # plt.title(re.findall(r'search_(.*)', log_path_name)[0])
    plt.legend(list(df),loc=0)
    plt.xlabel('Order of the New Feature Scale')
    plt.ylabel('Proportion of New Features to Original Features')
    plt.ylim()
    plt.xticks()

    # plt.subplot(2, 1, 2)
    # for col in list(infer):
    #     plt.plot(infer.index.tolist(),infer[col],linewidth=3)
    # plt.legend(list(infer), loc=0)
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\new_fe_ratio.png')

def plot_merge():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    df = pd.read_excel(path, sheet_name='Filter')
    df = df.set_index('Task')
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # # plt.subplot(2,1,1)
    # marker_list= ['*','^','o','x','D','+']
    # for i,col in enumerate(list(df)):
    #     plt.plot(range(len(df[col])),df[col],linewidth=1.5,alpha=0.7,marker=marker_list[i],linestyle='--')
    # # plt.title(re.findall(r'search_(.*)', log_path_name)[0])
    # plt.legend(list(df),loc=0)
    # plt.xlabel('Order of the New Feature Scale')
    # plt.ylabel('Proportion of New Features to Original Features')
    # plt.ylim()
    # plt.xticks()
    size = 10

    x = np.arange(df.shape[0])
    total_width, n = 0.8, 3  #
    width = total_width / n
    x = x - (total_width - width) / 2
    color_list = ['darkorange','royalblue','darkcyan','lightseagreen','coral','darkblue','darkgreen','gold'] #("green")
    for i,col in enumerate(list(df)):
        plt.bar(x+ width*i, df[col], width=width
                , label=col, color=color_list[i],alpha=1)
    # plt.bar(x + width, y2, width=width, label='label2', color='deepskyblue')
    # plt.bar(x + 2 * width, y3, width=width, label='label3', color='green')

    # plt.subplot(2, 1, 2)
    # for col in list(infer):
    #     plt.plot(infer.index.tolist(),infer[col],linewidth=3)
    # plt.xlim(0,2)
    # plt.xticks(list(df.index))

    plt.xticks([0,1,2],list(df.index))
    plt.legend(list(df), loc=0)
    # plt.xlabel('Data for Different Tasks')
    plt.ylabel('Average Improvement of F1-score/1-rae(%)')
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\filter.png')

def plot_trans():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    df = pd.read_excel(path, sheet_name='action_comp')
    df = df.set_index('trans')
    #
    import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  #
    import numpy as np
    plt.figure(figsize=(6, 7))
    x = np.arange(df.shape[0])  #
    y1 = df['RF']
    y2 = df['LR']

    bar_width = 0.35  #
    tick_label = list(df.index)

    # 'darkorange','royalblue'
    plt.barh(x, y1, bar_width, color='darkorange', label='RF')
    plt.barh(x + bar_width, y2, bar_width, color='royalblue', label='LR')#darkcyan

    plt.legend(loc=4)  #
    plt.yticks(x + bar_width / 2, tick_label)  #

    plt.xlabel('Percentage of Practical Actions(%)')
    # plt.ylabel('Average Improvement of F1-score/1-rae(%)')
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\trans.png')

def plot_merge_():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    df = pd.read_excel(path, sheet_name='Filter')
    df = df.set_index('Task')
    import matplotlib.pyplot as plt

    x = np.arange(df.shape[0])
    total_width, n = 0.8, 3  #
    width = total_width / n
    x = x - (total_width - width) / 2
    color_list = ['darkorange','royalblue','darkcyan','lightseagreen','coral','darkblue','darkgreen','gold'] #("green")
    for i,col in enumerate(list(df)):
        plt.bar(x+ width*i, df[col], width=width
                , label=col, color=color_list[i],alpha=1)
    # plt.bar(x + width, y2, width=width, label='label2', color='deepskyblue')
    # plt.bar(x + 2 * width, y3, width=width, label='label3', color='green')

    # plt.subplot(2, 1, 2)
    # for col in list(infer):
    #     plt.plot(infer.index.tolist(),infer[col],linewidth=3)
    # plt.xlim(0,2)
    # plt.xticks(list(df.index))

    plt.xticks([0,1,2],list(df.index))
    plt.legend(list(df), loc=0)
    # plt.xlabel('Data for Different Tasks')
    plt.ylabel('Average Improvement of F1-score/1-rae(%)')
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\filter.png')

def plot_baseline_impv():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    df = pd.read_excel(path, sheet_name='from_0')
    # df = df.set_index('log_baseline')
    import matplotlib.pyplot as plt
    for i,col in enumerate(list(df)):
        if 'avg' not in col:
            plt.plot(range(len(df[col])), df[col], linewidth=0.3, alpha=0.7)
            pass
        else:
            plt.plot(range(len(df[col])), df[col], linewidth=3, alpha=1)

    plt.show()


def plot_episodemax_impv():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    # df = pd.read_excel(path, sheet_name='DIFER_BASE_LOG_AVG')
    df = pd.read_excel(path, sheet_name='DIFER_BASE_LOG-BASE')
    # df = df.set_index('log_baseline')
    import matplotlib.pyplot as plt
    leg = []
    plt.figure(figsize=(8, 6))
    for i,col in enumerate(list(df)):
        # plt.plot(range(len(df[col])), df[col], linewidth=1, alpha=0.7)
        if col not in ['Regression','Classification','All data sets']:
            plt.plot(range(len(df[col])), df[col], linewidth=0.3, alpha=0.7)
            # leg.append(None)
        #     pass
        else:
            plt.plot(range(len(df[col])), df[col], linewidth=2, alpha=1,label=col)
            # leg.append(col)
    plt.legend(loc=0)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Improvement of F1-score/1-rae(%)')
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\epoch.png')

def cycle1_5_plot():
    path = r'D:\download\AuthorKit22\AuthorKit22\LaTeX\Table.xlsx'
    df = pd.read_excel(path, sheet_name='cycle1-5')
    df = df.set_index('cate')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))

    marker_list = ['o', 'x', 'D']
    for i, col in enumerate(list(df)):
        plt.plot(range(len(df[col])), df[col], linewidth=1.5, alpha=0.7, marker=marker_list[i])
    plt.legend(list(df), loc=0)
    plt.xlabel('Order of Transformations')
    plt.ylabel('Average Improvement of F1-score/1-rae(%)')
    plt.ylim()
    plt.xticks()

    # plt.subplot(2, 1, 2)
    # for col in list(infer):
    #     plt.plot(infer.index.tolist(),infer[col],linewidth=3)
    # plt.legend(list(infer), loc=0)
    # plt.show()
    plt.savefig(r'D:\download\AuthorKit22\AuthorKit22\LaTeX\comp-hight-order.png')


if __name__ == '__main__':
    # cycle_order_plot()
    # df = replace_concat_sta()
    # df.to_excel(r'D:\work\project\Gitee\Auto-ML\double_rnn\cycle_concat_rate.xlsx',index=False)

    # concat_rate_plot()
    plot_merge()
    # plot_trans()
    # plot_episodemax_impv()
    # cycle1_5_plot()