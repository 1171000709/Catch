from boruta import BorutaPy
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score
from pypinyin import lazy_pinyin

# from sklearn.ensemble import RandomForestClassifier
#
#
# model = RandomForestClassifier(random_state=42)
# print(model.get_params())
#
# hp_param = {'n_estimators': 100, 'max_depth': 20, 'max_features': 'log2',
#             'min_samples_split': 8, 'min_samples_leaf': 2, 'bootstrap': True}
#
# model.set_params(**hp_param)
#
# print(model.get_params())
# raw_df = pd.DataFrame(pd.read_csv('data/Bikeshare_DC.csv'))
# print(raw_df.shape)
#
# aa = raw_df.iloc[:, 1:]
# print(aa.corr())

# aa = [0.0559, 0.0547, 0.0553, 0.0581, 0.0536, 0.0558, 0.0543, 0.0549, 0.0563,
#       0.0561, 0.0543, 0.0578, 0.0556, 0.0535, 0.0569, 0.0554, 0.0555, 0.0561]
#
# zz = sum(aa)
# print(zz)

zz = 23
bb = 25

cc = max(zz, bb)
print('cc', cc)
"""
zz = raw_df.duplicated()
print(zz.value_counts())
# 
raw_df.drop_duplicates(keep='first', inplace=True)
raw_df.reset_index(drop=True, inplace=True)
print(raw_df.shape)
zz = raw_df.duplicated()
print(zz.value_counts())
raw_df.to_csv('data/hzd_mend.csv', index=False)

# import pstats
# p = pstats.Stats('AutoFE_NAS2.pstat')
# p.sort_stats('cumulative')
# p.print_stats(1000)

# import pickle
# from pathlib import Path
#
#
# def get_pkl(workers_ins_path_name):
#     workers_ins_path = \
#         Path('train_log') / workers_ins_path_name / Path('workers_ins.pkl')
#     workers_ins_f = open(workers_ins_path, 'rb')
#     workers_ins = pickle.load(workers_ins_f)
#     print(workers_ins[0])
#     print(len(workers_ins))
#
#
# if __name__ == '__main__':
#     workers_path_name = 'search_pre_data_hzd_20210707-211329'
#     get_pkl(workers_path_name)



# data = pd.read_csv('data/pre_data_hzd.csv', encoding='gbk')
# print(data.columns)
# # zh_col = name_list = lazy_pinyin(data.columns)
# for i in data.columns:
#     print(lazy_pinyin(i))

# data_enl = pd.read_csv('data/hzd_encol.csv')
# print(len(data.columns))
# print(len(data.columns))
#
# data.columns = data_enl.columns
#
# data['is_financial'] = data['is_financial'].fillna(0)
# data['financing_bank_num'] = data['financing_bank_num'].fillna(0)
# data['financing_num'] = data['financing_num'].fillna(0)
# data['financing_amount'] = data['financing_amount'].fillna(0)
# data['avg_sixmonth_should_repayment'] = \
#     data['avg_sixmonth_should_repayment'].fillna(0)
#
# data['debit_card_total_amount'] = data['debit_card_total_amount'].fillna(0)
# data['avg_sixmonth_debit_card_use_amount'] = \
#     data['avg_sixmonth_debit_card_use_amount'].fillna(0)
#
# data['external_guarantors_num'] = data['external_guarantors_num'].fillna(0)
# data['external_guarantors_amount'] = data['external_guarantors_amount'].fillna(0)
# data['debit_card_amount_use_rate'] = data['debit_card_amount_use_rate'].fillna(0)
#
#
#
# num = data.isna().sum(axis=1)
# # print(data.shape)
# data['queshi'] = num
# print('num', num.value_counts())
# print('zzzz', data.shape)
# data = data[data['queshi'] <= 0]
#
# # data.reset_index(inplace=True)
# print('zzzz', data.shape)
# data.to_csv('process_hzd.csv', index=False)
#
# # 
# sum_xx = data.isna().sum(axis=0)
# print(sum_xx)
# print(data.shape)
#
# # data.reset_index(inplace=True)
# data = data.drop('queshi', axis=1)
# data.to_csv('process_hzd.csv', index=False)

# print(num.value_counts())

# zz = data.isnull().sum(axis=1)
# print('zz', zz)




# col = data_enl.columns
# print(col)

# data = pd.read_csv('data/adult.csv')

#
#
# df_target = data['label']
# df_feature = data.drop('label', axis=1)
#
# for header in df_feature.columns.values:
#     if df_feature[header].dtype == "object":
#         oe = preprocessing.OrdinalEncoder()
#         trans_column = oe.fit_transform(df_feature[header].values.reshape(-1, 1))
#         df_feature[header] = trans_column.reshape(1, -1)[0]
#
# if df_target.dtype == "object":
#     le = preprocessing.LabelEncoder()
#     df_target = pd.Series(le.fit_transform(df_target), name=df_target.name)
#
#
#
#
# # print(y_data)
# # print(x_data)
#
# # 
# x_train, x_test, y_train, y_test = train_test_split(
#     df_feature, df_target, test_size=0.25,
#     shuffle=True, stratify=df_target, random_state=1)
#
#
# xgb1 = XGBClassifier(max_depth=7, learning_rate="0.1", n_estimators=600,
#                      use_label_encoder=False, verbosity=0)
#
# xgb1.fit(x_train.values, y_train.values)
# y_test_pred = xgb1.predict(x_test.values)
# raw_score = f1_score(y_test.values, y_test_pred, average='binary')
# print('raw_score', raw_score)
#
# xgb_select = XGBClassifier(max_depth=7, learning_rate="0.1", n_estimators=300,
#                            use_label_encoder=False, verbosity=0)
#
# fe_selector = BorutaPy(xgb_select, n_estimators='auto', two_step=True, verbose=2,
#                        random_state=2, max_iter=40)
#
# fe_selector.fit(x_train.values, y_train.values)
#
# print(x_train.columns[fe_selector.support_])
# print(fe_selector.ranking_)
# print(fe_selector.support_)
#
# X_train_filtered = fe_selector.transform(x_train.values)
# x_test_filtered = fe_selector.transform(x_test.values)
# print('X_train_filtered', X_train_filtered.shape)
# print('x_test_filtered', x_test_filtered.shape)
#
# x_train_selected = x_train.iloc[:, fe_selector.support_]
# x_test_selected = x_test.iloc[:, fe_selector.support_]
# print('x_train_selected', x_train_selected.values.shape)
# print('x_test_selected', x_test_selected.values.shape)
#
# xgb2 = XGBClassifier(max_depth=7, learning_rate="0.1", n_estimators=600,
#                      use_label_encoder=False, verbosity=0)
#
# xgb2.fit(x_train_selected, y_train)
# y_test_pred2 = xgb2.predict(x_test_selected)
#
# selected_score = f1_score(y_test, y_test_pred2, average='binary')
# print('selected_score', selected_score)
"""


