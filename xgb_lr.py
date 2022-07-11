# -*- coding: utf-8 -*-
import pandas as pd
from feature_engineering.model_base import ModelBase
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import ks_2samp
from sklearn.feature_selection import SelectFromModel
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression

def xgb_classify(param):
    model = XGBClassifier(**param)
    return model
def lr_classify_penalty(param):
    model = LogisticRegression(**param)
    return model

get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic

class MixModel(object):
    '''Now Just for binary predict'''
    def __init__(self, x,y,c_indexs,d_indexs,score_type = 'f1'):
        self.x = x
        self.y = y
        self.c_indexs = c_indexs
        self.d_indexs = d_indexs
        self.score_type = self._get_score_type(score_type)
        assert self.score_type in ['f1','auc','ks'],'Only f1_score,roc_auc_score and Ks_score supported'
        # self.sample_type = self._get_sample_type(y)
        self.enc_old,self.enc_new = self._get_encoder()
        self.bin_boundry = {}
        # self.mode = self._get_mode(mode)
        self.gbdt_params = None
        self.gbdt_model = None
        self.linear_model = None
        self.select_model = None

    def _get_encoder(self):
        enc_old_feature = OneHotEncoder(handle_unknown='ignore')
        enc_new_feature = OneHotEncoder()
        return enc_old_feature,enc_new_feature

    def _get_score_type(self,score_type):
        score_type = score_type if score_type == 'f1' else 'auc'
        assert score_type in ['auc', 'f1', 'ks'], 'Now score type only support f1_score,ks and auc'
        return score_type

    def _get_xy(self,c_cols,d_cols,label):
        return np.hstack((c_cols, d_cols)), label

    def fit_search(self,test_data,gbdt_fit_params,linear_fit_params=None):
        x_train,y_train = self.x,self.y
        x_test, y_test = test_data
        eval_set = [(x_test, y_test)]
        gbdt_fit_params['eval_set'] = eval_set
        self.gbdt_model.fit(x_train,y_train,**gbdt_fit_params)
        new_feature_train = self.gbdt_model.apply(x_train)
        x_train_new, x_train_old = self.rebuild_feature(x_train, new_feature_train)
        # x_train_new = self.rebuild_feature_tree(x_train, y_train,new_feature_train)
        print(x_train_new.shape)
        self.linear_model.fit(x_train_new, y_train,**linear_fit_params)
        return self

    def fit_valid(self,**params):
        pass

    def feature_select(self,X,y):
        param = {'penalty':'l1', 'solver': 'liblinear'}
        linear_model = lr_classify_penalty(param) # use lasso reg to select features
        linear_model.fit(X,y)
        self.select_model = SelectFromModel(linear_model,prefit=True)
        return self.select_model.transform(X)

    def rebuild_feature(self, x_train,new_feature_train):
        self.enc_new.fit(new_feature_train)
        new_feature_train_enc = self.enc_new.fit_transform(new_feature_train).toarray()
        c_feature_shape = len(self.c_indexs) if isinstance(self.c_indexs,list) else 0
        d_feature_shape = len(self.d_indexs) if isinstance(self.d_indexs, list) else 0
        if d_feature_shape != 0:
            self.enc_old.fit(x_train[:, c_feature_shape:])
            old_train_d_enc = self.enc_old.fit_transform(x_train[:, c_feature_shape:]).toarray()
            #
            old_train_c = x_train[:, :c_feature_shape]
            x_train_new = np.hstack((old_train_c, old_train_d_enc, new_feature_train_enc))
            x_train_old = np.hstack((old_train_c, old_train_d_enc))
        else:
            old_train_c = x_train[:, :c_feature_shape]
            x_train_new = np.hstack((old_train_c, new_feature_train_enc))
            x_train_old = old_train_c
        return x_train_new, x_train_old

    @staticmethod
    def calculate_weight(label):
        return (len(label) - sum(label)) / sum(label)

    def transform(self,x_test):
        new_test_enc = self.enc_new.transform(self.gbdt_model.apply(x_test)).toarray()
        x_test_enc = None
        if len(self.c_indexs) > 0:
            c_features = x_test[:, self.c_indexs]
            x_test_enc = c_features

        if len(self.d_indexs) > 0:
            d_features = x_test[:,self.d_indexs]
            old_test_enc = self.enc_old.transform(d_features).toarray()
            if x_test_enc is None:
                x_test_enc = old_test_enc
            else:
                x_test_enc = np.hstack((x_test_enc, old_test_enc))
        x_test_enc = np.hstack((x_test_enc,new_test_enc))
        # assert self.select_model is not None,'Select model has not trained,check it'
        # x_test_enc = self.select_model.transform(x_test_enc)
        return x_test_enc


    def predict(self,x_test):
        if not isinstance(x_test,np.ndarray):
            raise ValueError('X must be numpy.ndarray')
        x_test_enc = self.transform(x_test)
        # x_test_enc = self.transform_tree(x_test)
        return self.linear_model.predict(x_test_enc)

    def predict_proba(self,x_test):
        if not isinstance(x_test,np.ndarray):
            raise ValueError('X must be numpy.ndarray')
        x_test_enc = self.transform(x_test)
        # x_test_enc = self.transform_tree(x_test)
        return self.linear_model.predict_proba(x_test_enc)

    def evaluate(self,test_data,score_type):
        x_val,y_val = test_data
        y_predict = self.predict(x_val)
        y_p = self.predict_proba(x_val)
        if score_type == 'f1':
            return f1_score(y_val,y_predict)
        elif score_type == 'auc':
            return roc_auc_score(y_val,y_p[:,1])
        elif score_type == 'ks':
            return get_ks(y_p[:,1],y_val)
        else:
            raise ValueError('Score type not supported')

    #
    def prob_calibration(self,y_prob,rat):
        pass

    def __delete__(self, instance):
        del instance


class XGB_LR(MixModel):
    def __init__(self, x, y, c_indexs, d_indexs, xgb_params,score_type='f1'):
        super(XGB_LR, self).__init__(x, y, c_indexs, d_indexs, score_type)
        self.gbdt_params = xgb_params
        self.gbdt_params['scale_pos_weight'] = self.calculate_weight(y)
        self.gbdt_params['eval_metric'] = 'logloss' if self.score_type == 'f1' else 'auc'
        # self.gbdt_model = xgb_classify_mix(self.gbdt_params)
        self.gbdt_model = xgb_classify(self.gbdt_params)#ModelBase.
        self.linear_model = ModelBase.lr_classify()

    def fit_valid(self,gbdt_fit_params,linear_fit_params = None):
        x_train,y_train = self.x,self.y
        self.gbdt_model.fit(x_train,y_train,**gbdt_fit_params)
        new_feature_train = self.gbdt_model.apply(x_train)
        x_train_new,_ = self.rebuild_feature(x_train,new_feature_train)
        # x_train_new = self.rebuild_feature_tree(x_train, y_train,new_feature_train)
        print(x_train_new.shape)
        # x_train_new = self.feature_select(x_train_new,y_train)
        # print(x_train_new.shape)
        self.linear_model.fit(x_train_new, y_train,**linear_fit_params)
        return self

    def __delete__(self, instance):
        super(XGB_LR,self).__delete__(instance)


if __name__ == '__main__':
    xgb_params = {
        'max_depth': 1,
        'learning_rate': 0.05,
        'n_estimator': 500,
        'use_label_encoder': False,
        'verbosity': 0,
        'scale_pos_weight': None,
        'reg_lambda': 0.05,
        'min_child_weight': 3
    }
    train_df = pd.read_csv(r'D:\work\project\Gitee\new_pan\AutoFE\train_df.csv')
    test_df = pd.read_csv(r'D:\work\project\Gitee\new_pan\AutoFE\test_df.csv')
    c_indexs = list(range(24))
    d_indexs = list(range(25,97))

    x = train_df.iloc[:, :-1].values
    y = train_df.iloc[:,-1].values
    x_val= test_df.iloc[:, :-1].values
    y_val = test_df.iloc[:,-1].values

    xgblr = XGB_LR(x, y, c_indexs, d_indexs, xgb_params, score_type='f1')

    gbdt_fit_params = {
        'eval_metric' : 'auc'
    }
    linear_fit_params = {}
    model = xgblr.fit_valid(gbdt_fit_params,linear_fit_params)

    y_predict = model.predict(x_val)
    print(f1_score(y_val, y_predict))

