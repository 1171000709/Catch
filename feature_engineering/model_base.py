import xgboost
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,\
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression,Lasso
from sklearn.svm import LinearSVC, LinearSVR


class ModelBase(object):
    def __init__(self):
        self.n_estimators = 10
        self.random_state = 0

    @staticmethod
    def xgb_classify():
        # model = XGBClassifier(max_depth=6, learning_rate="0.1",
        #                       n_estimators=500, verbosity=0,
        #                       use_label_encoder=False)
        # model = XGBClassifier(max_depth=6, learning_rate="0.1",
        #                       n_estimators=200, verbosity=0,
        #                       use_label_encoder=False)
        # model = XGBClassifier(max_depth=6, learning_rate="0.1",
        #                       n_estimators=200, verbosity=0, subsample=0.8,
        #                       colsample_bytree=0.8, use_label_encoder=False)
        model = XGBClassifier(random_state=0,verbosity=0,objective='binary:logistic', n_jobs=-1)
        return model

    @staticmethod
    def xgb_regression():
        model = XGBRegressor(max_depth=6, learning_rate="0.1",
                              n_estimators=600, verbosity=0, subsample=0.8,
                              colsample_bytree=0.8, use_label_encoder=False, scale_pos_weight=1, n_jobs=-1)
        return model

    @staticmethod
    def rf_classify():
        model = RandomForestClassifier(random_state=0, n_estimators=10)
        # model = RandomForestClassifier(n_estimators=600,
        #                                max_depth=8,
        #                                random_state=0,
        #                                class_weight='balanced')
        return model

    @staticmethod
    def rf_regeression():
        model = RandomForestRegressor(random_state=0, n_estimators=10)
        # model = RandomForestRegressor(n_estimators=20,
        #                               max_depth=6,
        #                               random_state=0)
        return model

    @staticmethod
    def lr_classify():
        model = LogisticRegression(penalty='l2', solver='liblinear',
                                   class_weight='balanced', n_jobs=-1,
                                   tol=0.0005, C=0.3, max_iter=10000,
                                   random_state=42)
        # model = LogisticRegression(class_weight='balanced', n_jobs=1,
        #                            tol=0.0005, C=0.5, max_iter=10000)
        return model

    @staticmethod
    def lr_regression():
        # model = LinearRegression()
        model = Lasso(tol=0.0005, max_iter=10000, random_state=42,alpha=0.1)
        return model

    @staticmethod
    def svm_liner_svc():
        model = LinearSVC(random_state=42)
        return model

    @staticmethod
    def svm_liner_svr():
        model = LinearSVR(random_state=42)
        return model

