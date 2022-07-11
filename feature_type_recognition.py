import pandas as pd
import datetime

from CONST import FEATURE_TYPE


def detect_TIMESTAMP(df, col):
    try:
        ts_min = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].min()))
        ts_max = int(float(df.loc[~(df[col] == '') & (df[col].notnull()), col].max()))
        datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M:%S')
        datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M:%S')
        if datetime_min > '2000-01-01 00:00:01' and datetime_max < '2030-01-01 00:00:01' and datetime_max > datetime_min:
            return True
    except:
        return False

def detect_DATETIME(df, col):
    is_DATETIME = False
    if df[col].dtypes == object or str(df[col].dtypes) == 'category':
        is_DATETIME = True
        try:
            pd.to_datetime(df[col])
        except:
            is_DATETIME = False
    return is_DATETIME

def get_data_type(df, col):
    if detect_DATETIME(df, col):
        return FEATURE_TYPE['datetime']
    if detect_TIMESTAMP(df, col):
        return FEATURE_TYPE['timestamp']
    if df[col].dtypes == object or df[col].dtypes == bool or str(df[col].dtypes) == 'category':
        if df[col].fillna(df[col].mode()).apply(lambda x: len(str(x))).astype('float').mean() > 25:
            return FEATURE_TYPE['txt']
        return FEATURE_TYPE['cat']
    if 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
        return FEATURE_TYPE['num']

class Feature_type_recognition():
    def __init__(self):
        self.df = None
        self.feature_type = None

    def fit(self, df):
        self.df = df
        self.feature_type = {}
        self.num = []
        self.cat = []
        self.txt = []
        self.ord = []
        self.datatime = []
        self.timestamp = []
        for col in self.df.columns:
            cur_type = get_data_type(self.df, col)
            self.feature_type[col] = cur_type
            if (cur_type == 'num'):
                self.num.append(col)
            elif (cur_type == 'cat'):
                self.cat.append(col)
            elif (cur_type == 'txt'):
                self.txt.append(col)
            elif (cur_type == 'ord'):
                self.ord.append(col)
            elif (cur_type == 'datatime'):
                self.datatime.append(col)
            elif (cur_type == 'timestamp'):
                self.timestamp.append(col)
        return self.feature_type
