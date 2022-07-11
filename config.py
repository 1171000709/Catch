
dataset_config = dict()

# 格式示例
dataset_config['adult_dataset'] = {
    'dataset_path': 'data/adult.csv',
    'continuous_col': ['age', 'fnlwgt', 'earnings', 'loss', 'hour',
                       'edu_nums'],
    'discrete_col': ['work_cate', 'education', 'marital',
                     'profession', 'relation', 'race', 'gender',
                     'motherland'],
    'target_col': 'label',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    # 'eval_method': 'ks'
    # 'eval_method': 'auc'
}

dataset_config['p_hzd'] = {
    'dataset_path': 'data/p_hzd.csv',
    'continuous_col': ['age', 'avg_sixmonth_amount', 'service_years',
                       'avg_sixmonth_frequency', 'avg_sixmonth_daily_deposit',
                       'contract_rmb', 'contract_interest_rate', 'contract_deadline',
                       'mortgage_discount_rate',
                       'debit_card_total_amount',
                       'avg_sixmonth_debit_card_use_amount',
                       'external_guarantors_amount',
                       'financing_amount',
                       'avg_sixmonth_should_repayment',
                       'debit_card_amount_use_rate'],
    'discrete_col': ['marital_status', 'is_vip', 'is_Ebank',
                     'is_financial', 'num_medium',
                     'guarantee_type', 'produce_name',
                     'external_guarantors_num',
                     'financing_bank_num', 'financing_num',
                     'guarantee_type_second'],
    'target_col': 'label',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['pre_data_hzd'] = {
    'dataset_path': 'data/pre_data_hzd.csv',
    'continuous_col': ['年龄', '我行服务年数', '贷前6个月月均贷方发生额', '贷前6个月月均贷方发生笔数', '贷款日前6个月存款日均',
                       '合同金额折人民币', '合同利率', '合同期限', '抵押折扣率', '对外担保金额', '融资总额', '贷款最近6个月平均应还款',
                       '贷记卡额度占用率', '对外担保人数', '融资银行家数', '融资笔数'],
    'discrete_col': ['婚姻状况代码', '是否贵宾卡客户', '是否网银客户', '是否理财客户', '介质数量',
                     '主担保方式名称', '产品名称', '担保类型'],
    'target_col': 'rst',
    'task_type': 'classifier',
    # 'eval_method': 'ks',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['hzd_fillna'] = {
    'dataset_path': 'data/hzd_fillna.csv',
    'continuous_col': ['年龄', '我行服务年数', '贷前6个月月均贷方发生额', '贷前6个月月均贷方发生笔数', '贷款日前6个月存款日均',
                       '合同金额折人民币', '合同利率', '合同期限', '抵押折扣率', '对外担保金额', '融资总额', '贷款最近6个月平均应还款',
                       '贷记卡额度占用率', '对外担保人数', '融资银行家数', '融资笔数'],
    'discrete_col': ['婚姻状况代码', '是否贵宾卡客户', '是否网银客户', '是否理财客户', '介质数量',
                     '主担保方式名称', '产品名称', '担保类型'],
    'target_col': 'rst',
    'task_type': 'classifier',
    # 'eval_method': 'auc',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

#
# dataset_config['hzd_amend'] = {
#     'dataset_path': 'data/hzd_amend.csv',
#     'continuous_col': ['年龄', '我行服务年数', '贷款日前6个月存款日均', '合同金额折人民币',
#                        ''],
#     'discrete_col': ['产品名称', '主担保方式名称'],
#     'target_col': 'rst',
#     'task_type': 'classifier',
#     'eval_method': 'auc',
#     # 'eval_method': 'ks',
#     # 'eval_method': 'f1_score',
# }


dataset_config['hzd_amend'] = {
    'dataset_path': 'data/hzd_mend.csv',
    'continuous_col': ['年龄', '贷前6个月月均贷方发生额', '贷前6个月月均贷方发生笔数',
                       '贷款日前6个月存款日均', '合同金额折人民币', '合同利率', '抵押折扣率',
                       '我行服务年数', '合同期限'],
    'discrete_col': ['婚姻状况代码', '是否贵宾卡客户', '是否网银客户', '是否理财客户',
                     '介质数量', '主担保方式名称', '产品名称', '担保类型'],
    'target_col': 'rst',
    'task_type': 'classifier',
    #'eval_method': 'auc',
    # 'eval_method': 'ks',
     'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['poker_hand'] = {
        'dataset_path': 'data/poker_hand.csv',
        'continuous_col': [],
        'discrete_col': ['S1','C1','S2','C2','S3','C3','S4','C4','S5', 'C5'],
        'target_col': 'CLASS',
        'task_type': 'classifier',
        'eval_method': 'f1_score',
        'f1_average': 'micro'
}
dataset_config['gisette'] = {
        'dataset_path': 'data/gisette.csv',
        'continuous_col': [],
        'discrete_col': [],
        'target_col': 'label',
        'task_type': 'classifier',
        'eval_method': 'f1_score',
        'f1_average': 'micro'
}


#
dataset_config['default_credit_card'] = {
    'dataset_path': 'data/default_credit_card.csv',
    'task_type': 'classifier',
    'target_col': 'target',
    'continuous_col': ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2',
                       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                       'PAY_AMT5', 'PAY_AMT6'],
    'discrete_col': ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
                     'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

#
dataset_config['dry_bean_dataset'] = {
    'dataset_path': 'data/Dry_Bean_Dataset.csv',
    'task_type': 'classifier',
    'target_col': 'Class',
    'continuous_col': ['Area', 'Perimeter', 'MajorAxisLength',
                       'MinorAxisLength', 'AspectRation', 'Eccentricity',
                       'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
                       'roundness', 'Compactness', 'ShapeFactor1',
                       'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['hour_dataset'] = {
    'dataset_path': 'data/hour.csv',
    'task_type': 'regression',
    'target_col': 'cnt',
    'continuous_col': ['windspeed', 'temp', 'atemp', 'hum', 'hr'],
    'discrete_col': ['season', 'yr', 'mnth',  'holiday', 'weekday',
                     'workingday', 'weathersit'],
    'eval_method': 'r_squared',
}

dataset_config['water_potability'] = {
    'dataset_path': 'data/water_potability.csv',
    'task_type': 'classifier',
    'target_col': 'Potability',
    'continuous_col': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                       'Conductivity', 'Organic_carbon', 'Trihalomethanes',
                       'Turbidity'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['titanic'] = {
    'dataset_path': 'data/titanic.csv',
    'continuous_col': ['Age', 'Fare'],
    'discrete_col': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
    'target_col': 'Survived',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['Melbourne_housing_FULL'] = {
    'dataset_path': 'data/titanic.csv',
    'continuous_col': ['Age', 'Fare'],
    'discrete_col': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
    'target_col': 'Survived',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['winequality_red'] = {
    'dataset_path': 'data/winequality_red.csv',
    'continuous_col': ['fixed acidity', 'volatile acidity', 'citric acid',
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates',
                       'alcohol'],
    'discrete_col': [],
    'target_col': 'quality',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['club_loan'] = {
    'dataset_path': 'data/club_loan.csv',
    'continuous_col': ['int.rate', 'installment', 'log.annual.inc', 'dti',
                       'fico', 'days.with.cr.line', 'revol.bal', 'revol.util'],
    'discrete_col': ['credit.policy', 'purpose', 'inq.last.6mths',
                     'delinq.2yrs', 'pub.rec'],
    'target_col': 'not.fully.paid',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['creditcard'] = {
    'dataset_path': 'data/creditcard.csv',
    'continuous_col': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
                       'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
                       'V26', 'V27', 'V28', 'Amount'],
    'discrete_col': [],
    'target_col': 'Class',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'

}


dataset_config['house_prices'] = {
    'dataset_path': 'data/house_prices.csv',
    'continuous_col': ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                       'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'],
    'discrete_col': ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                     'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                     'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
                     'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition'],
    'target_col': 'SalePrice',
    'task_type': 'regression',
    'eval_method': 'sub_rae'
}

dataset_config['lymphography'] = {
    'dataset_path': 'data/lymphography.csv',
    'continuous_col': ['V11', 'V12'],
    'discrete_col': ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V13','V14','V15','V16','V17','V18'],
    'target_col': 'class',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['house_prices_filter'] = {
    'dataset_path': 'data/house_prices_filter.csv',
    'continuous_col': ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1',
                       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MiscVal', 'YrSold'],
    'discrete_col': ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle',
                     'RoofStyle', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
                     'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition'],
    'target_col': 'SalePrice',
    'task_type': 'regression',
    'eval_method': 'mae'
}

dataset_config['amazon_employee'] = {
    'dataset_path': 'data/amazon_employee.csv',
    'continuous_col': [],
    'discrete_col': ['RESOURCE','MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY','ROLE_CODE'],
    'target_col': 'ACTION',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
                   'f1_average': 'micro'
}

dataset_config['svmguide3'] = {
    'dataset_path': 'data/svmguide3.csv',
    'continuous_col': ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v'],
    'discrete_col': [],
    'target_col': 'label',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['bankrupt'] = {
    'dataset_path': 'data/bankrupt.csv',
    'continuous_col': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
                       'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
                       'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37',
                       'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49',
                       'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61',
                       'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73',
                       'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85',
                       'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93'],
    'discrete_col': ['C1'],
    'target_col': 'label',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['bank_add'] = {
    'dataset_path': 'data/bank_add.csv',
    'continuous_col': ['emp.var.rate', 'cons.price.idx',	'cons.conf.idx', 'euribor3m', 'nr.employed', 'age', 'duration'],
    'discrete_col': ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign',
                     'pdays', 'previous', 'poutcome'],
    'target_col': 'y',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    #'eval_method': 'auc'
    'f1_average': 'micro'
}

dataset_config['AP_Omentum_Ovary'] = {
    'dataset_path': 'data/AP_Omentum_Ovary.csv',
    'continuous_col': ['205913_at', '207175_at', '1555778_a_at', '213247_at', '220988_s_at', '212344_at', '229849_at', '216442_x_at', '212419_at', '219873_at', '229479_at', '204589_at', '201744_s_at', '209763_at', '203824_at', '227566_at', '209612_s_at', '201125_s_at', '200788_s_at', '218468_s_at', '235978_at', '209242_at', '201149_s_at', '203980_at', '214505_s_at', '204548_at', '209581_at', '220102_at', '225242_s_at', '209090_s_at', '227061_at', '235733_at', '201117_s_at', '223122_s_at', '225241_at', '201150_s_at', '213125_at', '225424_at', '219087_at', '212354_at', '225987_at', '240135_x_at', '37892_at', '212587_s_at', '205941_s_at', '221730_at', '212488_at', '225681_at', '210072_at', '202273_at'],
    'discrete_col': [],
    'target_col': 'Tissue',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['credit_default'] = {
    'dataset_path': 'data/credit_default.csv',
    'continuous_col': ['ID','LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'],
    'discrete_col': ['SEX','EDUCATION','MARRIAGE'],
    'target_col': 'default payment next month',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['credit_dataset'] = {
    'dataset_path': 'data/credit_dataset.csv',
    'continuous_col': ['YEARS_EMPLOYED', 'BEGIN_MONTH', 'AGE', 'INCOME'],
    'discrete_col': ['GENDER', 'CAR', 'REALITY', 'NO_OF_CHILD', 'INCOME_TYPE', 'EDUCATION_TYPE', 'FAMILY_TYPE',
                     'HOUSE_TYPE', 'WORK_PHONE', 'PHONE', 'E_MAIL', 'FAMILY SIZE'],
    'target_col': 'TARGET',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['credit_dataset1688'] = {
    'dataset_path': 'data/credit_dataset1688.csv',
    'continuous_col': ['YEARS_EMPLOYED', 'BEGIN_MONTH', 'AGE', 'INCOME'],
    'discrete_col': ['GENDER', 'CAR', 'REALITY', 'NO_OF_CHILD', 'INCOME_TYPE', 'EDUCATION_TYPE', 'FAMILY_TYPE',
                     'HOUSE_TYPE', 'WORK_PHONE', 'PHONE', 'E_MAIL', 'FAMILY SIZE'],
    'target_col': 'TARGET',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['water_potability_drop'] = {
    'dataset_path': 'data/water_potability.csv',
    'task_type': 'classifier',
    'target_col': 'Potability',
    'continuous_col': ['ph', 'Hardness', 'Sulfate',
                       'Trihalomethanes',
                       'Turbidity'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['credit_dataset1688_drop'] = {
    'dataset_path': 'data/credit_dataset1688.csv',
    'continuous_col': ['YEARS_EMPLOYED', 'BEGIN_MONTH', 'AGE', 'INCOME'],
    'discrete_col': ['GENDER',
                     #'CAR','REALITY','NO_OF_CHILD',
                     #'INCOME_TYPE','EDUCATION_TYPE',
                     'FAMILY_TYPE',
                     #'HOUSE_TYPE','WORK_PHONE','PHONE',
                     'E_MAIL'
                     #'FAMILY SIZE'
                     ],
    'target_col': 'TARGET',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['titanic_drop'] = {
    'dataset_path': 'data/titanic.csv',
    'continuous_col': [#'Age',
        'Fare'],
    'discrete_col': ['Pclass', 'Sex',
                     #'SibSp',
                     'Parch', 'Embarked'],
    'target_col': 'Survived',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['winequality_red_3'] = {
    'dataset_path': 'data/winequality_red_3.csv',
    'continuous_col': ['fixed acidity', 'volatile acidity', 'citric acid',
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates',
                       'alcohol'],
    'discrete_col': [],
    'target_col': 'quality',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['lemur_data'] = {
    'dataset_path': 'data/lemur_data.csv',
    'continuous_col': ['AgeMax_LiveOrDead_y', 'Weight_g', 'MonthOfWeight', 'AgeAtWt_d', 'AgeAtWt_wk',
                       'AgeAtWt_mo', 'AgeAtWt_mo_NoDec', 'AgeAtWt_y', 'Change_Since_PrevWt_g', 'Days_Since_PrevWt',
                       'Avg_Daily_WtChange_g', 'R_Min_Dam_AgeAtConcep_y'],
    'discrete_col': ['Taxon', 'Hybrid', 'Sex', 'Name', 'Current_Resident', 'StudBook', 'Birth_Month', 'Birth_Type',
                     'Birth_Institution', 'Expected_Gestation', 'Concep_Month', 'Dam_ID', 'Dam_Name', 'Dam_Taxon',
                     'Sire_ID', 'Age_Category'],
    'target_col': 'Preg_Status',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['Korea_Income'] = {
    'dataset_path': 'data/Korea_Income.csv',
    'continuous_col': [],
    'discrete_col': ["year", "wave", "region", "family_member", "gender",'year_born','education_level',
                     'marriage','religion','occupation','company_size','reason_none_worker'],
    'target_col': 'income',
    'task_type': 'regression',
    'eval_method': 'mae'
}

dataset_config['Placement_Data_Full_Class'] = {
    'dataset_path': 'data/Placement_Data_Full_Class.csv',
    'continuous_col': ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'],
    'discrete_col': ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t",'workex','specialisation'],
    'target_col': 'status',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['heart'] = {
    'dataset_path': 'data/heart.csv',
    'continuous_col': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
    'discrete_col': ["sex", "cp", "fbs", "restecg", "exang",'slope','ca', 'thal'],
    'target_col': 'target',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['airline_passenger_satisfaction'] = {
    'dataset_path': 'data/airline_passenger_satisfaction.csv',
    'continuous_col': ['age', 'flight_distance', 'departure_delay_in_minutes', 'arrival_delay_in_minutes'],
    'discrete_col': ['Gender', 'customer_type', 'type_of_travel', 'customer_class', 'inflight_wifi_service', 'departure_arrival_time_convenient',
                     'ease_of_online_booking', 'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort', 'inflight_entertainment',
                     'onboard_service', 'leg_room_service', 'baggage_handling', 'checkin_service', 'inflight_service', 'cleanliness'],
    'target_col': 'satisfaction',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['aug_train'] = {
    'dataset_path': 'data/aug_train.csv',
    'continuous_col': ['city_development_index', 'training_hours'],
    'discrete_col': ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline',
                     'experience', 'company_size', 'company_type', 'last_new_job'],
    'target_col': 'target',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['mobile_pricerange_train'] = {
    'dataset_path': 'data/mobile_pricerange_train.csv',
    'continuous_col': ['battery_power', 'clock_speed','int_memory', 'm_dep', 'mobile_wt', 'px_height',
                       'px_width', 'ram','fc', 'pc', 'sc_h', 'sc_w'],
    'discrete_col': ['blue', 'dual_sim', 'four_g', 'n_cores', 'talk_time', 'three_g', 'touch_screen', 'wifi'],
    'target_col': 'price_range',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['healthcare-dataset-stroke-data'] = {
    'dataset_path': 'data/healthcare-dataset-stroke-data.csv',
    'continuous_col': ['age', 'avg_glucose_level' ,'bmi'],
    'discrete_col': ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
    'target_col': 'stroke',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['Tamil_Nadu_Elections'] = {
    'dataset_path': 'data/Tamil_Nadu_State_Elections_2021_Details.csv',
    'continuous_col': ['EVM_Votes', 'Postal_Votes', 'Total_Votes', '%_of_Votes', 'Tot_Constituency_votes_polled',
                       'Tot_votes_by_parties','Winning_votes'],
    'discrete_col': ['Constituency', 'Candidate', 'Party'],
    'target_col': 'Win_Lost_Flag',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['Customer_Segmentation'] = {
    'dataset_path': 'data/Customer_Segmentation.csv',
    'continuous_col': ['Age', 'Work_Experience'],
    'discrete_col': ['Gender', 'Ever_Married', 'Graduated', 'Profession','Spending_Score','Family_Size', 'Var_1'],
    'target_col': 'Segmentation',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['recon'] = {
    'dataset_path': 'data/recon.csv',
    'continuous_col': ['年龄', '贷前6个月月均贷方发生额', '贷前6个月月均贷方发生笔数',
                       '贷款日前6个月存款日均', '合同金额折人民币', '合同利率', '抵押折扣率',
                       '我行服务年数', '合同期限'],
    'discrete_col': ['婚姻状况代码', '是否贵宾卡客户', '是否网银客户', '是否理财客户',
                     '介质数量', '主担保方式名称', '产品名称', '担保类型'],
    'target_col': 'rst',
    'task_type': 'classifier',
    'eval_method': 'auc',
    # 'eval_method': 'ks',
    # 'eval_method': 'f1_score',
}

# dataset_config['lymphography'] = {
#     'dataset_path': 'data/NFS_datasets/lymphography.csv',
#     'task_type': 'classifier',
#     'target_col': 'label',
#     'continuous_col': [],
#     'discrete_col': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17'],
#     'eval_method': 'f1_score',
# }

dataset_config['spectf'] = {
    'dataset_path': 'data/spectf.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['spectf_80'] = {
    'dataset_path': 'data/spectf_80.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['winequality_white'] = {
    'dataset_path': 'data/winequality_white.csv',
    'continuous_col': ['fixed acidity', 'volatile acidity', 'citric acid',
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates',
                       'alcohol'],
    'discrete_col': [],
    'target_col': 'quality',
    'task_type': 'classifier',
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['credit_a'] = {
    'dataset_path': 'data/credit_a.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'],
    'discrete_col': ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['credit_a_p'] = {
    'dataset_path': 'data/credit_a_p.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'],
    'discrete_col': ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['ionosphere'] = {
    'dataset_path': 'data/ionosphere.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
                       'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33'],
    'discrete_col':['V0', 'V1'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['messidor_features'] = {
    'dataset_path': 'data/messidor_features.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                       'C13', 'C14', 'C15', 'C16'],
    'discrete_col': ['D1','D2','D3'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['PimaIndian'] = {
    'dataset_path': 'data/PimaIndian.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['C0', 'C1', 'C2','C3', 'C4', 'C5', 'C6', 'C7'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['spambase'] = {
    'dataset_path': 'data/spambase.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
                       'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
                       'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
                       'V51', 'V52', 'V53', 'V54', 'V55', 'V56'],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['hepatitis'] = {
    'dataset_path': 'data/hepatitis.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['V0', 'V13', 'V14', 'V15', 'V16', 'V17'],
    'discrete_col': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V18'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['megawatt1'] = {
    'dataset_path': 'data/megawatt1.csv',
    'task_type': 'classifier',
    'target_col': 'def',
    'continuous_col': ['a', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
       'l', 'm', 'n', 'p', 's', 't', 'u', 'v', 'z',
       'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj',
       'ak', 'al', 'am', 'an', 'ao'],
    'discrete_col': ['d', 'o', 'r'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['fertility_Diagnosis'] = {
    'dataset_path': 'data/fertility_Diagnosis.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['V1', 'V6', 'V8'],
    'discrete_col':['V0', 'V2', 'V3', 'V4', 'V5', 'V7'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['Bikeshare_DC'] = {
    'dataset_path': 'data/Bikeshare_DC.csv',
    'task_type': 'regression',
    'target_col': 'count',
    # 'time_col': ['datetime'],
    'continuous_col': ['temp','atemp', 'humidity', 'windspeed', 'casual', 'registered'],
    'discrete_col': ['season', 'holiday', 'workingday', 'weather'],
    'eval_method': 'sub_rae',
}

dataset_config['BNG_wisconsin'] = {
    'dataset_path': 'data/BNG_wisconsin.csv',
    'task_type': 'regression',
    'target_col': 'time',
    # 'time_col': ['datetime'],
    'continuous_col': [],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}

dataset_config['airlines_train'] = {
    'dataset_path': 'data/airlines_train.csv',
    'task_type': 'regression',
    'target_col': 'Distance',
    # 'time_col': ['datetime'],
    'continuous_col': ['DepDelay', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime'],
    'discrete_col': ['UniqueCarrier', 'Origin', 'Dest'],
    'eval_method': 'sub_rae',
}


dataset_config['BNG_wine_quality'] = {
    'dataset_path': 'data/BNG_wine_quality.csv',
    'task_type': 'regression',
    'target_col': 'quality',
    # 'time_col': ['datetime'],
    'continuous_col': [],
    'discrete_col': [],
    'eval_method': 'mae',
}

dataset_config['covtype'] = {
    'dataset_path': 'data/covtype.csv',
    'task_type': 'classifier',
    'target_col': 's55',
    'continuous_col': [],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}
dataset_config['covtype_small'] = {
    'dataset_path': 'data/covtype_small.csv',
    'task_type': 'classifier',
    'target_col': 's55',
    'continuous_col': [],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['accelerometer'] = {
    'dataset_path': 'data/accelerometer.csv',
    'task_type': 'classifier',
    'target_col': 'wconfid',
    'continuous_col': ['x','y','z'],
    'discrete_col': ['pctid'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['test'] = {
    'dataset_path': 'data/test.csv',
    'task_type': 'classifier',
    'target_col': 'rst',
    'continuous_col': [],
    'discrete_col': [],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['sepsis_survival'] = {
    'dataset_path': 'data/sepsis.csv',
    'task_type': 'classifier',
    'target_col': 'hospital_outcome_1alive_0dead',
    'continuous_col': ['age_years','episode_number'],
    'discrete_col': ['sex_0male_1female'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}

dataset_config['name'] = {
    'dataset_path': 'data/name_gender_dataset.csv',
    'task_type': 'regression',
    'target_col': 'Probability',
    'continuous_col': ['Count'],
    'discrete_col': ['Gender'],
    'eval_method': 'sub_rae',
}

dataset_config['PetFinder'] = {
    'dataset_path': 'data/PetFinder.csv',
    'task_type': 'regression',
    'target_col': 'Pawpularity',
    'continuous_col': [],
    'discrete_col': ['Id','Subject Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur'],
    'eval_method': 'sub_rae',
}



dataset_config['airfoil'] = {
    'dataset_path': 'data/airfoil.csv',
    'task_type': 'regression',
    'target_col': 'label',
    'continuous_col': ['V0', 'V1', 'V2', 'V3', 'V4'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}
dataset_config['mark'] = {
    'dataset_path': 'data/mark.csv',
    'task_type': 'regression',
    'target_col': 'target',
    'discrete_col': [],
    'eval_method': 'sub_rae',
    'continuous_col': ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_58', 'f_59', 'f_60', 'f_61', 'f_62', 'f_63', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79', 'f_80', 'f_81', 'f_82', 'f_83', 'f_84', 'f_85', 'f_86', 'f_87', 'f_88', 'f_89', 'f_90', 'f_91', 'f_92', 'f_93', 'f_94', 'f_95', 'f_96', 'f_97', 'f_98', 'f_99', 'f_100', 'f_101', 'f_102', 'f_103', 'f_104', 'f_105', 'f_106', 'f_107', 'f_108', 'f_109', 'f_110', 'f_111', 'f_112', 'f_113', 'f_114', 'f_115', 'f_116', 'f_117', 'f_118', 'f_119', 'f_120', 'f_121', 'f_122', 'f_123', 'f_124', 'f_125', 'f_126', 'f_127', 'f_128', 'f_129', 'f_130', 'f_131', 'f_132', 'f_133', 'f_134', 'f_135', 'f_136', 'f_137', 'f_138', 'f_139', 'f_140', 'f_141', 'f_142', 'f_143', 'f_144', 'f_145', 'f_146', 'f_147', 'f_148', 'f_149', 'f_150', 'f_151', 'f_152', 'f_153', 'f_154', 'f_155', 'f_156', 'f_157', 'f_158', 'f_159', 'f_160', 'f_161', 'f_162', 'f_163', 'f_164', 'f_165', 'f_166', 'f_167', 'f_168', 'f_169', 'f_170', 'f_171', 'f_172', 'f_173', 'f_174', 'f_175', 'f_176', 'f_177', 'f_178', 'f_179', 'f_180', 'f_181', 'f_182', 'f_183', 'f_184', 'f_185', 'f_186', 'f_187', 'f_188', 'f_189', 'f_190', 'f_191', 'f_192', 'f_193', 'f_194', 'f_195', 'f_196', 'f_197', 'f_198', 'f_199', 'f_200', 'f_201', 'f_202', 'f_203', 'f_204', 'f_205', 'f_206', 'f_207', 'f_208', 'f_209', 'f_210', 'f_211', 'f_212', 'f_213', 'f_214', 'f_215', 'f_216', 'f_217', 'f_218', 'f_219', 'f_220', 'f_221', 'f_222', 'f_223', 'f_224', 'f_225', 'f_226', 'f_227', 'f_228', 'f_229', 'f_230', 'f_231', 'f_232', 'f_233', 'f_234', 'f_235', 'f_236', 'f_237', 'f_238', 'f_239', 'f_240', 'f_241', 'f_242', 'f_243', 'f_244', 'f_245', 'f_246', 'f_247', 'f_248', 'f_249', 'f_250', 'f_251', 'f_252', 'f_253', 'f_254', 'f_255', 'f_256', 'f_257', 'f_258', 'f_259', 'f_260', 'f_261', 'f_262', 'f_263', 'f_264', 'f_265', 'f_266', 'f_267', 'f_268', 'f_269', 'f_270', 'f_271', 'f_272', 'f_273', 'f_274', 'f_275', 'f_276', 'f_277', 'f_278', 'f_279', 'f_280', 'f_281', 'f_282', 'f_283', 'f_284', 'f_285', 'f_286', 'f_287', 'f_288', 'f_289', 'f_290', 'f_291', 'f_292', 'f_293', 'f_294', 'f_295', 'f_296', 'f_297', 'f_298', 'f_299']
}


dataset_config['Housing_Boston'] = {
    'dataset_path': 'data/Housing_Boston.csv',
    'task_type': 'regression',
    'target_col': 'label',
    'continuous_col': ['V0', 'V1', 'V2', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12'],
    'discrete_col': ['V3'],
    'eval_method': 'sub_rae',
}

dataset_config['BNG_cpu_act'] = {
    'dataset_path': 'data/BNG_cpu_act.csv',
    'task_type': 'regression',
    'target_col': 'usr',
    'continuous_col': [],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}

dataset_config['Openml_586'] = {
    'dataset_path': 'data/Openml_586.csv',
    'task_type': 'regression',
    'target_col': 'oz26',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9', 'oz10',
                       'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17', 'oz18', 'oz19',
                       'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}

dataset_config['Openml_589'] = {
    'dataset_path': 'data/Openml_589.csv',
    'task_type': 'regression',
    'target_col': 'oz26',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9', 'oz10',
                       'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17', 'oz18', 'oz19',
                       'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}


dataset_config['Openml_607'] = {
    'dataset_path': 'data/Openml_607.csv',
    'task_type': 'regression',
    'target_col': 'oz51',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8',
                       'oz9', 'oz10', 'oz11', 'oz12', 'oz13', 'oz14', 'oz15',
                       'oz16', 'oz17', 'oz18', 'oz19', 'oz20', 'oz21', 'oz22',
                       'oz23', 'oz24', 'oz25', 'oz26', 'oz27', 'oz28', 'oz29',
                       'oz30', 'oz31', 'oz32', 'oz33', 'oz34', 'oz35', 'oz36',
                       'oz37', 'oz38', 'oz39', 'oz40', 'oz41', 'oz42', 'oz43',
                       'oz44', 'oz45', 'oz46', 'oz47', 'oz48', 'oz49', 'oz50',
                       ],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}


dataset_config['Openml_616'] = {
    'dataset_path': 'data/Openml_616.csv',
    'task_type': 'regression',
    'target_col': 'oz51',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9', 'oz10',
                       'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17', 'oz18', 'oz19',
                       'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25', 'oz26', 'oz27', 'oz28',
                       'oz29', 'oz30', 'oz31', 'oz32', 'oz33', 'oz34', 'oz35', 'oz36', 'oz37',
                       'oz38', 'oz39', 'oz40', 'oz41', 'oz42', 'oz43', 'oz44', 'oz45', 'oz46',
                       'oz47', 'oz48', 'oz49', 'oz50'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}

dataset_config['Openml_618'] = {
    'dataset_path': 'data/Openml_618.csv',
    'task_type': 'regression',
    'target_col': 'oz51',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9',
                       'oz10', 'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17',
                       'oz18', 'oz19', 'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25',
                       'oz26', 'oz27', 'oz28', 'oz29', 'oz30', 'oz31', 'oz32', 'oz33',
                       'oz34', 'oz35', 'oz36', 'oz37', 'oz38', 'oz39', 'oz40', 'oz41',
                       'oz42', 'oz43', 'oz44', 'oz45', 'oz46', 'oz47', 'oz48', 'oz49', 'oz50'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}


dataset_config['Openml_620'] = {
    'dataset_path': 'data/Openml_620.csv',
    'task_type': 'regression',
    'target_col': 'oz26',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9', 'oz10', 'oz11',
                       'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17', 'oz18', 'oz19', 'oz20', 'oz21',
                       'oz22', 'oz23', 'oz24', 'oz25'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}


dataset_config['Openml_637'] = {
    'dataset_path': 'data/Openml_637.csv',
    'task_type': 'regression',
    'target_col': 'oz51',
    'continuous_col': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9', 'oz10', 'oz11',
                       'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17', 'oz18', 'oz19', 'oz20', 'oz21',
                       'oz22', 'oz23', 'oz24', 'oz25', 'oz26', 'oz27', 'oz28', 'oz29', 'oz30', 'oz31',
                       'oz32', 'oz33', 'oz34', 'oz35', 'oz36', 'oz37', 'oz38', 'oz39', 'oz40', 'oz41',
                       'oz42', 'oz43', 'oz44', 'oz45', 'oz46', 'oz47', 'oz48', 'oz49', 'oz50'],
    'discrete_col': [],
    'eval_method': 'sub_rae',
}

dataset_config['german_credit_24'] = {
    'dataset_path': 'data/german_credit_24.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['C0','C1','C2'],
    'discrete_col': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                     'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20'],
    'eval_method': 'f1_score',
    'f1_average': 'micro'
}


dataset_config['chb'] = {
    'dataset_path': 'data/chb.csv',
    'task_type': 'classifier',
    'target_col': 'label',
    'continuous_col': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579', '580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659', '660', '661', '662', '663', '664', '665', '666', '667', '668', '669', '670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683', '684', '685', '686', '687', '688', '689', '690', '691', '692', '693', '694', '695', '696', '697', '698', '699', '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739', '740', '741', '742', '743', '744', '745', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '780', '781', '782', '783', '784', '785', '786', '787', '788', '789', '790', '791', '792', '793', '794', '795', '796', '797', '798', '799', '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '970', '971', '972', '973', '974', '975', '976', '977', '978', '979', '980', '981', '982', '983', '984', '985', '986', '987', '988', '989', '990', '991', '992', '993', '994', '995', '996', '997', '998', '999', '1000', '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012', '1013', '1014', '1015', '1016', '1017', '1018', '1019', '1020', '1021', '1022', '1023', '1024', '1025', '1026', '1027', '1028', '1029', '1030', '1031', '1032', '1033', '1034', '1035', '1036', '1037', '1038', '1039', '1040', '1041', '1042', '1043', '1044', '1045', '1046', '1047', '1048', '1049', '1050', '1051', '1052', '1053', '1054', '1055', '1056', '1057', '1058', '1059', '1060', '1061', '1062', '1063', '1064', '1065', '1066', '1067', '1068', '1069', '1070', '1071', '1072', '1073', '1074', '1075', '1076', '1077', '1078', '1079', '1080', '1081', '1082', '1083', '1084', '1085', '1086', '1087', '1088', '1089', '1090', '1091', '1092', '1093', '1094', '1095', '1096', '1097', '1098', '1099', '1100', '1101', '1102', '1103', '1104', '1105', '1106', '1107', '1108', '1109', '1110', '1111', '1112', '1113', '1114', '1115', '1116', '1117', '1118', '1119', '1120', '1121', '1122', '1123', '1124', '1125', '1126', '1127', '1128', '1129', '1130', '1131', '1132', '1133', '1134', '1135', '1136', '1137', '1138', '1139', '1140', '1141', '1142', '1143', '1144', '1145', '1146', '1147', '1148', '1149', '1150', '1151', '1152', '1153', '1154', '1155', '1156', '1157', '1158', '1159', '1160', '1161', '1162', '1163', '1164', '1165', '1166', '1167', '1168', '1169', '1170', '1171', '1172', '1173', '1174', '1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184', '1185', '1186', '1187', '1188', '1189', '1190', '1191', '1192', '1193', '1194', '1195', '1196', '1197', '1198', '1199', '1200', '1201', '1202', '1203', '1204', '1205', '1206', '1207', '1208', '1209', '1210', '1211', '1212', '1213', '1214', '1215', '1216', '1217', '1218', '1219', '1220', '1221', '1222', '1223', '1224', '1225', '1226', '1227', '1228', '1229', '1230', '1231', '1232', '1233', '1234', '1235', '1236', '1237', '1238', '1239', '1240', '1241', '1242', '1243', '1244', '1245', '1246', '1247', '1248', '1249', '1250', '1251', '1252', '1253', '1254', '1255', '1256', '1257', '1258', '1259', '1260', '1261', '1262', '1263', '1264', '1265', '1266', '1267', '1268', '1269', '1270', '1271', '1272', '1273', '1274', '1275', '1276', '1277', '1278', '1279', '1280', '1281', '1282', '1283', '1284', '1285', '1286', '1287', '1288', '1289', '1290', '1291', '1292', '1293', '1294', '1295', '1296', '1297', '1298', '1299', '1300', '1301', '1302', '1303', '1304', '1305', '1306', '1307', '1308', '1309', '1310', '1311', '1312', '1313', '1314', '1315', '1316', '1317', '1318', '1319', '1320', '1321', '1322', '1323', '1324', '1325', '1326', '1327', '1328', '1329', '1330', '1331', '1332', '1333', '1334', '1335', '1336', '1337', '1338', '1339', '1340', '1341', '1342', '1343', '1344', '1345', '1346', '1347', '1348', '1349', '1350', '1351', '1352', '1353', '1354', '1355', '1356', '1357', '1358', '1359', '1360', '1361', '1362', '1363', '1364', '1365', '1366', '1367', '1368', '1369', '1370', '1371', '1372', '1373', '1374', '1375', '1376', '1377', '1378', '1379', '1380', '1381', '1382', '1383', '1384', '1385', '1386', '1387', '1388', '1389', '1390', '1391', '1392', '1393', '1394', '1395', '1396', '1397', '1398', '1399', '1400', '1401', '1402', '1403', '1404', '1405', '1406', '1407', '1408', '1409', '1410', '1411', '1412', '1413', '1414', '1415', '1416', '1417', '1418', '1419', '1420', '1421', '1422', '1423', '1424', '1425', '1426', '1427', '1428', '1429', '1430', '1431', '1432', '1433', '1434', '1435', '1436', '1437', '1438', '1439', '1440', '1441', '1442', '1443', '1444', '1445', '1446', '1447', '1448', '1449', '1450', '1451', '1452', '1453', '1454', '1455', '1456', '1457', '1458', '1459', '1460', '1461', '1462', '1463', '1464', '1465', '1466', '1467', '1468', '1469', '1470', '1471', '1472', '1473', '1474', '1475', '1476', '1477', '1478', '1479', '1480', '1481', '1482', '1483', '1484', '1485', '1486', '1487', '1488', '1489', '1490', '1491', '1492', '1493', '1494', '1495', '1496', '1497', '1498', '1499', '1500', '1501', '1502', '1503', '1504', '1505', '1506', '1507', '1508', '1509', '1510', '1511', '1512', '1513', '1514', '1515', '1516', '1517', '1518', '1519', '1520', '1521', '1522', '1523', '1524', '1525', '1526', '1527', '1528', '1529', '1530', '1531', '1532', '1533', '1534', '1535', '1536', '1537', '1538', '1539', '1540', '1541', '1542', '1543', '1544', '1545', '1546', '1547', '1548', '1549', '1550', '1551', '1552', '1553', '1554', '1555', '1556', '1557', '1558', '1559', '1560', '1561', '1562', '1563', '1564', '1565', '1566', '1567', '1568', '1569', '1570', '1571', '1572', '1573', '1574', '1575', '1576', '1577', '1578', '1579', '1580', '1581', '1582', '1583', '1584', '1585', '1586', '1587', '1588', '1589', '1590', '1591', '1592', '1593', '1594', '1595', '1596', '1597', '1598', '1599', '1600', '1601', '1602', '1603', '1604', '1605', '1606', '1607', '1608', '1609', '1610', '1611', '1612', '1613', '1614', '1615', '1616', '1617', '1618', '1619', '1620', '1621', '1622', '1623', '1624', '1625', '1626', '1627', '1628', '1629', '1630', '1631', '1632', '1633', '1634', '1635', '1636', '1637', '1638', '1639', '1640', '1641', '1642', '1643', '1644', '1645', '1646', '1647', '1648', '1649', '1650', '1651', '1652', '1653', '1654', '1655', '1656', '1657', '1658', '1659', '1660', '1661', '1662', '1663', '1664', '1665', '1666', '1667', '1668', '1669', '1670', '1671', '1672', '1673', '1674', '1675', '1676', '1677', '1678', '1679', '1680', '1681', '1682', '1683', '1684', '1685', '1686', '1687', '1688', '1689', '1690', '1691', '1692', '1693', '1694', '1695', '1696', '1697', '1698', '1699', '1700', '1701', '1702', '1703', '1704', '1705', '1706', '1707', '1708', '1709', '1710', '1711', '1712', '1713', '1714', '1715', '1716', '1717', '1718', '1719', '1720', '1721', '1722', '1723', '1724', '1725', '1726', '1727', '1728', '1729', '1730', '1731', '1732', '1733', '1734', '1735', '1736', '1737', '1738', '1739', '1740', '1741', '1742', '1743', '1744', '1745', '1746', '1747', '1748', '1749', '1750', '1751', '1752', '1753', '1754', '1755', '1756', '1757', '1758', '1759', '1760', '1761', '1762', '1763', '1764', '1765', '1766', '1767', '1768', '1769', '1770', '1771', '1772', '1773', '1774', '1775', '1776', '1777', '1778', '1779', '1780', '1781', '1782', '1783', '1784', '1785', '1786', '1787', '1788', '1789', '1790', '1791', '1792', '1793', '1794', '1795', '1796', '1797', '1798', '1799', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '1808', '1809', '1810', '1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819', '1820', '1821', '1822', '1823', '1824', '1825', '1826', '1827', '1828', '1829', '1830', '1831', '1832', '1833', '1834', '1835', '1836', '1837', '1838', '1839', '1840', '1841', '1842', '1843', '1844', '1845', '1846', '1847', '1848', '1849', '1850', '1851', '1852', '1853', '1854', '1855', '1856', '1857', '1858', '1859', '1860', '1861', '1862', '1863', '1864', '1865', '1866', '1867', '1868', '1869', '1870', '1871', '1872', '1873', '1874', '1875', '1876', '1877', '1878', '1879', '1880', '1881', '1882', '1883', '1884', '1885', '1886', '1887', '1888', '1889', '1890', '1891', '1892', '1893', '1894', '1895', '1896', '1897', '1898', '1899', '1900', '1901', '1902', '1903', '1904', '1905', '1906', '1907', '1908', '1909', '1910', '1911', '1912', '1913', '1914', '1915', '1916', '1917', '1918', '1919', '1920', '1921', '1922', '1923', '1924', '1925', '1926', '1927', '1928', '1929', '1930', '1931', '1932', '1933', '1934', '1935', '1936', '1937', '1938', '1939', '1940', '1941', '1942', '1943', '1944', '1945', '1946', '1947', '1948', '1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059', '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077', '2078', '2079', '2080', '2081', '2082', '2083', '2084', '2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099', '2100', '2101', '2102', '2103', '2104', '2105', '2106', '2107', '2108', '2109', '2110', '2111', '2112', '2113', '2114', '2115', '2116', '2117', '2118', '2119', '2120', '2121', '2122', '2123', '2124', '2125', '2126', '2127', '2128', '2129', '2130', '2131', '2132', '2133', '2134', '2135', '2136', '2137', '2138', '2139', '2140', '2141', '2142', '2143', '2144', '2145', '2146', '2147', '2148', '2149', '2150', '2151', '2152', '2153', '2154', '2155', '2156', '2157', '2158', '2159', '2160', '2161', '2162', '2163', '2164', '2165', '2166', '2167', '2168', '2169', '2170', '2171', '2172', '2173', '2174', '2175', '2176', '2177', '2178', '2179', '2180', '2181', '2182', '2183', '2184', '2185', '2186', '2187', '2188', '2189', '2190', '2191', '2192', '2193', '2194', '2195', '2196', '2197', '2198', '2199', '2200', '2201', '2202', '2203', '2204', '2205', '2206', '2207', '2208', '2209', '2210', '2211', '2212', '2213', '2214', '2215', '2216', '2217', '2218', '2219', '2220', '2221', '2222', '2223', '2224', '2225', '2226', '2227', '2228', '2229', '2230', '2231', '2232', '2233', '2234', '2235', '2236', '2237', '2238', '2239', '2240', '2241', '2242', '2243', '2244', '2245', '2246', '2247', '2248', '2249', '2250', '2251', '2252', '2253', '2254', '2255', '2256', '2257', '2258', '2259', '2260', '2261', '2262', '2263', '2264', '2265', '2266', '2267', '2268', '2269', '2270', '2271', '2272', '2273', '2274', '2275', '2276', '2277', '2278', '2279', '2280', '2281', '2282', '2283', '2284', '2285', '2286', '2287', '2288', '2289', '2290', '2291', '2292', '2293'],
    'discrete_col': [],
    'eval_method': 'confusion_matrix',
    # 'f1_average': 'micro'
}


dataset_config['financial_train'] = {
    'dataset_path': 'data/financial_train.csv',
    'task_type': 'classifier',
    'target_col': 'return_bin',
    'continuous_col': ['EP', 'BP', 'SP', 'CFP', 'financial_leverage', 'debtequityratio', 'cashratio', 'currentratio', 'NI', 'GPM', 'ROE', 'ROA', 'asset_turnover',
                       'net_operating_cash_flow', 'Sales_G_q', 'Profit_G_q', 'RSI', 'BIAS', 'PSY', 'DIF', 'DEA', 'MACD', 'AR', 'ARBR', 'ATR14', 'VOL5', 'VOL60',
                       'Skewness20', 'Skewness60'],
    'discrete_col': [],
    'eval_method': 'auc',
}


