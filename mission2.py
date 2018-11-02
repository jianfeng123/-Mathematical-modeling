# Load in our libraries
import pandas as pd
import re
import copy
import random
from scipy import stats
import warnings
import math
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import norm
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib import keras
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import KFold
from datetime import timedelta
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, Birch,AgglomerativeClustering, FeatureAgglomeration, SpectralClustering
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes
# %%
train_data = pd.read_excel('data.xlsx')
### %%数据清洗 ###
train_data_b = copy.deepcopy(train_data)
text_1 = re.findall('\（(.*?)\）','大概日期（approxdate）、事件摘要（summary）、相关事件（related）、省/行政区域/州（provstate）、城市（city）、位置描述 （location）、武器细节（weapdetail）、实体名称（corp1）、具体目标/受害者（target1）、第二实体名称（corp2）、第二具体目标/受害者（target2）、第三实体名称（corp3）、第三具体目标/受害者（target3）、犯罪集团的名称（gname）、犯罪子集团名称（gsubname）、第二犯罪集团名称（gname2）、第二犯罪子集团名称（gsubname2）、第三犯罪集团名称（gname3）、第三犯罪子集团名称（gsubname3）、动机（motive）、财产损失评价（propcomment）、绑匪/劫匪转移到的国家（divert）、赎金笔记(ransomnote)、附加说明（addnotes）、第一引用来源（scite1）、第二引用来源（scite2）、第三引用来源（scite3）、数据收集（dbsource）')
text_1.remove('corp1')
text_1.remove('gname')
train_data_b.drop(text_1,inplace=True, axis=1)
text_2 = [tx for tx in train_data_b.columns if 'txt' in tx]
train_data_b.drop(text_2,inplace=True, axis=1)
train_b1 = copy.deepcopy(train_data_b)

def sample(train_X):
    train_len_GB = int(len(train_X) * 1)
    arr_A = random.sample(range(len(train_X)), train_len_GB)
    train_index = arr_A[:int(train_len_GB*0.99)]
    val_index = arr_A[int(train_len_GB*0.99):]
    ## sample
    train_X_BU = train_X.loc[train_index, :]
    val_X_BU = train_X.loc[val_index, :]
    ## produce
    train_Y0 = train_X_BU['gname'].values
    train_X0 = train_X_BU.drop('gname', axis=1)
    val_Y0 = val_X_BU['gname'].values
    val_X0 = val_X_BU.drop('gname', axis=1)
    return train_X0, train_Y0, val_X0, val_Y0
class Model(object):
    def evaluate(self, X_val, y_val):
        # assert(min(y_val) > 0)
        guessed_class = self.guess(X_val)
        class_p = np.argmax(guessed_class, axis=1)
        acc_cout = np.equal(class_p, y_val).astype(int)
        result = np.sum(acc_cout) / len(y_val)
        return result
class XGBoost(Model):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evallist = [(dtrain, 'train')]
        param = {
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'multi:softprob',
                 'num_class':6,
                 'colsample_bytree': 0.7,
                 'subsample': 0.7,
                'thread': -1
            }
        num_round = 500
        self.bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=True)
        # self.bst = xgb.train(param, dtrain, num_round)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return self.bst.predict(dtest)
train_b1.drop(['resolution', 'latitude', 'longitude'],inplace=True, axis=1)
## %地理编码特征：填众数
train_b1['specificity'] = train_b1['specificity'].fillna(train_b1['specificity'].mode()[0])
## %附近地区：未知填众数
train_b1['vicinity'] = train_b1['vicinity'].map(lambda t: train_b1['vicinity'].mode()[0] if t ==-9 else t)
## %怀疑恐怖组织处理
train_b1['doubtterr'] = train_b1['doubtterr'].fillna(train_b1['doubtterr'].mode()[0])
## %对其他进行处理，填0表示与已经存在的5类均没有关系
train_b1['alternative'] = train_b1['alternative'].fillna(0)
## %对是否是事件组处理
train_b1['multiple'] = train_b1['multiple'].fillna(train_b1['multiple'].mode()[0])
## %对攻击方式处理
train_b1.drop('attacktype2', inplace=True, axis=1)
train_b1.drop('attacktype3', inplace=True, axis=1)
train_b1.drop(['corp1','targsubtype1'], inplace=True, axis=1) ## 去除实体名
train_b1.drop(['targtype2', 'targsubtype2', 'natlty2', 'targtype3', 'targsubtype3', 'natlty3'], inplace=True, axis=1)
train_b1['guncertain1'] = train_b1['guncertain1'].fillna(train_b1['guncertain1'].mode()[0])
train_b2 = copy.deepcopy(train_b1)
train_b2.drop(['weaptype2', 'weapsubtype1', 'weapsubtype2', 'weaptype3', 'weaptype4', 'weapsubtype3', 'weapsubtype4'], inplace=True, axis=1)
## % 伤亡情况处理
train_b2['nkill'] = train_b2['nkill'].fillna(train_b2['nkill'].median())
train_b2['nkillter'] = train_b2['nkillter'].fillna(train_b2['nkillter'].median())
train_b2['nwound'] = train_b2['nwound'].fillna(train_b2['nwound'].median())
train_b2['nwoundte'] = train_b2['nwoundte'].fillna(train_b2['nwoundte'].median())
train_b2['nkill'] = train_b2['nkill'].map(lambda t: np.random.randint(low=800, high=1000) if t > 1000 else t)
train_b2['nkillter'] = train_b2['nkillter'].map(lambda t: np.random.randint(low=800, high=1000) if t > 1000 else t)
train_b2['nwound'] = train_b2['nwound'].map(lambda t: np.random.randint(low=800, high=1000) if t > 1000 else t)
train_b2['nwoundte'] = train_b2['nwoundte'].map(lambda t: np.random.randint(low=1000, high=1000) if t > 1000 else t)
## %%对财产数据处理
train_b2.drop(['propvalue','property'], inplace=True, axis=1)
train_b2['propextent'] = train_b2['propextent'].map(lambda t: np.random.randint(low=3, high=5) if math.isnan(t) else t)
## %% 对绑架信息的处理
train_b2['ishostkid'] = train_b2['ishostkid'].fillna(train_b2['ishostkid'].mode()[0])
## %% 对附加信息处理
train_b2.drop(['INT_LOG', 'INT_MISC', 'INT_IDEO', 'INT_ANY'], inplace=True, axis=1)
## %% 根据相关性删除数据
train_b2.drop(['specificity', 'vicinity'], inplace=True, axis=1)
train_b3 = copy.deepcopy(train_b2[(train_b2['iyear']>2014) & (train_b2['iyear']<2017)])
## %% 删除缺失率较高的数据
all_data_na = (train_b3.isnull().sum() / len(train_b3)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
train_b3.drop(all_data_na[all_data_na>90].index, inplace=True, axis=1)
## %% 凶手信息处理
train_b3['nperpcap'] = train_b3['nperpcap'].map(lambda t: 1 if t!=0 else 0)   # 是否抓获凶手
train_b3.drop('claimmode', inplace=True, axis=1)
buffer = train_b1['natlty1'].isnull()
train_b3['natlty1'][buffer] = train_b3['country'][buffer]
train_b3['nkillus'] = train_b3['nkillus'].fillna(train_b3['nkillus'].mode()[0])
train_b3['nwoundus'] = train_b3['nwoundus'].fillna(train_b3['nwoundus'].mode()[0])
train_b4 = copy.deepcopy(train_b3[train_b3['claimed']==0])
train_b4.drop('claimed', inplace=True, axis=1)
remove_game = list(train_b4['gname'].value_counts()[(train_b4['gname'].value_counts() < 410)].index)
train_b4 = train_b4.drop(train_b4[train_b4['gname'].isin(remove_game)].index, axis=0)
test_data0 = train_b4.drop(train_b4[train_b4['gname'] != 'Unknown'].index, axis=0)
train_data0 = train_b4.drop(train_b4[train_b4['gname'] == 'Unknown'].index, axis=0)
train_data0.drop('iyear', inplace=True, axis=1)
train_p = copy.deepcopy(train_b2[train_b2['iyear']==2017])
eventid_p = [201701090031, 201702210037, 201703120023, 201705050009, 201705050010,201707010028,201707020006,201708110018,201711010006,201712010003]
train_p_last = train_p[train_p['eventid'].isin(eventid_p)]
train_p_last.drop(['gname','claimed'], inplace=True, axis=1)
all_data_na = (train_p_last.isnull().sum() / len(train_p_last)) * 100
train_p_last.drop(all_data_na[all_data_na>90].index, inplace=True, axis=1)
train_p_last.drop('iyear', inplace=True, axis=1)
train_data0.index = range(0, len(train_data0))
test_data0.index = range(0, len(test_data0))
train_data0.drop(['eventid'], inplace=True, axis=1)
test_data0.drop(['eventid', 'gname'], inplace=True, axis=1)
### % 对gname打标签
lbl = LabelEncoder()
lbl.fit(list(train_data0['gname'].values))
train_data0['gname'] = lbl.transform(list(train_data0['gname'].values))
train_x, train_y, val_x, val_y = sample(train_data0)
## % 模型训练测试
model = XGBoost(train_x, train_y, val_x, val_y)
train_x.head()
## % 删除事件ID
train_p_last0 = train_p_last.drop('eventid', axis=1)
dtest = xgb.DMatrix(train_p_last0)
result = model.bst.predict(dtest)

## 生成结果
result1 = np.argsort(result,axis=1)
result2 = []
for v in result1:
    bu = list(reversed(v))
    result2.append(bu[:5])
xiongshou = [[] for i in range(10)]
for i,v in enumerate(result2):
    for k in v:
        xiongshou[i].append(lbl.classes_[k])
