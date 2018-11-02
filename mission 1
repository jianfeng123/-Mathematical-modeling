# Load in our libraries
import pandas as pd
import re
import copy
import random
from scipy import stats
import warnings
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import norm
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
train_b1.drop(['eventid', 'iday', 'resolution', 'latitude', 'longitude'],inplace=True, axis=1)
corrmat = train_b1.corr()
remove_columns = [v for v in corrmat.columns if sum(abs(corrmat[v])) < 2.1]
train_b1.drop(remove_columns,inplace=True, axis=1)
## %国家分类 按照国家恐怖袭击发生的频率分为5类
thre = [0,10,100,1000,5000,30000]
train_b1_b = train_b1['country']
for i, v in enumerate(thre):
    if i < 5:
        for v1 in np.array(train_b1['country'].value_counts()[(train_b1['country'].value_counts() >= thre[i]) & (train_b1['country'].value_counts() < thre[i+1])].index):
            train_b1_b = train_b1_b.map(lambda t: i+1 if t==v1 else t)
train_b1['country'] = train_b1_b
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
# 受害者国籍
thre = [0,10,100,1000,5000,30000]
train_b1_b = train_b1['natlty1']
for i, v in enumerate(thre):
    if i < 5:
        for v1 in np.array(train_b1['natlty1'].value_counts()[(train_b1['natlty1'].value_counts() >= thre[i]) & (train_b1['natlty1'].value_counts() < thre[i+1])].index):
            train_b1_b = train_b1_b.map(lambda t: i if t==v1 else t)
train_b1['natlty1'] = train_b1_b
buffer = train_b1['natlty1'].isnull()
train_b1['natlty1'][buffer] = train_b1['country'][buffer]
# 输出影响较小的特征
train_b1.drop(['targtype2', 'targsubtype2', 'natlty2', 'targtype3', 'targsubtype3', 'natlty3'], inplace=True, axis=1)
## %对凶手信息的处理
thre = [0,10,100,2000,6000,10000,60000]
train_b1_b = train_b1['gname']
for i, v in enumerate(thre):
    if i < len(thre)-1:
        for v1 in np.array(train_b1['gname'].value_counts()[(train_b1['gname'].value_counts() >= thre[i]) & (train_b1['gname'].value_counts() < thre[i+1])].index):
            train_b1_b = train_b1_b.map(lambda t: i if t==v1 else t)
train_b1['gname'] = train_b1_b
train_b1.drop(['guncertain2', 'guncertain3', 'guncertain1'], inplace=True, axis=1)
## %%对声明的处理
train_b2 = copy.deepcopy(train_b1)
train_b2.drop(['nperpcap', 'claimed','nperps', 'claim2', 'claimmode', 'claimmode2', 'claimmode3', 'claim3', 'compclaim'], inplace=True, axis=1)
## %% 对使用的武器处理
train_b2.drop(['weaptype2', 'weapsubtype1', 'weapsubtype2', 'weaptype3', 'weaptype4', 'weapsubtype3', 'weapsubtype4'], inplace=True, axis=1)
## %% 对伤亡情况的处理
train_b2['nkill'] = train_b2['nkill'].fillna(train_b2['nkill'].median())
train_b2['nkillter'] = train_b2['nkillter'].fillna(train_b2['nkillter'].median())
train_b2['nwound'] = train_b2['nwound'].fillna(train_b2['nwound'].median())
train_b2['nwoundte'] = train_b2['nwoundte'].fillna(train_b2['nwoundte'].median())
## method 1 处理死亡噪声点
train_b2.drop(['nkillus', 'nwoundus'], inplace=True, axis=1)
train_b2['nkill'] = train_b2['nkill'].map(lambda t: np.random.randint(low=800, high=1000) if t > 1000 else t)
train_b2['nkillter'] = train_b2['nkillter'].map(lambda t: np.random.randint(low=800, high=1000) if t > 1000 else t)
train_b2['nwound'] = train_b2['nwound'].map(lambda t: np.random.randint(low=800, high=1000) if t > 1000 else t)
train_b2['nwoundte'] = train_b2['nwoundte'].map(lambda t: np.random.randint(low=1000, high=1000) if t > 1000 else t)
## %%对财产数据处理
train_b2['property'] = train_b2['property'].map(lambda t: np.random.randint(2) if t==-9 else t)
train_b2.drop(['propvalue'], inplace=True, axis=1)
train_b2['propextent'] = train_b2['propextent'].map(lambda t: np.random.randint(low=3, high=5) if math.isnan(t) else t)
## %% 对绑架信息的处理
train_b2['ishostkid'] = train_b2['ishostkid'].fillna(train_b2['ishostkid'].mode()[0])
train_b2['ishostkid'] = train_b2['ishostkid'].map(lambda t: train_b2['ishostkid'].mode()[0] if t==-9 else t)
train_b2.drop(['nhostkid', 'nhostkidus', 'nhours', 'ndays', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid'], inplace=True, axis=1)
train_b2.drop(['ransomnote', 'hostkidoutcome', 'nreleased','ransompaidus'], inplace=True, axis=1)
## %% 对附加信息处理
train_b2.drop(['INT_LOG', 'INT_MISC', 'INT_IDEO', 'INT_ANY'], inplace=True, axis=1)
## %% 根据相关性删除数据
train_b2.drop(['specificity', 'vicinity'], inplace=True, axis=1)
lbl = LabelEncoder()
lbl.fit(list(train_b2['iyear'].values))
train_b2['iyear'] = lbl.transform(list(train_b2['iyear'].values))
train_b2.drop('imonth', inplace=True, axis=1)
train_b3 = copy.deepcopy(train_b2)
## 高斯混合聚类
train_b3 = copy.deepcopy(train_b2)
clu = GaussianMixture(n_components=5,max_iter=100)
clu.fit(train_b3)
train_b3['class'] = clu.predict(train_b3)
for i in range(5):
    a = train_b3[train_b3['class'] == i]['nkill'].mean()
    b = train_b3[train_b3['class'] == i]['nwound'].mean()
    d = train_b3[train_b3['class'] == i]['propextent'].mean()
    e = len(train_b3[train_b3['class'] == i])
    print('class is {0:>6}, mean nkill is {1:>8.3}, mean nwound is {2:>7.2},mean propextent is {3:>4.2}, len is {4}'.format(i,a,b,d,e))    
## KPrototypes聚类
train_b3 = copy.deepcopy(train_b2)
Kpro = KPrototypes(n_clusters=5, init='Huang', verbose=10)
train_b3['class'] = Kpro.fit_predict(train_b3.values, categorical=[1,2,3,4,5])
for i in range(5):
    a = train_b3[train_b3['class'] == i]['nkill'].mean()
    b = train_b3[train_b3['class'] == i]['nwound'].mean()
    d = train_b3[train_b3['class'] == i]['propextent'].mean()
    e = len(train_b3[train_b3['class'] == i])
    print('class is {0:>6}, mean nkill is {1:>8.3}, mean nwound is {2:>7.2},mean propextent is {3:>4.2}, len is {4}'.format(i,a,b,d,e))    

## Kmeans聚类
kmeans = KMeans(n_clusters=5,random_state=20, max_iter=100, n_jobs=-1, n_init=100).fit(train_b3)
train_b3['class'] = kmeans.labels_
for i in range(5):
    a = train_b3[train_b3['class'] == i]['nkill'].mean()
    b = train_b3[train_b3['class'] == i]['nwound'].mean()    
    d = train_b3[train_b3['class'] == i]['propextent'].mean()
    e = len(train_b3[train_b3['class'] == i])
    print('class is {0:>6}, mean nkill is {1:>8.3}, mean nwound is {2:>7.2},mean propextent is {3:>4.2}, len is {4}'.format(i,a,b,d,e))

eventid_c = [200108110012, 200511180002, 200901170021, 201402110015, 201405010071,201411070002,201412160041,201508010015,201705080012]
train_b3['eventid'] = train_data_b['eventid']
train_b3[train_b3['eventid'].isin(eventid_c)]
print(train_b3[train_b3['class']==2]['eventid'].values)
train_b3['class'] = train_b3['class'].map(lambda t: 5 if t == 0 else t)
train_b3['class'] = train_b3['class'].map(lambda t: 3 if t == 1 else t)
train_b3['class'] = train_b3['class'].map(lambda t: 1 if t == 2 else t)
train_b3['class'] = train_b3['class'].map(lambda t: 4 if t == 3 else t)
train_b3['class'] = train_b3['class'].map(lambda t: 2 if t == 4 else t)

## 相关系数绘制
colormap = plt.cm.RdBu
corrmat = train_b3.corr('spearman')
plt.subplots(figsize=(30,30))
sns.heatmap(corrmat, cbar=True, cmap=colormap, annot=True, square=True, fmt='.2f', annot_kws={'size': 8})
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20, rotation=0)
plt.savefig('heatmap2')
plt.show()
