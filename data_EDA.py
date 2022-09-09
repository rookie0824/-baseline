import random
import pandas as pd
from sklearn import preprocessing, manifold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import GridSearchCV
import glob
import datetime
import os
plt.rcParams['font.sans-serif']=['SimHei']

# data_paths = glob.glob(r"./dataset/数据集/*.csv")
# data_paths = sorted(data_paths, key = lambda x: int(str((str(x).split('\\')[-1])).split('.')[0]))
# print(data_paths)
# path = data_paths[0]
# dataset = pd.read_csv(path, encoding='utf-8')
# for i, path in enumerate(data_paths):
#     if i > 0:
#         id_data = pd.read_csv(path, encoding='utf-8')
#         dataset = pd.concat([dataset, id_data], axis=0)
#
# dataset = dataset.reset_index(drop=True)
#
# # 处理时间数据
# dataset['时间'] = pd.to_datetime(dataset['时间'])
# dataset['时间'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
# dataset['year'] = dataset['时间'].dt.year
# dataset['month'] = dataset['时间'].dt.month
# dataset['day'] = dataset['时间'].dt.day
# dataset = dataset.drop(['时间', 'year'], axis=1)
#
# features = dataset.columns
# print(features)
# num_feat = features.drop(['商品id', '总销量'])

# 添加属性
# dataset_agg = dataset.groupby(['商品id'])[num_feat].agg(['mean', 'std', 'min', 'max', 'first', 'last'])
# dataset_agg.columns = ['_'.join(x) for x in dataset_agg.columns]
# dataset_agg.reset_index(inplace=True)
# print(dataset_agg.columns)
# dataset = dataset.merge(dataset_agg, on = '商品id')

days = 60
dataset = pd.read_csv(f'./dataset/after_{days}days_dataset.csv', encoding='utf-8')

# 查看缺省值
print(dataset.isnull().sum())
# dataset.describe().to_csv('./dataset_describe.csv', encoding='utf-8')

# 查看每一种商品随着时间的变化规律
# count = 0
# for key, values in dataset.groupby(['商品id']):
#     t = values['month']*30 + values['day']
#     id = values['商品id'].unique()
#     plt.plot(t, values['总销量_mean'], 'b.')
#     plt.title(id)
#     plt.savefig(f'./features_distribution/{count}.jpg')
#     count = count + 1
#     plt.show()
origin_feature = [column for column in dataset.columns if column not in ['商品id', '总销量_mean']]
step = 4
for i in range(len(origin_feature)//(4*4) + 1):
    features = origin_feature[i*step**2:(i+1)*step**2]
    fig_f2o, ax_f2o = plt.subplots(step, step, figsize=(10, 10))
    for idx, col in enumerate(features):
        ax_f2o[idx // step, idx % step].plot(dataset[col], dataset['总销量_mean'], 'b.')
        ax_f2o[idx // step, idx % step].set_title(col)
    fig_f2o.tight_layout()
    plt.savefig('./dataset_distribution/%d-%d.jpg' % (i*step, (i+1)*step))
    plt.show()


# 显示特征的分布情况
# origin_feature = [column for column in dataset.columns if column not in ['商品id']]
# step = 4
# for i in range(len(origin_feature)//(4*4) + 1):
#     features = origin_feature[i*step**2:(i+1)*step**2]
#     fig_f2o, ax_f2o = plt.subplots(step, step, figsize=(10, 10))
#     for idx, col in enumerate(features):
#         ax_f2o[idx // step, idx % step].boxplot(dataset[col])
#         ax_f2o[idx // step, idx % step].set_title(col)
#     fig_f2o.tight_layout()
#     plt.savefig('./dataset_distribution/boxplot-%d.jpg' % i)
#     plt.show()

# 显示特征的分布情况
# origin_feature = [column for column in dataset.columns if column not in ['商品id']]
# # step = 4
# # for i in range(len(origin_feature)//(4*4) + 1):
# #     features = origin_feature[i*step**2:(i+1)*step**2]
# #     fig_f2o, ax_f2o = plt.subplots(step, step, figsize=(10, 10))
# #     for idx, col in enumerate(features):
# #         ax_f2o[idx // step, idx % step].hist(dataset.loc[~(dataset[col].isna()), col], density=True, bins=15)
# #         ax_f2o[idx // step, idx % step].set_title(col)
# #     fig_f2o.tight_layout()
# #     plt.savefig('./dataset_distribution/dist-%d.jpg' % i)
# #     plt.show()


# 显示特征与Target的分布情况
# origin_feature = [column for column in dataset.columns if column not in ['商品id', '总销量']]
# step = 4
# for i in range(len(origin_feature)//(4*4) + 1):
#     features = origin_feature[i*step**2:(i+1)*step**2]
#     fig_f2o, ax_f2o = plt.subplots(step, step, figsize=(10, 10))
#     for idx, col in enumerate(features):
#         ax_f2o[idx // step, idx % step].plot(dataset[col], dataset['总销量'], 'b.')
#         ax_f2o[idx // step, idx % step].set_title(col)
#     fig_f2o.tight_layout()
#     plt.savefig('./dataset_distribution/%d-%d.jpg' % (i*step, (i+1)*step))
#     plt.show()

# 显示特征与目标的相关性
feat_corr = dataset.corr()
print(feat_corr.shape, type(feat_corr))
step = 16
for i in range(2):
    for j in range(2):
        corr = feat_corr.iloc[i*41: (i+1)*41, j*41: (j+1)*41]
        fig, ax = plt.subplots(figsize=(20, 20))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=None, vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
                    cmap='coolwarm', annot_kws={'fontsize': 10, 'fontweight': 'bold'}, cbar=False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        plt.title('Correlations between feat Variables\n', fontsize=16)
        plt.savefig("./dataset60_distribution/feat_corr_%d_%d.jpg" % (i, j))
        plt.show()





