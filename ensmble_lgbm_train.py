import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12),
                           height=500, width=1000))

def split_dataset(dataset, cv):

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    count = 0
    for key, values in dataset.groupby(['商品id']):
        if count == 0:
            train_data = values[:-7*cv]
            test_data = values[-7*cv:-7*(cv-1)]
        else:
            train_data = pd.concat([train_data, values[:-7*cv]], axis=0)
            test_data = pd.concat([test_data, values[-7*cv:-7*(cv-1)]], axis=0)
        count = count + 1

    return train_data, test_data

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

dataset = pd.read_csv("./dataset/del_dataset.csv", encoding='utf-8')
test_dataset = pd.read_csv("./dataset/del_test_dataset.csv", encoding='utf-8')
print(dataset.columns)

print(dataset.isnull().sum())
print(test_dataset.isnull().sum())

features = dataset.columns.drop(['商品id', '总销量'])

test_feat = test_dataset[features]
test_target = test_dataset['总销量']
cv_test_preds = np.zeros(test_feat.shape[0])
y_valid, gbm_val_probs, gbm_test_preds, gini = [], [], [], []
ft_importance = pd.DataFrame(index=test_feat.columns)

cvs = [1, 2, 3]
for cv in cvs:
    train_dataset, val_dataset = split_dataset(dataset, cv)
    X_train, X_val = train_dataset[features], test_dataset[features]
    y_train, y_val = train_dataset['总销量'], test_dataset['总销量']

    print("\nFold {}".format(cv))

    print("Train shape: {}, {}, Valid shape: {}, {}\n".format(
        X_train.shape, y_train.shape, X_val.shape, y_val.shape))

    params = {'boosting_type': 'dart',
              'n_estimators':2000,
              'num_leaves': 100,
              'learning_rate': 0.1,
              'max_depth': -1,
              'colsample_bytree': 0.5,
              'subsample': 0.5,
              'min_child_samples':200,
              'max_bins': 250,
              'reg_alpha': 1.0,
              'reg_lambda': 0.3,
              'objective': 'rmse',
              'random_state': 42,
              }

    gbm = LGBMRegressor(**params).fit(X_train, y_train,
                                        eval_set=[(X_train, y_train), (X_val, y_val)],
                                        callbacks=[early_stopping(200), log_evaluation(500)],
                                        eval_metric=['auc','rmse'])

    gbm_prob = gbm.predict(X_val)
    gbm_val_probs.append(gbm_prob)
    y_valid.append(y_val)

    y_pred = pd.DataFrame(data={'prediction': gbm_prob})
    y_true = pd.DataFrame(data={'target': y_val.reset_index(drop=True)})
    gini_score = amex_metric(y_true=y_true, y_pred=y_pred)
    gini.append(gini_score)

    rmse = ((y_true['target'] - y_pred['prediction']) ** 2).mean() ** 0.5
    ft_importance["Importance_Fold" + str(cv)] = gbm.feature_importances_
    print("Validation Gini: {:.4f}, RMSE: {:.4f}".format(gini_score, rmse))

    test_preds = gbm.predict(test_feat)
    cv_test_preds += test_preds

# show features importance
ft_importance['avg']=ft_importance.mean(axis=1)
ft_importance=ft_importance.avg.nlargest(50).sort_values(ascending=True)
pal=sns.color_palette("YlGnBu", 65).as_hex()
fig=go.Figure()
for i in range(len(ft_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=ft_importance[i],
                       line_color=pal[::-1][i],opacity=0.8,line_width=4))
fig.add_trace(go.Scatter(x=ft_importance, y=ft_importance.index, mode='markers',
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
fig.update_layout(template=temp,title='LGBM Feature Importance<br>Top 50',
                  margin=dict(l=150,t=80),
                  xaxis=dict(title='Importance', zeroline=False),
                  yaxis_showgrid=False, height=1000, width=800)
fig.show()

test_label = cv_test_preds/len(cvs)
print(test_label.shape)

submission = pd.DataFrame({'商品id': test_dataset['商品id'], '未来一周天均销量':test_label})
submission.to_csv('submission.csv', index=False, encoding='utf-8')






