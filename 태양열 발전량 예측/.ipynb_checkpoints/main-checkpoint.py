# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import time
import statsmodels.api as sm
from sklearn import linear_model
from tqdm import tnrange, tqdm_notebook
from sklearn.metrics import mean_squared_error
from statsmodels.regression.quantile_regression import QuantReg
from fbprophet import Prophet


# # Data Load

train = pd.read_csv('./train.csv')
sub = pd.read_csv('./sample_submission.csv')
#2nd

train.head()

# # 전처리

for i in range(0,81):
    s1 = "test_%d = pd.read_csv('test/%d.csv')"%(i,i)
    exec(s1)

for i in range(0, 1093):
    s1 = "train_%d = pd.concat([train.loc[train['Day'] == %d].reset_index(drop = True), train.loc[train['Day'] == %d + 2].reset_index(drop = True).rename(columns = {'TARGET' : 'TARGET_2'})['TARGET_2']], axis = 1)"%(i, i, i)
    exec(s1)

train_x = pd.concat([train_0, train_1], axis = 0).reset_index(drop=True)
#train_x
train['Day']+ 

for i in range(2, 1093):
    s1 = "train_x = pd.concat([train_x, train_%d], axis = 0).reset_index(drop = True)"%(i)
    exec(s1)

#변수생성
hour_mean = train_x.groupby(['Hour'])['TARGET'].mean()
minute_mean = train_x.groupby(['Minute'])['TARGET'].mean()
train_x['Hour_mean'] = train_x['Hour'].map(hour_mean)
train_x['Minute_mean'] = train_x['Minute'].map(minute_mean)

train_x = train_x.drop(['Day', 'Hour', 'Minute'], axis = 1)
train_x

x_train = train_x.drop(['TARGET_2'], axis = 1)
y_train = train_x['TARGET_2']

for i in range(0, 81):
    s1 = "test_%d = test_%d[240:].reset_index(drop = True)"%(i,i)
    s2 = "hour_mean = test_%d.groupby(['Hour'])['TARGET'].mean()"%(i)
    s3 = "minute_mean = test_%d.groupby(['Minute'])['TARGET'].mean()"%(i)
    s4 = "test_%d['Hour_mean'] = test_%d['Hour'].map(hour_mean)"%(i,i)
    s5 = "test_%d['Minute_mean'] = test_%d['Minute'].map(minute_mean)"%(i,i)
    s6 = "test_%d = test_%d.drop(['Day', 'Hour', 'Minute'], axis = 1)"%(i,i)
    exec(s1)
    exec(s2)
    exec(s3)
    exec(s4)
    exec(s5)
    exec(s6)

test_x = pd.concat([test_0, test_1], axis = 0).reset_index(drop=True)
test_x

for i in range(2, 81):
    s1 = "test_x = pd.concat([test_x, test_%d], axis = 0).reset_index(drop = True)"%(i)
    exec(s1)

x_test = test_x.copy()


# +
def linear_reg(X,y):
    model = linear_model.LinearRegression(fit_intercept = True)
    model.fit(X,y)
    RSS = mean_squared_error(y,model.predict(X)) * len(y)
    R_squared = model.score(X,y)
    return RSS, R_squared

#X변수 개수
m = len(y_train)
k = 8
RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []

#1 ~ 8까지 모든 조합에 대한 성능 테스트(RSS, R_Squared)
for k in tnrange(1,len(x_train.columns) + 1):
    for combo in itertools.combinations(x_train.columns,k):
        tmp_result = linear_reg(x_train[list(combo)],y_train)  
        RSS_list.append(tmp_result[0])              
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))   

#결과
df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
# -

df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
df.head()

df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
display(df_min.head(5))

# +
fig = plt.figure(figsize = (16,6))
ax = fig.add_subplot(1, 2, 1)

ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
ax.set_xlabel('Features')
ax.set_ylabel('RSS')
ax.set_title('RSS - Best subset selection')
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
ax.set_xlabel('Features')
ax.set_ylabel('R squared')
ax.set_title('R_squared - Best subset selection')
ax.legend()

plt.show()
# -

sub['q_0.1'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.1).predict(x_test['Hour_mean'])
sub['q_0.2'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.2).predict(x_test['Hour_mean'])
sub['q_0.3'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.3).predict(x_test['Hour_mean'])
sub['q_0.4'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.4).predict(x_test['Hour_mean'])
sub['q_0.5'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.5).predict(x_test['Hour_mean'])
sub['q_0.6'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.6).predict(x_test['Hour_mean'])
sub['q_0.7'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.7).predict(x_test['Hour_mean'])
sub['q_0.8'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.8).predict(x_test['Hour_mean'])
sub['q_0.9'] = QuantReg(y_train, x_train['Hour_mean']).fit(q=0.9).predict(x_test['Hour_mean'])

sub.to_csv('1214_third.csv', index = False)








