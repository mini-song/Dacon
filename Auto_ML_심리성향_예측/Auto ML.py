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

import pandas as pd
train = pd.read_csv('train.csv',index_col='index')
test = pd.read_csv('test_x.csv',index_col='index')
submission = pd.read_csv('sample_submission.csv')

# ## EDA

train.head()

test.head()

train.info()

test.info()

train.dtypes

test.dtypes

train.dtypes

# ## 마키아벨리니즘 Score 제시

#모든 항목을 다 계산에 쓰기 때문에 상관관계를 도출할때Secret인 것도 알수가 있었다. 
#상관관계를 Reverse 해야 빨간색이 나오기 때문에
# 양의 항목과 음의 항목이 만나면 음의 상관관계를 가질 것 이기 떄문에 
Answers = ['QaA', 'QbA', 'QcA', 'QdA', 'QeA','QfA', 'QgA', 'QhA', 'QiA', 'QjA','QkA', 'QlA', 'QmA', 'QnA', 'QoA', 'QpA', 'QqA', 'QrA', 'QsA', 'QtA']





# +
import seaborn as sns

correlations = train[Answers].corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)
# -

flipping_columns = ["QeA", "QfA", "QkA", "QqA", "QrA"]
for flip in flipping_columns: 
    train[flip] = 6 - train[flip]
    test[flip] = 6 - test[flip]

correlations = train[Answers].corr(method='spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

train['Mach_score'] = train[Answers].mean(axis = 1)
test['Mach_score'] = test[Answers].mean(axis = 1)
train.head()

# +
for i in train:
    train['T'] = train['QcA'] - train['QfA'] + train['QoA'] - train['QrA'] + train['QsA']
    train['V'] = train['QbA'] - train['QeA'] + train['QhA'] + train['QjA'] + train['QmA'] - train['QqA']
    train['M'] = - train['QkA']

for i in test:
    test['T'] = test['QcA'] - test['QfA'] + test['QoA'] - test['QrA'] + test['QsA']
    test['V'] = test['QbA'] - test['QeA'] + test['QhA'] + test['QjA'] + test['QmA'] - test['QqA']
    test['M'] = - test['QkA']



flipping_secret_columns = ["QaA", "QdA", "QgA", "QiA", "QnA"]

for flip in flipping_secret_columns: 
    train[flip] = 6 - train[flip]
    test[flip] = 6 - test[flip]
# -



import matplotlib.pyplot as plt
# %matplotlib inline
plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'age_group', hue = train['voted'])

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'education', hue = train['voted'])

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'engnat', hue = train['voted'])

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'familysize', hue = train['voted'])

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'gender', hue = train['voted'])

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'hand', hue = train['voted'])

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'married', hue = train['voted'])

plt.figure(figsize = (8,6))
plt.xticks(rotation=90)
sns.countplot(data = train, x = 'race', hue = train['voted'])

plt.figure(figsize = (8,6))
plt.xticks(rotation=90)
sns.countplot(data = train, x = 'religion', hue = train['voted'])

# #  TIPI(Ten-Item Personality Inventory) 

tp_columns = ["tp01","tp02","tp03","tp04","tp05","tp06","tp07","tp08","tp09","tp10"]

# +
import numpy as np
for i in tp_columns:
    train[i] = train[i].replace(0, np.nan)
    mean = train[i].mean(axis=0)
    train[i] = train[i].replace(np.nan , mean)

for i in tp_columns:
    test[i] = test[i].replace(0, np.nan)
    mean = test[i].mean(axis=0)
    test[i] = test[i].replace(np.nan , mean)
    
    
train['Extraversion'] = train['tp01'] - train['tp06']
train['Agreeableness'] = train['tp07'] - train['tp02']
train['Conscientiousness'] = train['tp03'] - train['tp08']
train['Emotional_Stability'] = train['tp09'] - train['tp04']
train['Openness_to_Experiences'] = train['tp05'] - train['tp10']

test['Extraversion'] = test['tp01'] - test['tp06']
test['Agreeableness'] = test['tp07'] - test['tp02']
test['Conscientiousness'] = test['tp03'] - test['tp08']
test['Emotional_Stability'] = test['tp09'] - test['tp04']
test['Openness_to_Experiences'] = test['tp05'] - test['tp10']


train = train.drop(tp_columns, axis = 1)

test = test.drop(tp_columns, axis = 1)


# -

train['Mach_score'] = train[Answers].mean(axis = 1)
test['Mach_score'] = test[Answers].mean(axis = 1)
train.head()

# %matplotlib inline
plt.figure(figsize = (8,6))
plt.xticks(rotation=90)
sns.countplot(data = train, x = 'Mach_score', hue = train['voted'])


train = train.drop(Answers, axis = 1)
test = test.drop(Answers, axis = 1)

# +
실존 = ['wr_01','wr_02','wr_03','wr_04','wr_05','wr_06','wr_07','wr_08','wr_09','wr_10','wr_11','wr_12','wr_13']
허구 = ['wf_01','wf_02','wf_03']


train['wr_sum']=train[실존].agg('sum',axis=1)
train = train.drop(실존, axis = 1)
train['wf_sum']=train[허구].agg('sum',axis=1)
train = train.drop(허구, axis = 1)


test['wr_sum']=test[실존].agg('sum',axis=1)
test['wf_sum']=test[허구].agg('sum',axis=1)

test = test.drop(실존, axis = 1)
test = test.drop(허구, axis = 1)
# -

plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'wr_sum', hue = train['voted'])


plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'wf_sum', hue = train['voted'])

word=['Extraversion'
'Agreeableness'
'Conscientiousness'
'Emotional_Stability'
'Openness_to_Experiences']

import matplotlib.pyplot as plt
plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'Extraversion', hue = train['voted'])

import matplotlib.pyplot as plt
# %matplotlib inline
plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'Agreeableness', hue = train['voted'])

# +

import matplotlib.pyplot as plt
# %matplotlib inline
plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'Conscientiousness', hue = train['voted'])
# -

import matplotlib.pyplot as plt
# %matplotlib inline
plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'Emotional_Stability', hue = train['voted'])

import matplotlib.pyplot as plt
# %matplotlib inline
plt.figure(figsize = (8,6))
sns.countplot(data = train, x = 'Openness_to_Experiences', hue = train['voted'])

train=pd.get_dummies(train,drop_first=True)
test=pd.get_dummies(test,drop_first=True)

# +
#import numpy as np
#Corr_list = train.corr().index[abs(train.corr()["voted"])>=0.1]
#Corr_list

# +
#import matplotlib.pyplot as plt # 여러 시각화 도구들
#import seaborn as sns #
#plt.figure(figsize=(20,18))
#sns.heatmap(data=train[Corr_list].corr() ,annot=True, cmap="RdYlGn")
#plt.show()

# +
#test

# +
#train=train[Corr_list]
#a=Corr_list.tolist()
#a.remove('voted')
#test=test[a]
# -

train

# # Modeling

from pycaret.classification import *

clf = setup(data = train, target = 'voted')
clf

best_3 = compare_models(sort = 'AUC', n_select = 3)

blended = blend_models(estimator_list = best_3, fold = 5, method = 'soft')

pred_holdout = predict_model(blended)

final_model = finalize_model(blended)

predictions = predict_model(final_model, data = test)
predictions

submission['voted'] = predictions['Score']

submission.to_csv('submission_proba1.csv', index = False)

# +
#https://pycaret.readthedocs.io/en/latest/index.html#pycaret.classification.setup
