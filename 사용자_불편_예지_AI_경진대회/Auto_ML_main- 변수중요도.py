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

from pycaret.classification import *
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

# +
train_err  = pd.read_csv('train_err_data.csv')
id_error = train_err[['user_id','errtype']].values
error = np.zeros((15000,42))
for person_idx, err in tqdm(id_error):
    # person_idx - 10000 위치에 person_idx, errtype에 해당하는 error값을 +1
    error[person_idx - 10000,err - 1] += 1

train_prob = pd.read_csv('train_problem_data.csv')
problem = np.zeros(15000)
problem[train_prob.user_id.unique()-10000] = 1 

train = pd.DataFrame(data=error)
train['problem'] = problem
del error, problem
train['problem'] = train['problem'].astype('category')
clf = setup(data = train, target = 'problem')

# -

train

train['problem'].value_counts()

train

'''
a = [9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 25, 29, 30, 33, 34, 5, 39]
b=[]
for i in range(42):
    if i not in a:
        b.append(str(i))
train=train.drop(b,axis=1)
train
'''





del train['0']

from pycaret.classification import *
best_5 = compare_models(sort = 'Accuracy', n_select = 6)


blended = blend_models(estimator_list = best_5, fold = 5, method = 'soft')

pred_holdout = predict_model(blended)

final_model = finalize_model(blended)

test_err  = pd.read_csv('test_err_data.csv')
id_error = test_err[['user_id','errtype']].values
test_x = np.zeros((14999,42))
for person_idx, err in tqdm(id_error):
    test_x[person_idx - 30000,err - 1] += 1
test_x = test_x.reshape(test_x.shape[0],-1)
test = pd.DataFrame(data=test_x)

predictions = predict_model(final_model, data = test)

'''
x = []
for i in range(len(predictions['Score'])):
    if predictions['Label'][i] =='1.0':
        x.append(predictions['Score'][i])
    else:
        x.append(1-predictions['Score'][i])

'''

sample_submssion = pd.read_csv('sample_submission.csv')
sample_submssion['problem'] = predictions['Score']
sample_submssion.to_csv("AutoML.csv", index = False)

# +
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn import tree
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
train_2 = train.drop('problem',axis=1)
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)

clf = clf.fit(train_2, train['problem'])
X_train, X_test, Y_train, Y_test = train_test_split(train_2, train['problem'], test_size=0.3)
clf.fit(X_train,Y_train)
print(round(clf.score(X_test,Y_test),2)*100,"%")
print("특성 중요도 : \n{}".format(clf.feature_importances_))
print(clf.feature_importances_)
list(train_2.columns.array)


imp = clf.feature_importances_
if platform.system() == 'Windows':
    font_name= font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
else:
    rc('font', family='AppleGothic')
# -

plt.barh(range(len(imp)), imp) 
#plt.yticks(range(len(imp)), list(df.columns.array)) 
plt.show()

a=imp.tolist()
for i in range(0,len(a)):
    
    print(str(i) , a[i])




