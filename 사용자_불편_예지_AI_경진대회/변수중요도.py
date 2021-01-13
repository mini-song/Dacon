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
#clf = setup(data = train, target = 'problem')


# +

y = train[['problem']]
X = train
del X['problem']
X


# +
import numpy as np
### make X_shadow by randomly permuting each column of X
np.random.seed(42)
X_shadow = X.apply(np.random.permutation)
X_shadow.columns = ['shadow_' + str(feat) for feat in X.columns]
### make X_boruta by appending X_shadow to X
X_boruta = pd.concat([train, X_shadow], axis = 1)

from sklearn.ensemble import RandomForestRegressor
### fit a random forest (suggested max_depth between 3 and 7)
forest = RandomForestRegressor(max_depth = 5, random_state = 42)
forest.fit(X_boruta, y)
### store feature importances
feat_imp_X = forest.feature_importances_[:len(X.columns)]
feat_imp_shadow = forest.feature_importances_[len(X.columns):]
### compute hits
hits = feat_imp_X > feat_imp_shadow.max()
# -

hits

### initialize hits counter
hits = np.zeros((len(X.columns)))
### repeat 20 times
for iter_ in range(20):
   ### make X_shadow by randomly permuting each column of X
   np.random.seed(iter_)
   X_shadow = X.apply(np.random.permutation)
   X_boruta = pd.concat([X, X_shadow], axis = 1)
   ### fit a random forest (suggested max_depth between 3 and 7)
   forest = RandomForestRegressor(max_depth = 5, random_state = 42)
   forest.fit(X_boruta, y)
   ### store feature importance
   feat_imp_X = forest.feature_importances_[:len(X.columns)]
   feat_imp_shadow = forest.feature_importances_[len(X.columns):]
   ### compute hits for this trial and add to counter
   hits += (feat_imp_X > feat_imp_shadow.max())

hits

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
###initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 100 # number of trials to perform
)
### fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X), np.array(y))
### print results
green_area = X.columns[boruta.support_].to_list()
blue_area = X.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)

print("a")

# +
from multiprocessing import Pool
import lightgbm as lgb
from functools import partial
def lgb_boruta(iter_,train_x) :
    print(iter_)
    np.random.seed(iter_)
    X_shadow = train_x.apply(np.random.permutation)
    X_boruta = pd.concat([train_x, X_shadow], axis = 1)
    columns = train_x.columns.tolist() + [f"shadow_{i}" for i in train_x.columns.tolist()]
    X_boruta.columns = columns
    boruta_fac_var = fac_var + [f"shadow_{i}" for i in fac_var]
    dtrain = lgb.Dataset(X_boruta,label='problem',
                         feature_name = columns,
                         categorical_feature = boruta_fac_var)
    param = {'numleaves': 20, 'mindatainleaf': 20,
             'objective':'binary','maxdepth': 5, 
             "boosting": "rf","bagging_freq" : 1 , "bagging_fraction" : 0.8,
             'learningrate': 0.01,"metric": 'auc',
             "lambdal1": 0.1, "randomstate": 133,"verbosity": -1,
             "num_threads" : 1
            }
    lgbmclf = lgb.train(param, dtrain, 500,verbose_eval=-1,
#                         valid_sets=dtrain,
#                         early_stopping_rounds=100,
                         feature_name = columns,
                        categorical_feature= boruta_fac_var)
    importacne = lgbmclf.feature_importance()
    importacne = importacne / importacne.sum()
    feat_imp_X = importacne[:len(train_x.columns)]
    feat_imp_shadow = importacne[len(train_x.columns):]
    value = (feat_imp_X > feat_imp_shadow.max())
    return value
    
n_iter = 10
pool = Pool(n_iter)
result = pool.map(
    partial(lgb_boruta, train_x =X ) ,
    np.arange(n_iter))
pool.close()
pool.join()
# -

a = np.array(result).sum(axis=0)
plt.barh(np.arange(len(a)),a)
plt.yticks(np.arange(len(a)),labels = train_x.columns.tolist())
plt.show()

# +
#https://data-newbie.tistory.com/494
# -








