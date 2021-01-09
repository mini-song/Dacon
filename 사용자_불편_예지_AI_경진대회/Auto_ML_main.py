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

train['problem'].value_counts()

from pycaret.classification import *
best_5 = compare_models(sort = 'Accuracy', n_select = 5)


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




