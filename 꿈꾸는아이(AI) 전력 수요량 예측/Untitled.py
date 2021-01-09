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

# +
import csv
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
submission = pd.read_csv("submission.csv")
# -

train.head()

test.head()

print(train.shape)
print(test.shape)
print(submission.shape)

train['Time']

# +
#시계열 데이터 분석을 위해서 데이터를 시계열 데이터에 맞게 포메팅 해준다.

train['Time'] = pd.to_datetime(train['Time'], format='%Y-%m-%d %H:%M', errors='raise')
# -



# +
# 컬럼 정보 리스트에 저장
cols = train.columns
cols_2 = train.columns
# 컬럼 별 평균 값 담기 위한 딕셔너리
train_mean = {}

x=[]
y=[]
for i in cols[1:-4]:
    print(str(i) + '의 평균 전력 수요량 :' , train[i].mean())
    train_mean[i] = train[i].mean()
    if train[i].mean() != 'nan':
        x.append(i)
        y.append(train[i].mean())

# +

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 6]
z=[]

for i in y:
    try:
        z.append(int(i))
    except Exception:
        pass
print(len(y))
print(len(z))
plt.ylim([0,100])
plt.boxplot(z)
plt.show()


# -

df = pd.DataFrame(y)

df.describe()


df.info()

df.
