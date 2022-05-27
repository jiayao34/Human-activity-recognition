#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:28:41 2022

@author: jiayaoyuan

This code finds feature importance utilizing XGBooster.

"""
import pandas as pd
from xgboost import plot_importance,XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('MHEALTH_all.csv')

act_0 = df[df['label'] == 0] 
act_0 = act_0.sample(n = 30720, random_state = 1)
act_else = df[df['label'] != 0]
data_resample0 = pd.concat([act_0, act_else])

data_resample0_train = data_resample0[(data_resample0['subjectID'] != 9) & (data_resample0['subjectID'] != 10)]
data_resample0_test = data_resample0.drop(data_resample0_train.index, axis = 0)


# prepare data into inputs and outputs format for both training and testing dataset
y_resample0_train = data_resample0_train['label']
X_resample0_train = data_resample0_train.drop(['label', 'subjectID'], axis = 1)
y_resample0_test = data_resample0_test['label']
X_resample0_test = data_resample0_test.drop(['label', 'subjectID'], axis = 1)

# data normalization
scaler = StandardScaler().fit(X_resample0_train)
X_train_scaled = scaler.transform(X_resample0_train)
X_test_scaled = scaler.transform(X_resample0_test)

xgb = XGBClassifier(use_label_encoder=False) 
xgb_model = xgb.fit(X_train_scaled,y_resample0_train, eval_metric = 'merror', early_stopping_rounds=5, eval_set = [(X_test_scaled, y_resample0_test)])

# convert to pandas dataframe. assign 'feature' and 'importance columns'
xgb_fea_imp = pd.DataFrame(list(xgb_model.get_booster().get_fscore().items()),
                         columns=['feature','importance']).sort_values('importance', ascending=False)

# print feature importance for each attribute
print('',xgb_fea_imp)

# plot importance
plt.figure(figsize=(14,8))
plot_importance(xgb_model, )
