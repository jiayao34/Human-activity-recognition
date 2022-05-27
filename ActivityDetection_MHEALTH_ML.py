#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:17:52 2022

@author: jiayaoyuan

ML models were trained and tested. Models including:
- logistic regression
- KNN
- SVM
- Random forest
- XGBoost
Tested model based on sample and tested model based on subjects

"""
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
import xgboost as xgb

#%% Data loading
df = pd.read_csv('MHEALTH_all.csv')
# df = df.drop(['acc_ch_x', 'acc_ch_y', 'acc_ch_z', 'ECG1', 'ECG2', 'mag_la_x', 'mag_la_y', 'mag_la_z', 'mag_ra_x', 'mag_ra_y', 'mag_ra_z'], axis = 1)

#%% Data preprocessing
# resample acitivity 0 to the same length as activity 1
act_0 = df[df['label'] == 0] 
act_0 = act_0.sample(n = 30720, random_state = 1)
act_else = df[df['label'] != 0]
data_resample0 = pd.concat([act_0, act_else])
print(data_resample0.label.value_counts())

# # split data into training and testing datset. We test model on new subjects
# # testSubjectID = random.sample(range(1, 11), 2)
# testSubjectID = [9, 10]
# data_resample0_train = data_resample0[(data_resample0['subjectID'] != testSubjectID[0]) & (data_resample0['subjectID'] != testSubjectID[1])]
# data_resample0_test = data_resample0.drop(data_resample0_train.index, axis = 0)
# X_train = data_resample0_train.drop(['label','subjectID'],axis=1)
# y_train = data_resample0_train['label']
# X_test = data_resample0_test.drop(['label','subjectID'],axis=1)
# y_test = data_resample0_test['label']

# Split data between predictors and output variable
X = data_resample0.drop(['label', 'subjectID'], axis=1)
y = data_resample0['label']
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# data normalization
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Helper function to plot and report performances
def results_summarizer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    activity_map = {
        0: 'Null',
        1: 'Standing still',  
        2: 'Sitting and relaxing', 
        3: 'Lying down',  
        4: 'Walking',  
        5: 'Climbing stairs',  
        6: 'Waist bends forward',
        7: 'Frontal elevation of arms', 
        8: 'Knees bending (crouching)', 
        9: 'Cycling', 
        10: 'Jogging', 
        11: 'Running', 
        12: 'Jump front & back' 
    }
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm,
                annot=True,
                cmap='YlGnBu',
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values()
               ) 
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Activity')
    plt.ylabel('Actual Activity')
    plt.show()
    
    print(f'Accuracy Score: ' + '{:.4%}'.format(acc))
    print(f'Precision Score: ' + '{:.4%}'.format(prec))
    print(f'Recall Score: ' + '{:.4%}'.format(rec))
    print(f'F_1 Score: ' + '{:.4%}'.format(f1))
    
#%% Classification model development
# Results were not good.
log_reg = LogisticRegression(solver='newton-cg', max_iter=300)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
results_summarizer(y_test, y_pred_log)


#%% KNN
# tested KNN with different n_neighbors (5, 7)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
results_summarizer(y_test, y_pred_knn)

#%% SVM 
# Running very slow. Used google colab to run.

cost_choices = [0.025, 0.05, 0.1, 1, 5, 10]
for c in cost_choices:
    svc = SVC(kernel='linear', random_state=1, C=c)
    score = cross_val_score(svc, X_scaled, y, cv=5).mean()
    print(f'Cost value: {c}', f'Cross-validation score: {score}'

         
# Support Vector Classifier - polynomial
svc = SVC(kernel='poly', random_state=1)
score = cross_val_score(svc, X_scaled, y, cv=5).mean()
print(f'Cross-validation score: {score}')


# Support Vector Classifier - radial
svc = SVC(kernel='rbf', random_state=1)
score = cross_val_score(svc, X_scaled, y, cv=5).mean()
print(f'Cross-validation score: {score}')

# Radial with different regularization params
cost_choices = [0.025, 0.05, 0.1, 1, 5, 10]
for c in cost_choices:
    svc = SVC(kernel='rbf', random_state=1, C=c)
    score = cross_val_score(svc, X_scaled, y, cv=5).mean()
    print(f'Cost value: {c}', f'Cross-validation score: {score}')

# Radial with different gamma choices
gamma_choices = [0.0005, 0.001, 0.01, 0.1, 1, 5]
for g in gamma_choices:
    svc = SVC(kernel='rbf', random_state=1, gamma=g, C=10)
    score = cross_val_score(svc, X_scaled, y, cv=5).mean()
    print(f'Gamma value: {g}', f'Cross-validation score: {score}')

# Produce confusion matrix for hypertuned SVC
svc = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10)
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)
results_summarizer(y_test, y_pred_svc)

#%% random forest 
rfst = RandomForestClassifier(random_state=1, n_estimators=200, max_depth=None)
rfst.fit(X_train_scaled, y_train)
y_pred_rfst = rfst.predict(X_test_scaled)
results_summarizer(y_pred_rfst, y_test)

#%% XGBoost
param = {'max_depth': 10}
param['nthread'] = 4
param['num_class'] = 13
# merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases)
param['eval_metric'] = 'merror'
# set XGBoost to do multiclass classification using the softmax objective
param['objective']= 'multi:softmax'

dtrain = xgb.DMatrix(X_train_scaled, label = y_train)
dtest = xgb.DMatrix(X_test_scaled, label = y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 10
model = xgb.train(param, dtrain, num_round, evallist)
y_pred_xgb = model.predict(dtest)
results_summarizer(y_pred_xgb, y_test)

