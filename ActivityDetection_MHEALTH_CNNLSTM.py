#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:47:18 2022

@author: jiayaoyuan

This code use only acc and gyro sensors to save compuation time.
Choosing acc and gyro because they are more important as shown in feature importance study.
To test models on new subjects, we continue modifying CNN archetecture.
This model added a LSTM layer on 1D CNN

"""

# Imports
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import keras
from sklearn.model_selection import train_test_split
from keras import layers
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import random

#%% load dataset
df = pd.read_csv('MHEALTH_all.csv')
df = df.drop(['acc_ch_x', 'acc_ch_y', 'acc_ch_z', 'ECG1', 'ECG2', 'mag_la_x', 'mag_la_y', 'mag_la_z', 'mag_ra_x', 'mag_ra_y', 'mag_ra_z'], axis = 1)

#%% Data preprocessing
# resample acitivity 0 to the same length as activity 1
act_0 = df[df['label'] == 0] 
act_0 = act_0.sample(n = 30720, random_state = 1)
act_else = df[df['label'] != 0]
data_resample0 = pd.concat([act_0, act_else])
print(data_resample0.label.value_counts())

# split data into training and testing datset. We test model on new subjects
# testSubjectID = random.sample(range(1, 11), 2)
testSubjectID = [9, 10]
data_resample0_train = data_resample0[(data_resample0['subjectID'] != testSubjectID[0]) & (data_resample0['subjectID'] != testSubjectID[1])]
data_resample0_test = data_resample0.drop(data_resample0_train.index, axis = 0)

X_train = data_resample0_train.drop(['label','subjectID'],axis=1)
y_train = data_resample0_train['label']
X_test = data_resample0_test.drop(['label','subjectID'],axis=1)
y_test = data_resample0_test['label']

# reformating the traiing data into sequence samples
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
# helper function to split sequences
def split_sequences(sequences, n_steps, step):
	X, y = list(), list()
	for i in range(0, len(sequences), step):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# Merge train and test X/y data to apply sequence transformation function
y_train_array = np.array(y_train)
train_set = np.c_[X_train, y_train_array]
y_test_array = np.array(y_test)
test_set = np.c_[X_test, y_test_array]

# Apply sequence transformation using window size of 100, step 50 for both train and test data
X_train_seq, y_train_seq = split_sequences(train_set, 100, 50)
print(X_train_seq.shape, y_train_seq.shape)

X_test_seq, y_test_seq = split_sequences(test_set, 100, 50)
print(X_test_seq.shape, y_test_seq.shape)

#%% Model development
model = keras.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, input_shape=(100, 12), padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Conv1D(filters=64, kernel_size=3, padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPool1D(2))
model.add(layers.LSTM(64))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(13, activation='softmax'))
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint("mhealth_best.h5", save_best_only = True, monitor="val_loss"),
             keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5, verbose = 1)]
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
model_history = model.fit(X_train_seq,y_train_seq, epochs = 10, validation_data=(X_test_seq,y_test_seq), callbacks=callbacks)

#%% Evaluate model against test data
_, accuracy = model.evaluate(X_test_seq, y_test_seq)
print(accuracy)

# Make predictions on the test data
y_pred = model.predict(X_test_seq)
y_pred = np.argmax(y_pred, axis = 1)
y_pred = y_pred.reshape(-1,1)
# Helper function to show all metrics
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
                cmap='Blues',
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
    
results_summarizer(y_test_seq, y_pred)

train_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
train_accuracy = model_history.history['sparse_categorical_accuracy']
val_accuracy = model_history.history['val_sparse_categorical_accuracy']

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(train_loss, 'r', label='Training loss')
plt.plot(val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracy, 'r', label='Training Accuracy')
plt.plot(val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
