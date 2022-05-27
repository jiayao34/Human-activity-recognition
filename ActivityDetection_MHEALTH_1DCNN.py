#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:34:35 2022

@author: jiayaoyuan

In this code, data for 1D CNN model was prepared.
1D CNN models with different structures were traied and tested. 
Models were tested from sample based and subject based dataset
1D CNN was designed following LeNet Archetecture.

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
import seaborn as sns
import matplotlib.pyplot as plt
#%% load dataset
df = pd.read_csv('MHEALTH_all.csv')

#%% Dataset info
print("=================================================================")
print("The entire dataset shape: ", df.shape)
print("=================================================================")
print("Data header: \n", df.columns)
print("=================================================================")
print("Number of data points each subject has\n", df.subjectID.value_counts())
print("=================================================================")
print("Unique activity labels for all subjects are: \n", df['label'].unique())
print("=================================================================")
print("Number of data pointns each activity has\n", df['label'].value_counts())
print("=================================================================")

#%% Data pre-processing
# the datset is unbalanced due to too much "0" activity.

# resample acitivity 0 to the same length as activity 1
act_0 = df[df['label'] == 0] 
act_0 = act_0.sample(n = 30720, random_state = 1)
act_else = df[df['label'] != 0]
data_resample0 = pd.concat([act_0, act_else])

# # train test data preparation
# X = data_resample0.drop(['label', 'subjectID'], axis=1)
# y = data_resample0['label']
# X_scaled = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# split data into training and testing datset. We test model on new subjects
# testSubjectID = random.sample(range(1, 11), 2)
testSubjectID = [7, 8]
data_resample0_train = data_resample0[(data_resample0['subjectID'] != testSubjectID[0]) & (data_resample0['subjectID'] != testSubjectID[1])]
data_resample0_test = data_resample0.drop(data_resample0_train.index, axis = 0)

X_train = data_resample0_train.drop(['label','subjectID'],axis=1)
y_train = data_resample0_train['label']
X_test = data_resample0_test.drop(['label','subjectID'],axis=1)
y_test = data_resample0_test['label']

# Scale train/ test predictors based on training data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
train_set = np.c_[X_train_scaled, y_train_array]
y_test_array = np.array(y_test)
test_set = np.c_[X_test_scaled, y_test_array]

# Apply sequence transformation using time step of 25 for both train and test data
X_train_seq, y_train_seq = split_sequences(train_set, 25, 1)
print(X_train_seq.shape, y_train_seq.shape)

X_test_seq, y_test_seq = split_sequences(test_set, 25, 1)
print(X_test_seq.shape, y_test_seq.shape)

# Convert output variables to categorical for CNN
y_train_seq = to_categorical(y_train_seq)
print(y_train_seq.shape)
print(y_train_seq)

y_test_seq = to_categorical(y_test_seq)
print(y_test_seq.shape)
print(y_test_seq)

n_timesteps, n_features, n_outputs = X_train_seq.shape[1], X_train_seq.shape[2], y_train_seq.shape[1]

#%% 1D CNN model development - developed following LeNet Archetecture
model = Sequential()
# 2 1D CNN layers for learning features from the time series samples
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features), padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
# Flatten learned features to vector
model.add(Flatten())
# Fully connected dense layer - interpret features
model.add(Dense(512, activation='relu'))
# Dropout layer for easing learning process
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
# Dropout layer for easing learning process
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
# Output layer using softmax
model.add(Dense(n_outputs, activation='softmax'))
model.summary()
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5, verbose = 1)]
# Fit model on training data with 10 epochs
model_history = model.fit(X_train_seq, y_train_seq, epochs=10, validation_data=(X_test_seq,y_test_seq),callbacks=callbacks)


#%% Model evaluation
y_pred = model.predict(X_test_seq)
y_pred = np.argmax(y_pred, axis = 1)
y_pred = y_pred.reshape(-1,1)
# Results summarizer function (scores, confusion matrix) for classification results
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
    
results_summarizer(y_test[24:], y_pred)

train_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
train_accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']

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
