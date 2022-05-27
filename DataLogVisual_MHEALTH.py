#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:37:04 2022
@author: jiayaoyuan
This code appends all subjects data, and performs data visulization
Time plots, histogram for different sensors and different activities were plotted and compared.
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load all data - Append subject data from 1-10 and save it as csv file
dataSetAll = pd.DataFrame()

# Append all the log files into Pandas DataFrame
for i in range(1, 11):
    dataSet = pd.read_csv(f'mHealth_subject{i}.log', sep = "\t", header = None)
    dataSet['subjectID'] = i
    print(f'======== Appending subject{i} file: data size {dataSet.shape} ======== ')
    dataSetAll = pd.concat([dataSetAll, dataSet])

# remove ECG data
df = dataSetAll.loc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 'subjectID']]
print(f'======== All data: size {df.shape} ======== ')

# la: left ankle, ra: right arm
df.rename({
        0: 'acc_ch_x', 
        1: 'acc_ch_y', 
        2: 'acc_ch_z', 
        3: 'ECG1',
        4: 'ECG2',
        5: 'acc_la_x', 
        6: 'acc_la_y',
        7: 'acc_la_z', 
        8: 'gyr_la_x',
        9: 'gyr_la_y', 
        10: 'gyr_la_z', 
        11: 'mag_la_x', 
        12: 'mag_la_y', 
        13: 'mag_la_z', 
        14: 'acc_ra_x', 
        15: 'acc_ra_y',
        16: 'acc_ra_z', 
        17: 'gyr_ra_x', 
        18: 'gyr_ra_y', 
        19: 'gyr_ra_z', 
        20: 'mag_ra_x', 
        21: 'mag_ra_y', 
        22: 'mag_ra_z',
        23: 'label'
    }, axis = 1, inplace = True)
# df.to_csv('MHEALTH_all.csv', index=False)

activity_map = {
    1: 'Standing still (1 min)',  
    2: 'Sitting and relaxing (1 min)', 
    3: 'Lying down (1 min)',  
    4: 'Walking (1 min)',  
    5: 'Climbing stairs (1 min)',  
    6: 'Waist bends forward (20x)',
    7: 'Frontal elevation of arms (20x)', 
    8: 'Knees bending (crouching) (20x)', 
    9: 'Cycling (1 min)', 
    10: 'Jogging (1 min)', 
    11: 'Running (1 min)', 
    12: 'Jump front & back (20x)' 
}

#%% Data info
print("=================================================================")
print("The entire data set has shape: ", df.shape)
print("=================================================================")
print("Data header: ", df.columns)
print("=================================================================")
df.info()
print("=================================================================")
print("Number of data points each subject has\n", df.subjectID.value_counts())
print("=================================================================")
print("Unique activity labels for all subjects are: \n", df['label'].unique())
print("=================================================================")
print("Number of data pointns each activity has\n", df['label'].value_counts())
print("=================================================================")

df.subjectID.value_counts().plot.bar(xlabel = 'Subject ID', ylabel = 'Number of samples', rot = 0)
df.label.value_counts().plot.bar(xlabel = 'Activity labels', ylabel = 'Number of samples',rot = 0)


#%% Data visualization - Plot sensor data subject by subject; plot subject data sensor by sensor
# plot chest acc for subject 1-10. 
for i in range(1, 11):
    subject = df[df['subjectID'] == i]
    time = np.linspace(0, len(subject)/50/60, len(subject))
    # plots
    ax = plt.figure(figsize = (14,4))
    ax = plt.subplot(3,1,1)
    plt.title(f'Subject {i}')
    ax2 = ax.twinx()
    l1 = ax.plot(time, subject['acc_ch_x'], color = 'blue', label = 'acc_ch_x')
    l2 = ax2.plot(time, subject['label'], color = 'red', alpha=.7, label = 'labels')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs,ncol=2,  loc = 2)
    ax.set_xticklabels([])
    ax.grid()
    
    ax = plt.subplot(3,1,2)
    ax2 = ax.twinx()
    l1 = ax.plot(time, subject['acc_ch_y'], color = 'blue', label = 'acc_ch_y')
    ax.set_ylabel('Chest Acc Sensor', color="blue",fontsize=14)
    l2 = ax2.plot(time, subject['label'], color = 'red', alpha=.7, label = 'labels')
    ax2.set_ylabel('Labels', color="red",fontsize=14)
    # added these three lines
    lns = l1 + l2 
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, ncol=2, loc = 2)
    ax.set_xticklabels([])
    ax.grid()
    
    ax = plt.subplot(3,1,3)
    ax2 = ax.twinx()
    l1 = ax.plot(time, subject['acc_ch_z'], color = 'blue', label = 'acc_ch_z')
    ax.set_xlabel('Time (mins)')
    l2 = ax2.plot(time, subject['label'], color = 'red', alpha=.7, label = 'labels')
    # added these three lines
    lns = l1 + l2 
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, ncol=2, loc = 2)
    ax.grid()
    plt.show()
    
#%% plot all sensor data for one subject
subject1 = df[df['subjectID'] == 1]
time1 = np.linspace(0, len(subject1)/50/60, len(subject1))
readings = ['acc', 'gyr', 'mag']
for r in readings:
    ax = plt.figure(figsize=(14,8))
    ax = plt.subplot(2,1,1)
    ax2 = ax.twinx()
    l1 = ax.plot(time1, subject1[r + '_la_x'], alpha=.7, label= r + '_la_x')
    l2 = ax.plot(time1, subject1[r + '_la_y'], alpha=.7, label= r + '_la_y')
    l3 = ax.plot(time1, subject1[r + '_la_z'], alpha=.7, label= r + '_la_z')
    l4 = ax2.plot(time1, subject1['label'], color = 'red')
    ax.set_ylabel('Sensor', color="black",fontsize=14)
    ax2.set_ylabel('Label', color="red",fontsize=14)
    plt.title('Left ankle sensor')
    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, ncol=3, loc = 2)
    ax.grid()
    plt.legend()
    
    ax = plt.subplot(2,1,2)
    ax2 = ax.twinx()
    l1 = ax.plot(time1, subject1[r + '_ra_x'], alpha=.7, label= r + '_ra_x')
    l2 = ax.plot(time1, subject1[r + '_ra_y'], alpha=.7, label= r + '_ra_y')
    l3 = ax.plot(time1, subject1[r + '_ra_z'], alpha=.7, label= r + '_ra_z')
    l4 = ax2.plot(time1, subject1['label'], color = 'red')
    ax.set_ylabel('Sensor', color="black",fontsize=14)
    ax2.set_ylabel('Label', color="red",fontsize=14)
    ax.set_xlabel('Time (mins)')
    plt.title('Right arm sensor')
    plt.legend()
    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, ncol=3, loc = 2)
    ax.grid()
    plt.show()
    
    if r == 'acc':
        ax = plt.figure(figsize=(14,4))
        ax = plt.subplot()
        ax2 = ax.twinx()
        l1 = plt.plot(time1, subject1[r + '_ch_x'], alpha=.7, label= r + '_ch_x')
        l2 = plt.plot(time1, subject1[r + '_ch_y'], alpha=.7, label= r + '_ch_y')
        l3 = plt.plot(time1, subject1[r + '_ch_z'], alpha=.7, label= r + '_ch_z')
        l4 = ax2.plot(time1, subject1['label'], color = 'red')
        ax.set_ylabel('Sensor', color="black",fontsize=14)
        ax2.set_ylabel('Label', color="red",fontsize=14)
        ax.set_xlabel('Time (mins)')
        plt.title('Chest sensor')
        plt.legend()
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, ncol=3, loc = 2)
        ax.grid()
        plt.show()
    
#%% Plot all sensor data for one subject, focused on one activity at a time.
# loop through all activity labels
for i in range(1, 13): 
    for r in readings:
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_la_x'], alpha=.7, label= r + '_la_x')
        plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_la_y'], alpha=.7, label= r + '_la_y')
        plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_la_z'], alpha=.7, label= r + '_la_z')
        plt.title(f'Left ankle sensor - {activity_map[i]}')
        plt.xlabel('Time (mins)')
        plt.grid()
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_ra_x'], alpha=.7, label= r + '_ra_x')
        plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_ra_y'], alpha=.7, label= r + '_ra_y')
        plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_ra_z'], alpha=.7, label= r + '_ra_z')
        plt.title(f'Right arm sensor - {activity_map[i]}')
        plt.xlabel('Time (mins)')
        plt.legend()
        plt.grid()
        plt.show()
        
        if r == 'acc':
            plt.figure(figsize=(7,4))
            plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_ch_x'], alpha=.7, label= r + '_ch_x')
            plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_ch_y'], alpha=.7, label= r + '_ch_y')
            plt.plot(time1[subject1['label']==i], subject1[subject1['label']==i][r + '_ch_z'], alpha=.7,label= r + '_ch_z')
            plt.title(f'Chest sensor - {activity_map[i]}')
            plt.xlabel('Time (mins)')
            plt.legend() 
            plt.grid()
            plt.show()

#%% Histogram showing data distribution per sensor - all data, one subjects
for i in range(1, 13):
    for r in readings:
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.hist(subject1[subject1['label']==i][r + '_la_x'], alpha=.7, label = r + '_la_x', bins = 50)
        plt.hist(subject1[subject1['label']==i][r + '_la_y'], alpha=.7, label = r + '_la_y', bins = 50)
        plt.hist(subject1[subject1['label']==i][r + '_la_z'], alpha=.7, label = r + '_la_z', bins = 550)
        plt.title(f'Left ankle sensor - {activity_map[i]}')
        plt.grid()
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.hist(subject1[subject1['label']==i][r + '_ra_x'], alpha=.7, label = r + '_ra_x', bins = 50)
        plt.hist(subject1[subject1['label']==i][r + '_ra_y'], alpha=.7, label = r + '_ra_y', bins = 50)
        plt.hist(subject1[subject1['label']==i][r + '_ra_z'], alpha=.7, label = r + '_ra_z', bins = 50)
        plt.title(f'Right arm sensor - {activity_map[i]}')
        plt.legend()
        plt.grid()
        plt.show()
        
        if r == 'acc':
            plt.figure(figsize=(7,4))
            plt.hist(subject1[subject1['label']==i][r + '_ch_x'], alpha=.7, label = r + '_ch_x', bins = 50)
            plt.hist(subject1[subject1['label']==i][r + '_ch_y'], alpha=.7, label = r + '_ch_y', bins = 50)
            plt.hist(subject1[subject1['label']==i][r + '_ch_z'], alpha=.7, label = r + '_ch_z', bins = 50)
            plt.title(f'Chest sensor - {activity_map[i]}')
            plt.legend() 
            plt.grid()
            plt.show()

#%% Plot hisgram for all 12 activities for each sensor and each direction
# COMMENTED OUT - EXPENSIVE COMPUTATION
# directions = ['x', 'y', 'z']
# pos = ['_la_', '_ra_']
# sensors = ['acc', 'gyr', 'mag']
# for s in sensors:
#     for p in pos:
#         for d in directions:
#             plt.figure(figsize=(7,4))
#             print('Plotting sensor ' + s + p + d)
#             for i in range(1, 13):
#                 plt.hist(df[df['label']==i][s + p + d], alpha= .5, label = f'{activity_map[i]}', bins = 50)
#             plt.title(s + p + d + f' - {activity_map[i]}')
#             plt.grid()
#             plt.legend()

#%% Histogram for sone selected sensor, all subjects data, different activity (randomly selected). 
directions = ['x', 'y', 'z']
pos = ['_la_', '_ra_']
sensors = ['gyr']
# randomAct = np.random.choice(range(1,13), size = 3, replace = False)
randomAct = [3, 9, 12]
# Histogram
for s in sensors:
    for p in pos:
        for d in directions:
            plt.figure(figsize=(7,4))
            print('======== Plotting sensor ' + s + p + d + ' ========')
            for i in randomAct:
                plt.hist(df[df['label']==i][s + p + d], alpha= .5, label = f'{activity_map[i]}', bins = 50)
            plt.title(s + p + d)
            plt.grid()
            plt.legend()
print('======== Finish plotting ========')

#%% Time plots for one sensor, one subject, different activity (randomly selected).
for s in sensors:
    for p in pos:
        for d in directions:
            plt.figure(figsize=(7,4))
            print('======== Plotting sensor ' + s + p + d + ' ========')
            for i in randomAct:
                plt.plot(subject1[subject1['label']==i].reset_index(drop=True)[s + p + d], alpha= .5, label = f'{activity_map[i]}')
            plt.title(s + p + d)
            plt.grid()
            plt.legend()
print('======== Finish plotting ========')

#%% Plot the spectrogram for each sensor (for subject1 only)
directions = ['x', 'y', 'z']
pos = ['_la_', '_ra_']
sensors = ['acc', 'gyr', 'mag']
for s in sensors:
    for p in pos:
        for d in directions:
            # plt.subplot(211)
            # plt.plot(subject1[s + p + d])
            plt.figure(figsize=(14,4))
            plt.subplot()
            powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(subject1[s + p + d], Fs=50, cmap="jet")
            plt.plot(time1*60, subject1['label'], color = 'red')
            plt.xlabel('Time (secs)')
            plt.ylabel('Frequency')
            plt.colorbar(imageAxis)
            plt.clim(-50, 50) 
            plt.title(s + p + d)
            plt.show()
# plot chest acc sensor
for d in directions:
    plt.figure(figsize=(7,4))
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(subject1['acc_ch_' + d], Fs=50, cmap="jet")
    plt.plot(time1*60, subject1['label'], color = 'red')
    plt.xlabel('Time (secs)')
    plt.ylabel('Frequency')
    plt.colorbar(imageAxis)
    plt.clim(-50, 50) 
    plt.title('acc_ch_' + d)
    plt.show()
     