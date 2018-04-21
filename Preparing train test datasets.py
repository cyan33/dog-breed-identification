# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:31:53 2018

@author: jibin
"""

import pandas as pd
import numpy as np
import os
import cv2
import torch

directoryTrainPath = 'D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/Data/train/Train - Resize/train'
directoryTrain = os.listdir(directoryTrainPath)
breedLabels = pd.read_csv('D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/Data/Labels/labels.csv')

X_data = []
Y_data = []

for i in range(len(directoryTrain)):
    filename = directoryTrain[i]
    filePath = 'D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/Data/train/Train - Resize/train/' + filename
    img = cv2.imread(filePath, 0) 
    img = cv2.resize(img,(200,200))
    X_data.append (img)
    splitFilename = filename.split('.')
    Y_data.append(breedLabels.loc[(breedLabels["id"]==splitFilename[0]),["breed_id"]].as_matrix()[0][0])


    
    





df = pd.DataFrame(columns=['image','breed'])
directoryTrainPath = 'D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/Data/train/Train - Resize/train'
directoryTrain = os.listdir(directoryTrainPath)
breedLabels = pd.read_csv('D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/CNN - Matt Farver/CNN/trainLabelsBreed.csv')
breedInts = pd.read_csv('D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/CNN - Matt Farver/CNN/breedsToInts.csv')

for i in range(len(directoryTrain)):
    filename = directoryTrain[i]
    filePath = 'D:/Spring\'18/CSC 522 - Automated Learning and Data Analysis/Projects and Assignments/Data/train/Train - Resize/train/' + filename
    img = cv2.imread(filePath, 0) #Reading each image from the folder
    img = cv2.resize(img, (200,200)) #Resizing each image to 200*200 (This is not a crop function)
    img = img.tolist() #Converting the image into a list
    splitFileName = filename.split('.') #Splitting the name of the image 
    index = list(breedLabels.index[breedLabels['id'] == splitFileName[0]])[0] #Getting the index associated with the image
    breedClass = breedLabels.loc[index,'breed'] #Identifying the breed associated with that particular image
    breedDF = list(breedInts.index[breedInts['breed'] == breedClass])[0]
    breed = breedInts.loc[breedDF, 'id']
    df.loc[i, 'breed'] = breed
    df.loc[i, 'image'] = img
