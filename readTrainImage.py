import pandas as pd
import numpy as np
import os
import cv2
import torch
from skimage.feature import daisy
from skimage.feature import CENSURE
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage import feature
from skimage.color import rgb2gray
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)

from skimage.filters import gaussian



df = pd.DataFrame(columns=['image','breed'])
directoryTrainPath = '/Users/enyalos/522/project/project/data/trainBreed'
directoryTrain = os.listdir(directoryTrainPath)
breedLabels = pd.read_csv('/Users/enyalos/522/project/project/data/preTrainLabels/trainLabelsBreed.csv')
breedInts = pd.read_csv('/Users/enyalos/522/project/project/data/preTrainLabels/breedsToInts.csv')




#airedale
index = list(breedInts.index[breedInts['breed'] == 'airedale'])[0]
print(breedInts.loc[index, 'id'])


count = 0
for i in range(len(directoryTrain)):
    filename = directoryTrain[i]
    filePath = '/Users/enyalos/522/project/project/data/trainBreed/' + filename
    img = cv2.imread(filePath)
    img = rgb2gray(img)
    img = cv2.resize(img, (150,150))

    detector = CENSURE()
    detector.detect(img)
    keypoint = detector.keypoints

    #img = gaussian(img, sigma=1)

    for point in range(len(keypoint)):
        row = keypoint[point][0]
        column = keypoint[point][1]
        img[row][column] = 255


    mirrorImage = cv2.flip(img,0)
    img = img.tolist()
    mirrorImage = mirrorImage.tolist()

    splitFileName = filename.split('.')
    index = list(breedLabels.index[breedLabels['id'] == splitFileName[0]])[0]
    breedClass = breedLabels.loc[index,'breed']
    breedDF = list(breedInts.index[breedInts['breed'] == breedClass])[0]
    breed = breedInts.loc[breedDF, 'id']
    df.loc[count, 'breed'] = breed
    df.loc[count, 'image'] = img
    count = count + 1
    df.loc[count, 'breed'] = breed
    df.loc[count, 'image'] = mirrorImage
    count = count + 1



dogLabels = pd.DataFrame(columns=['breed'])
print("dogLabels");
dogLabels['breed'] = df['breed']
dogLabels.to_csv("trainBreedLabels.csv", index=False)

trainTensor = torch.Tensor(df['image'])
trainTensor = torch.unsqueeze(trainTensor, 1)
torch.save(trainTensor, 'trainTensor.pt')
