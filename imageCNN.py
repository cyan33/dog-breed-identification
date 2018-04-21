import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import string
import csv

trainingBreedLabels = pd.read_csv('trainBreedLabels.csv')
trainTensors = torch.load('trainTensor.pt')

testBreedLabels = pd.read_csv('testBreedLabels.csv')
testTensors= torch.load('testTensor.pt')

def addClassifier(tensorData, labelData):
    classified = []
    for index in range(len(labelData)):
        tensor = tensorData[index]
        breed = labelData.loc[index, 'breed']
        tupleData = (tensor, breed)
        classified.append(tupleData)
    return classified

alldata = addClassifier(trainTensors, trainingBreedLabels)
print(len(alldata))

#trainData = alldata[:9000]
#testData = alldata[9001:]
trainData = alldata[:17000]
testData = alldata[17001:]
print(trainData[1])


num_epochs = 10
batch_size = 75
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=12, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=10, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2)

        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(2)

        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0)
        self.relu5 = nn.ReLU()

        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0)
        self.relu6 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(2)
        #self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0)
        #self.relu2 = nn.ReLU()
        #self.maxpool2 = nn.MaxPool2d(2)
        #121 is the number of classes
        self.fc1 = nn.Linear(256*11*11, 120)
        #self.fc2 = nn.Linear(1000, 120)




    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.cnn3(x)
        x = self.relu3(x)
        x = self.cnn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.cnn5(x)
        x = self.relu5(x)
        x = self.cnn6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)


        print(x.size())

        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        #x = self.fc2(x)

        return x

model = CNNModel()
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
#criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
epoc = 0
for epoch in range(num_epochs):
    for i, (image, labels) in enumerate(train_loader):

        image = Variable(image)
        labels = Variable(labels)
        #image = image.view(-1, *64)
        optimizer.zero_grad()

        outputs = model(image)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        iter += 1
        if iter % 10 == 0:
            print(iter)

        if iter % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, iter * len(image), len(train_loader.dataset),
                           100. * iter / len(train_loader), loss.data[0]))

    epoc += 1
    print("Epoch Done " + str(epoc))


test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    #data = data.view(-1, 64 * 64)
    net_out = model(data)
    # sum up batch loss
    test_loss += criterion(net_out, target).data[0]
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
