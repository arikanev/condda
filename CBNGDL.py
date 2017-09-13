
# coding: utf-8

# In[68]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.utils.data.dataset
import pandas as pd
from skimage import io
import os
from PIL import Image

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)


transformMnistm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
                                       root_dir = 'Downloads/mnist_m/mnist_m_train')

mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)

mnistmTestSet = mnistmTestingDataset(text_file ='Downloads/mnist_m/mnist_m_test_labels.txt',
                                       root_dir = 'Downloads/mnist_m/mnist_m_test')

mnistmTestLoader = torch.utils.data.DataLoader(mnistmTestSet,batch_size=16,shuffle=True, num_workers=2)



class mnistmTestingDataset(torch.utils.data.Dataset):
    
    def __init__(self,text_file,root_dir,transform=transformMnistm):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all test images
        """
        self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.name_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels}
        
        return sample



    
class mnistmTrainingDataset(torch.utils.data.Dataset):
    
    def __init__(self,text_file,root_dir,transform=transformMnistm):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.name_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels}
        
        return sample
        
        
        
#test trainloader ouput + batchsize        
for i_batch,sample_batched in enumerate(mnistmTrainLoader,0):
    print(i_batch,sample_batched['image'],sample_batched['labels'])
    if i_batch == 0:
        break
        
#test testloader output + batchsize     
for i_batch,sample_batched in enumerate(mnistmTestLoader,0):
    print(i_batch,sample_batched['image'],sample_batched['labels'])
    if i_batch == 0:
        break     
        
        
        
        
class classifierNet(nn.Module):
    def __init__(self,featureExtractNet,labelPredictNet):
        super(classifierNet,self).__init__()
        self.featureExtractNet = featureExtractNet
        self.labelPredictNet = labelPredictNet
    def forward(self,x):
        x = self.featureExtractNet(x)
        x = self.labelPredictNet(x)
        return x
    
class domainClassifierNet(nn.Module):
    def __init__(self,featureExtractNet,domainPredictNet):
        super(domainClassifierNet,self).__init__()
        self.featureExtractNet = featureExtractNet
        self.domainPredictNet = domainPredictNet
    def forward(self,x):
        x = self.featureExtractNet(x)
        x = self.domainPredictNet(x)
        
class featureExtractNet(nn.Module):
    def __init__(self):
        super(featureExtractNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,48,5,2)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x
    
class labelPredictNet(nn.Module):
    def __init__(self):
        super(labelPredictNet,self).__init__()
        self.fc1 = nn.Linear(48*7*7,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)
    def forward(self,x):
        x = x.view(-1,48*7*7)
        x = F.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return x

class domainPredictNet(nn.Module):
    def __init__(self, lambda_=0):
        super(domainPredictNet,self).__init__()
        self.grl = GRL(lambda_=lambda_)
        self.fc1 = nn.Linear(48*7*7,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,1)
    def forward(self,x):
        x = self.grl(x)
        x = x.view(-1,48*7*7)
        x = F.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return x

class GRL(torch.autograd.Function):
    def __init__(self,lambda_):
        super(GRL,self).__init__()
        self.lambda_ = lambda_
    def forward(input):
        self.save_for_backward(input)
        return input
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return - self.lambda_*grad_input


#featureExtractNet = featureExtractNet()
#labelPredictNet = labelPredictNet()
#domainPredictNet = domainPredictNet()
#classifierNet = classifierNet(featureExtractNet,labelPredictNet)
#domainClassifierNet = domainClassifierNet(featureExtractNet,domainPredictNet)

#params_list_dict = [{"params":labelPredictNet.parameters()},
#                   {"params":domainPredictNet.parameters()},
#                   {"params":featureExtractNet.parameters()}]

#criterion = nn.CrossEntropyLoss()






# In[ ]:



