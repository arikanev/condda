
# coding: utf-8

# In[1]:


#Evan Racah 2017


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler as lrs
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np


# In[3]:


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


# In[4]:


# thanks for the example, Justin:
#https://github.com/jcjohnson/pytorch-examples/blob/master/autograd/two_layer_net_custom_function.py


# In[5]:


class GRL(torch.autograd.Function):
    """ 
    Gradient Reversal Layer (GRL) described in 
    (Ganin, et al.) https://arxiv.org/pdf/1409.7495.pdf
    """
    def __init__(self, lambda_):
        super(GRL,self).__init__()
        
        #set the lambda coefficient hyperparameter
        self.lambda_ = lambda_
        
    def forward(self, input):
        """ 
        In the forward pass it is just the identity function
        as specified by Ganin: 'During the forward propagation, 
        GRL acts as an identity transform.'
        """
        #cache input for backward
        self.save_for_backward(input)
        return input
    
    def backward(self, grad_output):
        """
        In the backward pass, it multiplies gradient
        by negative lambda (goes in opposite direction of what 
        would help the domain classifier)
        'During the backpropagation though, GRL takes the gradient 
        from the subsequent level, multiplies it by −λ 
        and passes it to the preceding layer' - Ganin
        """
        grad_input = grad_output.clone()
        return - self.lambda_ * grad_input


# In[6]:


class mnistFeatExtractor(nn.Module):
    def __init__(self):
        super(mnistFeatExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 48,5, padding=2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        return x
        
        


# In[7]:


class mnistDomainClassifier(nn.Module):
    def __init__(self, lambda_=0):
        super(mnistDomainClassifier, self).__init__()
        self.grl = GRL(lambda_=lambda_)
        self.fc1 = nn.Linear(48*7*7, 100)
        self.fc2 = nn.Linear(100, 1)
    def forward(self,x):
        x = self.grl(x)
        x = x.view(-1, 48*7*7)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
            


# In[8]:


class mnistClassifier(nn.Module):
    def __init__(self):
        super(mnistClassifier, self).__init__()
        
        self.fc1 = nn.Linear(48*7*7, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self, x):
        x = x.view(-1, 48*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


# In[9]:


class classifNet(nn.Module):
    def __init__(self, mfe, mcls):
        super(classifNet,self).__init__()
        self.mfe = mfe
        self.mcls = mcls
    def forward(self,x):
        x = self.mfe(x)
        x = self.mcls(x)
        return x
        
    


# In[10]:


class domainClassifNet(nn.Module):
    def __init__(self, mfe, mdc):
        super(domainClassifNet,self).__init__()
        self.mfe = mfe
        self.mdc = mdc
    def forward(self,x):
        x = self.mfe(x)
        x = self.mdc(x)
        return x


# In[11]:


mfe = mnistFeatExtractor()
mcls = mnistClassifier()
mdc = mnistDomainClassifier()


# In[12]:


cls_net = classifNet(mfe, mcls)


# In[13]:


dc_net = domainClassifNet(mfe, mdc)


# In[14]:


params_list_dict = [{"params": mcls.parameters()},
                    {"params": mdc.parameters()},
                    {"params": mfe.parameters() }]


# In[15]:


criterion = nn.CrossEntropyLoss()


# In[16]:


def lr_annealing_rule(cur_epoch, last_epoch):
    mu_o = 0.01
    alpha = 10
    beta = 0.75
    p = float(cur_epoch) / last_epoch #I think this is what they mean by p -> hacky
    denominator = (1 + alpha * p) ** beta
    return mu_o / denominator

total_num_epochs = 10
lr_lambda = lambda cur_epoch: lr_annealing_rule(cur_epoch, total_num_epochs)


# In[17]:


def lambda_annealing_rule(cur_epoch, last_epoch):
    p = float(cur_epoch) / last_epoch #I think this is what they mean by p -> hacky
    gamma = 10.
    lambdap = (2 / (1 + np.exp(-gamma*p) )) -1.
    return lambdap
    


# In[18]:


#lr_0 = lr_lambda(0)

optimizer = optim.SGD(params=params_list_dict, momentum=0.9, lr=0.1)

#scheduler = lrs.LambdaLR(optimizer,lr_lambda=lr_lambda)


# In[20]:


for epoch in range(total_num_epochs):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, cls_labels = data

        #normal mnist is all domain 1 for now
        domain_labels = torch.zeros(cls_labels.size(0))
        domain_labels = domain_labels.long()

        
        # wrap them in Variable
        inputs, cls_labels, domain_labels = Variable(inputs), Variable(cls_labels), Variable(domain_labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        cls_outputs = cls_net(inputs)
        dc_outputs = dc_net(inputs)
        loss = criterion(cls_outputs, cls_labels) + criterion(dc_outputs, domain_labels)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        #todo update lambda here
        dc_net.lambda_ = lambda_annealing_rule(epoch, total_num_epochs)

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


print('Finished Training')


# In[ ]:




