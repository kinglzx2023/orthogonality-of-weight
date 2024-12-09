#!/usr/bin/env python
# coding: utf-8



import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
import sys
import adabound
torch.cuda.set_device(1)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine

input_size = 784
batch_size = 512
num_epochs = 50
learning_rate = 0.001
hidden_size = 500
number_H =5


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(seed) 

address =''
address_1 = address+str(seed)+'_'+'.txt'
address_2 = address+str(seed)+'_'+'.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')



def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out
def Gram_matrix_row(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix
def Gram_matrix_column(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    return Gram_matrix
def cos_similarity(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"Parameter name: {name}"+'  ')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+','+ str(mean_cos_sim_column)+'\n')
    print('='*50)


train_datasets = dsets.MNIST(root = '', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = '', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)

class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        self.norm = nn.BatchNorm1d(hidden)

        

    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        x = self.linear(x)
        x = self.r(x)
        for i in  range(number_H):
            x = self.linearH[i](x)
            x = self.r(x) 
        out = self.out(x)
        return out


class AdaHB(torch.optim.Optimizer):
    def __init__(self, params, eta=0.1, mu=0.9, epsilon=1e-8, a_t=1):
        defaults = dict(eta=eta, mu=mu, epsilon=epsilon, a_t=a_t)
        super(AdaHB, self).__init__(params, defaults)
        
        self.A = 0  # A_t = A_{t-1} + a_t, initialized to 0
        self.m = None  # m_t = momentum term
        self.v = None  # v_t = squared gradient accumulator

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Update parameters
        for group in self.param_groups:
            eta = group['eta']
            mu = group['mu']
            epsilon = group['epsilon']
            a_t = group['a_t']
            
            # Initialize v and m if first step
            if self.v is None:
                self.v = [torch.zeros_like(p, device='cuda') for p in group['params']]  # Move to GPU
                self.m = [torch.zeros_like(p, device='cuda') for p in group['params']]  # Move to GPU

            self.A += a_t  # A_t = A_{t-1} + a_t
            a_bar_t = self.A / (len(self.param_groups) + 1)  # A_t / t, assuming one step here

            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                
                grad = param.grad
                # Update v_t
                self.v[i] += a_t * grad.pow(2)
                
                # Update m_t (momentum)
                v_sqrt = torch.sqrt(self.v[i] / a_bar_t + epsilon)
                self.m[i] = mu * self.m[i] - eta * grad / v_sqrt
                
                # Update parameters
                param.data += self.m[i]

        return loss

if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()


optimizer = AdaHB(model.parameters(), eta=0.0001, mu=0.9, epsilon=1e-8, a_t=0.001)

optimizer.zero_grad()


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        #features = features.to(device)
        features = Variable(features.view(-1, 28*28)).cuda()
        
        #targets = targets.to(device)
        targets = Variable(targets).cuda()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs= model(images)
        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        
        
        loss.backward()
        optimizer.step()
        loss_out=round(loss.item(),4)
        if (i+1) % 40 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
    text_accuracy = round(compute_accuracy(model, test_loader).item(),4)
    print(str(text_accuracy)+'  '+str(loss_out))
    file2.writelines(str(text_accuracy)+'  '+str(loss_out)+'\n')

file1.writelines('epoch:'+str(epoch)+'\n')
for name, param in model.named_parameters():   
    if name == 'linearH.0.weight':
        cos_similarity(name)
    if name == 'linearH.1.weight':
        cos_similarity(name)
    if name == 'linearH.2.weight':
        cos_similarity(name)


file1.close() 
file2.close()






