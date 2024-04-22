
import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
torch.cuda.set_device(0)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine


input_size = 784
batch_size = 512
num_epochs = 20
learning_rate = 0.001
hidden_size = 500
number_H =5
probability = 0


address =''
address_1 = address+'cos_sim.txt'
address_2 = address+'acc.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')

def seed_torch(seed=42):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
seed_torch()

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

train_datasets = dsets.MNIST(root = './Datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = './Datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)


class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.GELU()
        self.dropout = nn.Dropout(probability)
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        self.norm = nn.BatchNorm1d(hidden)
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.linearH[1].weight, 0.1)
        nn.init.constant_(self.linearH[1].bias, 0.1)

    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        x = self.linear(x)
        x = self.r(x)
        
        for i in  range(number_H):
            x_in = x
            x = self.linearH[i](x)
            x = self.r(x) 
            x = self.norm(x+x_in)

        out = self.out(x)
        return out



if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
optimizer.zero_grad()



file1.writelines('Initial parameters'+'\n')
for name, param in model.named_parameters():
    if name == 'linear.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        
        cos_sim_row_1st = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_1st = Gram_matrix_row(param.cpu().data)
        Gram_column_1st = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_init_input_layer.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_init_input_layer.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_input_layer.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_input_layer.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines(str(mean_cos_sim_row_1st)+','+ str(mean_cos_sim_column_1st)+'\n')
        
        print('='*50)
    if name == 'linearH.0.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        
        cos_sim_row_H0 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_H0 = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_H0 = Gram_matrix_row(param.cpu().data)
        Gram_column_H0 = Gram_matrix_column(param.cpu().data)
        
        mean_cos_sim_row_H0 = Mean(cos_sim_row_H0)
        mean_cos_sim_column_H0 = Mean(cos_sim_column_H0)
        np.savetxt(address+'Gram_row_init_layer_0.txt', Gram_row_H0, fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer_0.txt', Gram_column_H0, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer_0.txt', cos_sim_row_H0, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer_0.txt', cos_sim_column_H0, fmt='%.3f')
        print(mean_cos_sim_row_H0, mean_cos_sim_column_H0)
        file1.writelines(str(mean_cos_sim_row_H0)+','+ str(mean_cos_sim_column_H0)+'\n')
        
        print('='*50)
    if name == 'linearH.1.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_H1 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_H1 = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_H1 = Gram_matrix_row(param.cpu().data)
        Gram_column_H1 = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_H1 = Mean(cos_sim_row_H1)
        mean_cos_sim_column_H1 = Mean(cos_sim_column_H1)
        np.savetxt(address+'Gram_row_init_layer_1.txt', Gram_row_H1, fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer_1.txt', Gram_column_H1, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer_1.txt', cos_sim_row_H1, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer_1.txt', cos_sim_column_H1, fmt='%.3f')
        print(mean_cos_sim_row_H1, mean_cos_sim_column_H1)
        file1.writelines(str(mean_cos_sim_row_H1)+','+ str(mean_cos_sim_column_H1)+'\n')
        print('='*50)
    if name == 'linearH.2.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")    
        cos_sim_row_H2 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_H2  = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_H2  = Gram_matrix_row(param.cpu().data)
        Gram_column_H2  = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_H2  = Mean(cos_sim_row_H2 )
        mean_cos_sim_column_H2  = Mean(cos_sim_column_H2 )
        np.savetxt(address+'Gram_row_init_layer_2.txt', Gram_row_H2 , fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer_2.txt', Gram_column_H2 , fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer_2.txt', cos_sim_row_H2 , fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer_2.txt', cos_sim_column_H2 , fmt='%.3f')
        print(mean_cos_sim_row_H2 , mean_cos_sim_column_H2 )
        file1.writelines(str(mean_cos_sim_row_H2 )+','+ str(mean_cos_sim_column_H2 )+'\n')

    if name == 'linearH.3.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_H3 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_H3  = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_H3  = Gram_matrix_row(param.cpu().data)
        Gram_column_H3  = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_H3  = Mean(cos_sim_row_H3 )
        mean_cos_sim_column_H3  = Mean(cos_sim_column_H3 )
        np.savetxt(address+'Gram_row_init_layer_3.txt', Gram_row_H3 , fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer_3.txt', Gram_column_H3 , fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer_3.txt', cos_sim_row_H3 , fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer_3.txt', cos_sim_column_H3 , fmt='%.3f')
        print(mean_cos_sim_row_H3 , mean_cos_sim_column_H3 )
        file1.writelines(str(mean_cos_sim_row_H3 )+','+ str(mean_cos_sim_column_H3 )+'\n')
    if name == 'linearH.4.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_H4 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_H4  = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_H4  = Gram_matrix_row(param.cpu().data)
        Gram_column_H4  = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_H4  = Mean(cos_sim_row_H4 )
        mean_cos_sim_column_H4  = Mean(cos_sim_column_H4 )
        np.savetxt(address+'Gram_row_init_layer_3.txt', Gram_row_H4 , fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer_3.txt', Gram_column_H4 , fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer_3.txt', cos_sim_row_H4 , fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer_3.txt', cos_sim_column_H4 , fmt='%.3f')
        print(mean_cos_sim_row_H4 , mean_cos_sim_column_H4 )
        file1.writelines(str(mean_cos_sim_row_H4 )+','+ str(mean_cos_sim_column_H4 )+'\n')
        print('='*50)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        features = Variable(features.view(-1, 28*28)).cuda()
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



file1.writelines('Trained parameters'+'\n')
for name, param in model.named_parameters():
    if name == 'linear.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_end_1st = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_end_1st = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_end_1st = Gram_matrix_row(param.cpu().data)
        Gram_column_end_1st = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
        mean_cos_sim_column_end_1st = Mean(cos_sim_column_end_1st)
        np.savetxt(address+'Gram_row_trained_input_layer.txt', Gram_row_end_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_input_layer.txt', Gram_column_end_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_input_layer.txt', cos_sim_row_end_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_input_layer.txt', cos_sim_column_end_1st, fmt='%.3f')
        print(mean_cos_sim_row_end_1st, mean_cos_sim_column_end_1st)
        file1.writelines(str(mean_cos_sim_row_end_1st)+','+ str(mean_cos_sim_column_end_1st)+'\n')
        print('='*50)
    if name == 'linearH.0.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_end_H0 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_end_H0 = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_end_H0 = Gram_matrix_row(param.cpu().data)
        Gram_column_end_H0 = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_end_H0 = Mean(cos_sim_row_end_H0)
        mean_cos_sim_column_end_H0 = Mean(cos_sim_column_end_H0)
        np.savetxt(address+'Gram_row_trained_layer_0.txt', Gram_row_end_H0, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer_0.txt', Gram_column_end_H0, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer_0.txt', cos_sim_row_end_H0, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer_0.txt', cos_sim_column_end_H0, fmt='%.3f')
        print(mean_cos_sim_row_end_H0, mean_cos_sim_column_end_H0)
        file1.writelines(str(mean_cos_sim_row_end_H0)+','+ str(mean_cos_sim_column_end_H0)+'\n')
        print('='*50)
    if name == 'linearH.1.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_end_H1 = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_end_H1  = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_end_H1  = Gram_matrix_row(param.cpu().data)
        Gram_column_end_H1  = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_end_H1  = Mean(cos_sim_row_end_H1 )
        mean_cos_sim_column_end_H1  = Mean(cos_sim_column_end_H1 )
        np.savetxt(address+'Gram_row_trained_layer_1.txt', Gram_row_end_H1 , fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer_1.txt', Gram_column_end_H1 , fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer_1.txt', cos_sim_row_end_H1 , fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer_1.txt', cos_sim_column_end_H1 , fmt='%.3f')
        print(mean_cos_sim_row_end_H1 , mean_cos_sim_column_end_H1 )
        file1.writelines(str(mean_cos_sim_row_end_H1 )+','+ str(mean_cos_sim_column_end_H1 )+'\n')
        print('='*50)
    if name == 'linearH.2.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_end_H2  = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_end_H2 = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_end_H2 = Gram_matrix_row(param.cpu().data)
        Gram_column_end_H2 = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_end_H2 = Mean(cos_sim_row_end_H2)
        mean_cos_sim_column_end_H2 = Mean(cos_sim_column_end_H2)
        np.savetxt(address+'Gram_row_trained_layer_2.txt', Gram_row_end_H2, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer_2.txt', Gram_column_end_H2, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer_2.txt', cos_sim_row_end_H2, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer_2.txt', cos_sim_column_end_H2, fmt='%.3f')
        print(mean_cos_sim_row_end_H2, mean_cos_sim_column_end_H2)
        file1.writelines(str(mean_cos_sim_row_end_H2)+','+ str(mean_cos_sim_column_end_H2)+'\n')
        print('='*50)
    if name == 'linearH.3.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_end_H3  = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_end_H3 = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_end_H3 = Gram_matrix_row(param.cpu().data)
        Gram_column_end_H3 = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_end_H3 = Mean(cos_sim_row_end_H3)
        mean_cos_sim_column_end_H3 = Mean(cos_sim_column_end_H3)
        np.savetxt(address+'Gram_row_trained_layer_2.txt', Gram_row_end_H3, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer_2.txt', Gram_column_end_H3, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer_2.txt', cos_sim_row_end_H3, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer_2.txt', cos_sim_column_end_H3, fmt='%.3f')
        print(mean_cos_sim_row_end_H3, mean_cos_sim_column_end_H3)
        file1.writelines(str(mean_cos_sim_row_end_H3)+','+ str(mean_cos_sim_column_end_H3)+'\n')       
        print('='*50)
    if name == 'linearH.4.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
        print(f"Parameter value: {param.data.size()}")
        cos_sim_row_end_H4  = cos_similarity_matrix_row(param.cpu().data)
        cos_sim_column_end_H4 = cos_similarity_matrix_column(param.cpu().data)
        Gram_row_end_H4 = Gram_matrix_row(param.cpu().data)
        Gram_column_end_H4 = Gram_matrix_column(param.cpu().data)
        mean_cos_sim_row_end_H4 = Mean(cos_sim_row_end_H4)
        mean_cos_sim_column_end_H4 = Mean(cos_sim_column_end_H4)
        np.savetxt(address+'Gram_row_trained_layer_2.txt', Gram_row_end_H4, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer_2.txt', Gram_column_end_H4, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer_2.txt', cos_sim_row_end_H4, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer_2.txt', cos_sim_column_end_H4, fmt='%.3f')
        print(mean_cos_sim_row_end_H4, mean_cos_sim_column_end_H4)
        file1.writelines(str(mean_cos_sim_row_end_H4)+','+ str(mean_cos_sim_column_end_H4)+'\n')       
        print('='*50)
    
file1.close() 
file2.close()






