import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
torch.cuda.set_device(0)

Batch_size = 128
num_epochs = 20
learning_rate = 0.001

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



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for name, param in model.named_parameters():
    if name == 'layer1.0.conv1.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data[:,:,1,1].size()}")
        Param = param.data[:,:,1,1]
        cos_sim_row_1st = cos_similarity_matrix_row(Param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(Param.cpu().data)
        Gram_row_1st = Gram_matrix_row(Param.cpu().data)
        Gram_column_1st = Gram_matrix_column(Param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_init_layer1.0.conv1.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer1.0.conv1.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer1.0.conv1.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer1.0.conv1.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines('layer1.0.conv1_row:'+str(mean_cos_sim_row_1st)+'  '+'layer1.0.conv1_column:'+str(mean_cos_sim_column_1st)+'\n')
    if name == 'layer2.0.conv1.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data[:,:,1,1].size()}")
        Param = param.data[:,:,1,1]
        cos_sim_row_1st = cos_similarity_matrix_row(Param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(Param.cpu().data)
        Gram_row_1st = Gram_matrix_row(Param.cpu().data)
        Gram_column_1st = Gram_matrix_column(Param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_init_layer2.0.conv1.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer2.0.conv1.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer2.0.conv1.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer2.0.conv1.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines('layer2.0.conv1_row:'+str(mean_cos_sim_row_1st)+'  '+'layer2.0.conv1_column:'+str(mean_cos_sim_column_1st)+'\n')
    if name == 'layer3.0.conv1.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data[:,:,1,1].size()}")
        Param = param.data[:,:,1,1]
        cos_sim_row_1st = cos_similarity_matrix_row(Param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(Param.cpu().data)
        Gram_row_1st = Gram_matrix_row(Param.cpu().data)
        Gram_column_1st = Gram_matrix_column(Param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_init_layer3.0.conv1.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_init_layer3.0.conv1.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_init_layer3.0.conv1.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_init_layer3.0.conv1.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines('layer3.0.conv1_row:'+str(mean_cos_sim_row_1st)+'  '+'layer3.0.conv1_column:'+str(mean_cos_sim_column_1st)+'\n')

file1.writelines('Trained parameters'+'\n')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = total_loss / len(trainloader)
    train_accuracy = correct / total

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    file2.writelines(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
for name, param in model.named_parameters():
    if name == 'linearH.2.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')
for name, param in model.named_parameters():
    if name == 'layer1.0.conv1.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data[:,:,1,1].size()}")
        Param = param.data[:,:,1,1]
        cos_sim_row_1st = cos_similarity_matrix_row(Param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(Param.cpu().data)
        Gram_row_1st = Gram_matrix_row(Param.cpu().data)
        Gram_column_1st = Gram_matrix_column(Param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_trained_layer1.0.conv1.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer1.0.conv1.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer1.0.conv1.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer1.0.conv1.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines('layer1.0.conv1_row:'+str(mean_cos_sim_row_1st)+'  '+'layer1.0.conv1_column:'+str(mean_cos_sim_column_1st)+'\n')
    if name == 'layer2.0.conv1.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data[:,:,1,1].size()}")
        Param = param.data[:,:,1,1]
        cos_sim_row_1st = cos_similarity_matrix_row(Param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(Param.cpu().data)
        Gram_row_1st = Gram_matrix_row(Param.cpu().data)
        Gram_column_1st = Gram_matrix_column(Param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_trained_layer2.0.conv1.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer2.0.conv1.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer2.0.conv1.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer2.0.conv1.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines('layer2.0.conv1_row:'+str(mean_cos_sim_row_1st)+'  '+'layer2.0.conv1_column:'+str(mean_cos_sim_column_1st)+'\n')
    if name == 'layer3.0.conv1.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data[:,:,1,1].size()}")
        Param = param.data[:,:,1,1]
        cos_sim_row_1st = cos_similarity_matrix_row(Param.cpu().data)
        cos_sim_column_1st = cos_similarity_matrix_column(Param.cpu().data)
        Gram_row_1st = Gram_matrix_row(Param.cpu().data)
        Gram_column_1st = Gram_matrix_column(Param.cpu().data)
        mean_cos_sim_row_1st = Mean(cos_sim_row_1st)
        mean_cos_sim_column_1st = Mean(cos_sim_column_1st)
        np.savetxt(address+'Gram_row_trained_layer3.0.conv1.txt', Gram_row_1st, fmt='%.3f')
        np.savetxt(address+'Gram_column_trained_layer3.0.conv1.txt', Gram_column_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_row_trained_layer3.0.conv1.txt', cos_sim_row_1st, fmt='%.3f')
        np.savetxt(address+'cos_sim_column_trained_layer3.0.conv1.txt', cos_sim_column_1st, fmt='%.3f')
        print(mean_cos_sim_row_1st, mean_cos_sim_column_1st)
        file1.writelines('layer3.0.conv1_row:'+str(mean_cos_sim_row_1st)+'  '+'layer3.0.conv1_column:'+str(mean_cos_sim_column_1st)+'\n')

file1.close()
file2.close()
print("Training finished.")
