#LeNet
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time

batch_size = 256

def data_loader(batch_size):
    trans = transforms.ToTensor()
    trans = transforms.Compose([trans,transforms.Normalize((0.1307,),(0.3081,))])

    train_data = torchvision.datasets.FashionMNIST(root='../data',train=True,transform=trans,download=True)
    test_data = torchvision.datasets.FashionMNIST(root='../data',train=False,transform=trans,download=True)

    return data.DataLoader(train_data,batch_size=batch_size,shuffle=True),data.DataLoader(test_data,batch_size=batch_size,shuffle=False)

train_iter,test_iter = data_loader(batch_size)

class Reshape(torch.nn.Module):
    def forward(self,X):
        return X.view((-1,1,28,28))

net = nn.Sequential(
    Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),nn.ReLU(),
    nn.MaxPool2d(2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
    nn.MaxPool2d(2,stride=2),nn.Flatten(),
    nn.Linear(400,120),nn.ReLU(),#nn.Dropout(0.5),
    nn.Linear(120,84),nn.ReLU(),#nn.Dropout(0.2),
    nn.Linear(84,10)
)

class LeNet(nn.Module):
    def __init__(self,in_channels,num_classes,):
        super(LeNet,self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.in_channels,6,kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(6,16,kernel_size=5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,self.num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.view((-1,self.in_channels,28,28))
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.reshape(-1,16*5*5)
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return self.softmax(x)

X = torch.randn(size=(1,1,28,28))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

#net = net()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params':[param for name,param in net.named_parameters() if 'bias' not in name],'weight_decay':0.0},
    {'params':[param for name,param in net.named_parameters() if 'bias' in name],'weight_decay':0.0}
],lr = 0.0085)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

epochs=10
train_losses = []
test_losses = []
if __name__=='__main__':
    torch.multiprocessing.freeze_support()
    start_time0 = time.time()
    sum_time = 0
    for epoch in range(epochs):
        strat_time = time.time()
        train_l_sum = 0
        test_l_sum = 0
        correct = 0
        total = 0
        for x,y in train_iter:
            x,y = x.to(device),y.to(device)
            y_hat = net(x)
            train_l = loss(y_hat,y)
            optimizer.zero_grad()
            train_l.backward()
            optimizer.step()
            train_l_sum += train_l.item()
        train_losses.append(train_l_sum/len(train_iter))

        with torch.no_grad():
            for x,y in test_iter:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                test_l = loss(outputs,y)
                test_l_sum += test_l.item()
                _,predicted = torch.max(outputs,1)
                total += y.size(0)
                correct += (y == predicted).sum().item()
            test_losses.append(test_l_sum/len(test_iter))
        end_time = time.time()
        train_time = end_time - strat_time
        sum_time += train_time
        print(f'epoch{epoch+1},train_loss = {train_l_sum/len(train_iter):.4},test_loss = {test_l_sum/len(test_iter):.4},accuracy = {100*correct/total:.4}%')
        print(f'单次用时:{train_time:.4},总用时:{sum_time:.4}')

    ax = plt.subplot()

    ax.plot(train_losses,label='train_losses')
    ax.plot(test_losses,label='test_losses')
    ax.legend()
    plt.show()

















