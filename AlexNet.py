#AlexNet
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import dataset

def data_loader(batch_size,resize):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
        #trans.append(transforms.Normalize((0.1307,),(0.3081,)))

    trans = transforms.Compose(trans)

    train_data = torchvision.datasets.FashionMNIST(root='../data',train=True,transform=trans,download=True)
    test_data = torchvision.datasets.FashionMNIST(root='../data',train=False,transform=trans,download=True)

    return data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4),data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=4)


class Reshape(nn.Module):
    def forward(self,X):
        return X.view(-1,3,224,224)
""""
def AlexNet(num_class):
    return nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(4096, num_class),
                    nn.Softmax(dim=1)
                )
"""

class AlexNet(nn.Module):
    def __init__(self, num_class):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_class)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.conv5(self.relu(self.conv4(self.relu(self.conv3(x)))))
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1,6400)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x


def layer_vision_fc(net):
    X = torch.randn(size=(1,3,224,224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)


#batch_size = 128
#train_iter,test_iter = data_loader(batch_size,resize=224)

def main(num_calss,epochs,train_iter,test_iter,layer_vision=False):
    net = AlexNet(num_class=num_calss)
    net.load_state_dict(torch.load('D:\PyCharm\Py_Projects\model_weights.pth'))
    loss = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam([
    #    {'params':[param for name,param in net.named_parameters() if 'bias' not in name],'weight_decay':0.0},
    #    {'params':[param for name,param in net.named_parameters() if 'bias' in name],'weight_decay':0.0}
    #],lr = 0.01)
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    if layer_vision==True:
        layer_vision_fc(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    epochs=epochs
    train_losses = []
    test_losses = []
    labels = []


    torch.multiprocessing.freeze_support()
    start_time0 = time.time()
    sum_time = 0
    for epoch in range(epochs):
        strat_time = time.time()
        train_l_sum = 0
        test_l_sum = 0
        correct = 0
        total = 0

        net.train()
        for x,y in train_iter:
            x,y = x.to(device),y.to(device)
            y_hat = net(x)
            train_l = loss(y_hat,y)
            optimizer.zero_grad()
            train_l.backward()
            optimizer.step()
            train_l_sum += train_l.item()
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (y == predicted).sum().item()
        if len(train_iter)==0:
            train_losses.append(train_l_sum/1e-30)
        else:
            train_losses.append(train_l_sum / len(train_iter))
        end_time = time.time()
        train_time = end_time - strat_time
        sum_time += train_time
        if len(train_iter)==0 and total!=0:
            print(
                f'epoch{epoch + 1},train_loss = {train_l_sum / 1e-30:.4},train_accuracy = {100 * correct / total:.8}%')
        if len(train_iter)!=0 and total==0:
            print(f'epoch{epoch + 1},train_loss = {train_l_sum / len(train_iter):.4},train_accuracy = {100 * correct / 1e-30:.8}%')
        if len(train_iter)==0 and total==0:
            print(
                f'epoch{epoch + 1},train_loss = {train_l_sum / 1e-30:.4},train_accuracy = {100 * correct / 1e-30:.8}%')
        if len(train_iter) != 0 and total != 0:
            print(
                f'epoch{epoch + 1},train_loss = {train_l_sum / len(train_iter):.4},train_accuracy = {100 * correct / total:.8}%')
        print(f'单次用时:{train_time:.4},总用时:{sum_time:.4}')
        net.eval()
        with torch.no_grad():
            for x in test_iter:
                x = x.to(device)
                outputs = net(x)
                predict = torch.softmax(outputs, dim=1)
                probs, classes = torch.max(predict, dim=1)
                for idx, cla in enumerate(classes):
                    labels.append(cla.item())
        print(labels)
        end_time = time.time()
        train_time = end_time - strat_time
        sum_time += train_time

    ax = plt.subplot()

    ax.plot(train_losses,label='train_losses')
    ax.plot(test_losses,label='test_losses')
    ax.legend()
    plt.show()

    torch.save(net.state_dict(), 'model_weights.pth')
    return labels
