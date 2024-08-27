import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from LeNet import LeNet
import wandb
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28))])
Dataset_train = torchvision.datasets.ImageFolder(root='C:/Users/Dumin/Desktop/fishnet/fishnet_data_enhance/train',transform=transform)
Dataset_test = torchvision.datasets.ImageFolder(root='C:/Users/Dumin/Desktop/fishnet/dataset/val',transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=Dataset_train, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=Dataset_test, batch_size=4, shuffle=False)

wandb.login(key='ddd005d13a9704b2f25fd1c6ace472b6ca714fc2')
wandb.config.update({
    'batch_size':4,
    'num_classes':5768,
    'epochs':20
})
wandb.init(
    project='ResNet50',
    config=wandb.config,
    name='LeNet'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features,num_classes)
model = LeNet(3,5)
model = nn.DataParallel(model,device_ids=[0]).to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    for epoch in range(20):
        wandb.log({
            'epoch':epoch+1
        })
        correct = 0
        total = 0
        train_loss_total = 0
        test_loss_total = 0
        for X,y in train_loader:
            X,y = X.to(device),y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            train_loss = loss(y_pred, y)
            train_loss_total += train_loss
            total+=y.shape[0]
            train_loss.backward()
            _,predicted = torch.max(y_pred,dim=1)
            correct+=(predicted==y).sum().item()
            optimizer.step()
            wandb.log({
                'train_loss': train_loss
            })
        train_acc = correct / total
        wandb.log({
                'train_loss_total':train_loss_total,
                'train_acc':train_acc
            })

        correct = 0
        total = 0
        for X,y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss = loss(y_pred, y)
            test_loss_total += test_loss
            total += y.shape[0]
            _, predicted = torch.max(y_pred, dim=1)
            correct += (predicted == y).sum().item()
            wandb.log({
                'test_loss':test_loss,
            })
        test_acc = correct / total
        wandb.log({
            'test_acc': test_acc,
            'test_loss_total':test_loss_total
        })




