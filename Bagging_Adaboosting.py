import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import os
import numpy as np
import torch
import math
from LeNet import LeNet


#error计算
def cal_error(weights,I):
    error = torch.sum(weights*I) / torch.sum(weights)
    return error

#alpha计算
def update_alpha(error,K=None):
    alpha = torch.log((1-error)/error) #+ math.log(1-K)
    return alpha

#权重更新
def update_weights(weights,predicted,labels):   #类别数K
    assert type(weights) == torch.Tensor
    assert len(predicted) == len(labels)
    #误差矩阵I
    I = (predicted != labels).float()
    error = cal_error(
        weights=weights,
        I=I
    )
    print('I',I)
    alpha = update_alpha(error=error)
    print('alpha',alpha)
    new_w = weights*torch.exp(alpha*I)
    print('new_w',new_w)
    new_w = new_w / torch.sum(new_w)
    print('new_w',new_w)
    return new_w,I,error,alpha

def init_statement(num_samples,**kwargs):
    w0 = torch.ones(num_samples.shape)/len(num_samples)
    return w0

def batch_data_loader(X_samples,y_samples,batch_size,class_weights):
    num_samples = len(X_samples)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    for i in range(0,num_samples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_samples)])
        yield X_samples[batch_indices],y_samples[batch_indices],class_weights[batch_indices]


def bootstrap_sample(X, y, weights):
    """
    根据给定的权重进行Bootstrap抽样。
    :param X: 输入特征，形状为[N, d]，其中N是样本数，d是特征维度。
    :param y: 目标标签，形状为[N]。
    :param weights: 样本权重，形状为[N]。
    :return: Bootstrap抽样后的样本和权重。
    """
    N = X.size(0)
    # 根据权重生成累积分布函数
    cum_weights = torch.cumsum(weights, dim=0)
    print('cum_weights',cum_weights)
    # 生成N个随机数，这些随机数将用于选择样本
    random_values = torch.rand(N) * cum_weights[-1] + cum_weights[0]
    print(cum_weights[-1],cum_weights[0])
    print('random_values',random_values)
    # 找到随机数在CDF中的索引位置
    indices = torch.tensor([x.item() if x<=4 and x>=0 else (x - 1).item() for x in torch.searchsorted(cum_weights, random_values)])
    print('indices',indices)
    # 使用索引抽取样本
    sampled_X = X[indices]
    sampled_y = y[indices]
    # 重新计算采样的权重，因为Bootstrap样本中的权重应该重新归一化
    sample_weights = torch.ones(N) / N
    print('sample_weights',sample_weights)
    return (sampled_X, sampled_y), sample_weights


def train_weak_classfier(S_b,weights,model,losser,optimizer,epochs,cuda_ids=None):
    if cuda_ids:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.nn.DataParallel(model,device_ids=cuda_ids).to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
    loss,optimizer = losser,optimizer
    predicted = []
    labels = []
    accuacy_list = [0]
    best_weights = None
    best_train_loss = None
    best_I = None
    best_error = None
    best_alpha = None
    for epoch in range(epochs):
        model.train()
        train_losses = 0
        correct = 0
        total = 0
        for X,y in S_b:
            optimizer.zero_grad()
            if cuda_ids:
                X,y = X.to(device),y.to(device)
            else:
                X,y = X,y
            labels.append(y.tolist())
            y_hat = model(X)
            predicted.append(y_hat.tolist())
            train_loss = loss(y_hat,y)
            train_loss.backward()
            _,predict = torch.max(y_hat,dim=1)
            total += y.size[0]
            correct += (predict==y).sum().item()
            optimizer.step()
            train_losses += train_loss
        accuacy = correct / total
        if all(accuacy >= x for x in accuacy_list):
            best_weights,best_I,best_error,best_alpha = update_weights(weights=weights,predicted=predicted,labels=labels)
            best_train_loss = train_losses
        else:
            best_weights,best_I,best_error,best_alpha = weights,best_I,best_error,best_alpha
            best_train_loss = best_train_loss
        accuacy_list.append(correct / total)
    return model,best_train_loss,best_weights,best_I,best_error,best_alpha,max(accuacy_list)

def test_weak_learner(testset,model,losser,cuda_ids=None):
    if cuda_ids:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.nn.DataParallel(model, device_ids=cuda_ids).to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
    model.eval()
    test_losses = 0
    prediction = []
    with torch.no_grad():
        for X,y in testset:
            if cuda_ids:
                X,y = X.to(device),y.to(device)
            else:
                X,y = X,y
            y_hat = model(X)
            test_loss = losser(y_hat,y)
            test_losses += test_loss
            prediction.append(y_hat.tolist())
    return prediction,test_losses

def predict(weak_learners, X, alphas):
    # 初始化预测结果矩阵，每个测试样本的每个类别初始化为0
    predictions = np.zeros((X.shape[0], len(alphas)))
    for b, learner in enumerate(weak_learners):
        # 使用弱学习器进行预测
        pred = learner.predict(X)
        # 加权预测结果
        predictions[:, b] = alphas[b] * (pred == 1)  # 假设是二分类问题
    # 返回得票最多的类别索引
    return np.argmax(np.sum(predictions, axis=1))

def SMMAE(X,y,test_S0,B,model,losser,optimizer,epochs,cuda_ids=None):
    weights = init_statement(X.shape[0])
    alphas = []
    weak_learners = []
    for b in range(B):
        S_b,sample_weights = bootstrap_sample(X=X,y=y,weights=weights)
        G_b,G_b_losses,G_b_weignts,G_b_I,G_b_error,G_b_alpha,G_b_acc = train_weak_classfier(
            S_b=S_b,
            weights=weights,
            model=model,
            losser=losser,
            optimizer=optimizer,
            epochs=epochs,
            cuda_ids=cuda_ids
        )
        alphas.append(G_b_alpha)
        weak_learners.append(G_b)



"""
X = torch.tensor([[1,2,3],
                  [1,1,3],
                  [1,1,3],
                  [1,2,3],
                  [1,2,2]])
labels = torch.tensor([0, 1, 1, 0, 2])
predictions = torch.tensor([0, 1, 0, 1, 2])
num_samples = len(labels)
K = 3  # 假设有三个类别
B = 10 #10次迭代

# 初始化权重
weights = init_statement(num_samples=labels, K=5)

model = LeNet(3,4)


"""


"""
a = torch.tensor([
    [
        [[1,2,3],[1,2,3],[1,2,3]],
        [[4,5,6],[4,5,6],[4,5,6]],
        [[7,8,9],[7,8,9],[7,8,9]]
    ],
    [
        [[1,2,3],[6,2,3],[1,2,3]],
        [[4,5,6],[4,5,6],[4,5,6]],
        [[7,8,9],[7,8,9],[7,8,9]]
    ],
    [
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[4, 5, 6], [4, 4, 6], [4, 5, 6]],
        [[7, 8, 9], [7, 8, 9], [7, 8, 9]]
    ],
    [
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        [[4, 5, 6], [4, 5, 6], [4, 5, 6]],
        [[7, 8, 9], [7, 3, 9], [7, 8, 9]]
    ],
])

b = torch.tensor([1,2,3,4])
class_weight = torch.tensor([1,1,1,1])
train_iter = batch_data_loader(a,b,1,class_weight)
model = LeNet(3,4)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.00001,momentum=0.9)

w0 = init_statement(4)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    predicted = []
    for X,y,_ in train_iter:
        optimizer.zero_grad()
        y_hat = model(X)
        predicted.append(torch.argmax(y_hat,dim=1))
        train_loss = loss(y_hat,y)
        train_loss.backward()
        optimizer.step()
    print(predicted)
    print(upadte_weights(weights=w0,predicted=predicted,labels=b,K=4))
"""










