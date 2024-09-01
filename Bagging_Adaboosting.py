import wandb
from tqdm import tqdm
import numpy as np
import torch




#error计算
def cal_error(weights,I):
    error = torch.sum(weights*I) / (torch.sum(weights) + 1e-6)
    return error

#alpha计算
def update_alpha(error,K=None):
    error_bias = 1e-6
    alpha = torch.log((1-error)/error) if error != 0 else torch.log((1-error)/(error+error_bias))  #+ math.log(1-K)
    return alpha

#权重更新
def update_weights(weights,predicted,labels):   #类别数K
    assert len(predicted) == len(labels)
    weights = torch.tensor(weights)
    predicted = torch.tensor(predicted)
    labels = torch.tensor(labels)
    #误差矩阵I
    I = (predicted != labels)
    I = I.float()
    error = cal_error(
        weights=weights,
        I=I
    )
    #print('I',I)
    print('error',error)
    alpha = update_alpha(error=error)
    print('alpha',alpha)
    new_w = weights*torch.exp(alpha*I)
    new_w = new_w / torch.sum(new_w)
    return new_w,I,error,alpha

def init_statement(num_samples,**kwargs):
    w0 = torch.ones(num_samples)/num_samples
    return w0


def bootstrap_sample(X, y, class_weights,weights):
    """
    根据给定的权重进行Bootstrap抽样。
    :param X: 输入特征，形状为[N, d]，其中N是样本数，d是特征维度。
    :param y: 目标标签，形状为[N]。
    :param weights: 样本权重，形状为[N]。
    :return: Bootstrap抽样后的样本和权重。
    """
    N = X.size(0)
    if N == 1:
        return X,y,class_weights,weights
    else:
        # 根据权重生成累积分布函数
        cum_weights = torch.cumsum(weights, dim=0)
        # 生成N个随机数，这些随机数将用于选择样本
        random_values = torch.rand(N) * cum_weights[-1] + cum_weights[0]
        # 找到随机数在CDF中的索引位置
        indices = torch.tensor([x.item() if x<=4 and x>=0 else (x - 1).item() for x in torch.searchsorted(cum_weights, random_values)])
   
        # 使用索引抽取样本
        sampled_X = X[indices]
        sampled_y = y[indices]
        new_class_weights = class_weights[indices]
        # 重新计算采样的权重，因为Bootstrap样本中的权重应该重新归一化
        sample_weights = torch.ones(N) / N

        return sampled_X, sampled_y, new_class_weights,sample_weights

def batch_data_loader(X_samples,y_samples,batch_size,class_weights = None,weights=None,is_shuffle=True):
    num_samples = len(X_samples)
    indices = list(range(num_samples))
    if is_shuffle:
        np.random.shuffle(indices)
    else:
        pass
    for i in range(0,num_samples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_samples)])
        yield X_samples[batch_indices],y_samples[batch_indices],class_weights[batch_indices] if class_weights is not None else None,weights[batch_indices] if weights is not None else None

def train_weak_classfier(S_b,model,losser,optimizer,epochs,cuda_ids=None):
    if cuda_ids:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.nn.DataParallel(model,device_ids=cuda_ids).to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
    loss,optimizer = losser,optimizer
    accuacy_list = [0]
    best_weights = None
    best_train_loss = None
    best_I = None
    best_error = None
    best_alpha = None
    for epoch in tqdm(range(epochs),desc='epoch'):
        print(' ')
        model.train()
        train_losses = 0
        correct = 0
        total = 0
        predicted = []
        labels = []
        weights = []
        for X,y,_,batch_weights in tqdm(S_b[epoch],desc='train boost batch'):
            print(' ')
            optimizer.zero_grad()
            if cuda_ids:
                X,y = X.to(device),y.to(device)
            else:
                X,y = X,y
            for x in y.tolist():
                labels.append(x)
            y_hat = model(X)
            train_loss = loss(y_hat,y)
            train_loss.backward()
            _,predict = torch.max(y_hat,dim=1)
            for x in predict:
                predicted.append(x)
            total += y.shape[0]
            correct += (predict==y).sum().item()
            optimizer.step()
            train_losses += train_loss
            for x in batch_weights.tolist():
                weights.append(x)
        accuacy = correct / total
        if all(accuacy >= x for x in accuacy_list):
            best_weights,best_I,best_error,best_alpha = update_weights(weights=weights,predicted=predicted,labels=labels)
            best_train_loss = train_losses
        else:
            best_weights,best_I,best_error,best_alpha = best_weights,best_I,best_error,best_alpha
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
    class_weights_test = []
    with torch.no_grad():
        for X,y,class_batch_weights,_ in tqdm(testset[0],desc='test boost batch'):
            print(' ')
            if cuda_ids:
                X,y = X.to(device),y.to(device)
            else:
                X,y = X,y
            y_hat = model(X)
            test_loss = losser(y_hat,y)
            _,predicted = torch.max(y_hat,dim=1)
            for i in range(len(y_hat)):
                if predicted[i] == y[i]:
                    class_batch_weights[i] = 1
                else:
                    class_batch_weights[i] = 0
            for i in class_batch_weights:
                class_weights_test.append(i)
            test_losses += test_loss
            for i in y_hat.tolist():
                prediction.append(i)
    return prediction,class_weights_test,test_losses

def predict(weak_learners, X,y,class_weights_test, losser,alphas,batch_size,epochs,cuda_ids=None):
    # 初始化预测结果矩阵，每个测试样本的每个类别初始化为0
    predictions = np.zeros((X.shape[0], len(alphas)))
    final_class_test_weights = np.zeros((X.shape[0],len(alphas)))
    for b, learner in enumerate(weak_learners):
        # 使用弱学习器进行预测
        _,test_iter = data_iter_product(
            n_iters=epochs,
            X_samples=None,
            y_samples=None,
            class_weights_train=None,
            X_samples_test=X,
            y_samples_test=y,
            class_weights_test= class_weights_test,
            weights=None,
            epochs=epochs,
            batch_size=batch_size,
            is_train=False,
            is_test=True
        )
        pred,per_class_test_weights,_ = test_weak_learner(testset=test_iter,model=learner,losser=losser,cuda_ids=cuda_ids)
        # 加权预测结果
        _,pred_max = torch.max(torch.tensor(pred),dim=1)
        print(y)
        print(pred_max)
        predictions[:, b] = pred_max  # 假设是二分类问题
        final_class_test_weights[: , b] = per_class_test_weights
    final_predictions = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)),axis=1,arr=np.array(predictions,dtype=int))
    final_final_class_test_weights = torch.ones(final_predictions.shape)
    for i in range(len(final_predictions)):
        if final_predictions[i] == y[i]:
            final_final_class_test_weights[i] = 1
        else:
            final_final_class_test_weights[i] = 0
    #final_final_class_test_weights = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)),axis=1,arr=np.array(final_class_test_weights,dtype=int))
    print('final_predictions',final_predictions)
    print('y',y)
    print('final_final_class_test_weights',final_final_class_test_weights)
    # 返回得票最多的类别索引
    return final_predictions,final_final_class_test_weights

def data_iter_product(n_iters,X_samples,y_samples,class_weights_train,X_samples_test,y_samples_test,class_weights_test,weights,epochs,batch_size,is_train=True,is_test=False):
    assert n_iters == epochs
    train_iter_list = []
    test_iter_list = []
    for epoch in range(epochs):
        if is_train:
            train_loader = batch_data_loader(
                    X_samples=X_samples,
                    y_samples=y_samples,
                    batch_size=batch_size,
                    class_weights=class_weights_train,
                    weights=weights,
                    is_shuffle=True
                )
            train_iter_list.append(train_loader)
        else:
            break
    if is_test:
        test_loader = batch_data_loader(
                    X_samples=X_samples_test,
                    y_samples=y_samples_test,
                    batch_size=batch_size,
                    class_weights=class_weights_test,
                    is_shuffle=False
                )
        test_iter_list.append(test_loader)
    return train_iter_list,test_iter_list

def SMMAE(X,y,class_weights_train,X_test,y_test,class_weights_test,B,model,losser,optimizer,epochs,batch_size,cuda_ids=None):
    weights = init_statement(X.shape[0])
    alphas = []
    weak_learners = []
    X_sample,y_sample = X,y
    for b in tqdm(range(B),desc='B'):
        print('')
        X_sample,y_sample,class_weights_boost,sample_weights = bootstrap_sample(X=X_sample,y=y_sample,class_weights=class_weights_train,weights=weights)
        S_b,_ = data_iter_product(
            n_iters=epochs,
            X_samples=X_sample,
            y_samples=y_sample,
            class_weights_train=class_weights_train,
            X_samples_test=X_test,
            y_samples_test=y_test,
            class_weights_test=class_weights_test,
            weights=weights,
            epochs=epochs,
            batch_size=batch_size,
            is_train=True,
            is_test=False
        )
        G_b,G_b_losses,G_b_weignts,G_b_I,G_b_error,G_b_alpha,G_b_acc = train_weak_classfier(
            S_b=S_b,
            model=model,
            losser=losser,
            optimizer=optimizer,
            epochs=epochs,
            cuda_ids=cuda_ids
        )
        alphas.append(G_b_alpha)
        weak_learners.append(G_b)
    final_pred,final_class_weights = predict(
        weak_learners=weak_learners,
        X=X_test,
        y=y_test,
        class_weights_test=class_weights_test,
        alphas=alphas,
        losser=losser,
        epochs=epochs,
        batch_size=batch_size,
        cuda_ids=cuda_ids,
    )
    return final_pred,final_class_weights



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
