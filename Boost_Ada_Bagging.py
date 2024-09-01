import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from LeNet import LeNet
from AlexNet import AlexNet
import torchvision
from torchvision import transforms
from tqdm import tqdm
import wandb
import Bagging_Adaboosting
import torchvision.models as models
from classguided_sampler import class_count_guided_data_loader

#wandb.login(key='ddd005d13a9704b2f25fd1c6ace472b6ca714fc2')

#数据加载器
def data_loader(data_dir,resize=None,is_category = False):
    #创建features与labels列表
    X = []
    y = []
    class_weights = []
    category = {}
    if os.path.exists(data_dir):
        for label,filename in enumerate(os .listdir(data_dir)):
            file_path = os.path.join(data_dir,filename)
            if os.path.isdir(file_path):
                if is_category:
                    category.update({label:sum(os.path.isfile(os.path.join(file_path,entery)) for entery in os.listdir(file_path))})
                for image in os.listdir(file_path):
                    image_path = os.path.join(file_path,image)
                    image = Image.open(image_path).convert('RGB')
                    if resize is not None:
                        image = image.resize(resize)
                    X.append(transforms.ToTensor()(image))
                    y.append(label)
        X = torch.stack(X).float()
        y_one_hot = torch.tensor(y).long()#.float()
        class_weights = torch.ones(y_one_hot.shape)
        #y_one_hot = F.one_hot(torch.tensor(np.array(y),dtype=torch.long),num_classes=len(category)).float()
    else:
        y_one_hot = F.one_hot(torch.tensor(np.array(y),dtype=torch.long),num_classes=1).float()
        class_weights = torch.ones(y_one_hot.shape)
        print('data_loader error!!!!')
    return X, y_one_hot,class_weights,category

#bagging自助抽样
def Bootstrap_samples(X_samples,y_samples,replace=True):
    #确保X和y是来自同一个数据集
    assert X_samples.shape[0] == y_samples.shape[0]
    samples_id = np.random.choice(len(X_samples),len(X_samples),replace=replace)
    X_new_samples = X_samples[samples_id]
    y_new_samples = y_samples[samples_id]
    return X_new_samples,y_new_samples

#自定义采样
'''每次从每个类别中随机采样n张照片组成新的数据集'''
def random_balance_samples(X_samples,y_samples,n_per_datas,category=None):
    assert X_samples.shape[0] == y_samples.shape[0]
    assert all(n_per_datas < x for x in list(category.values()))
    assert type(category) == dict
    X_new_samples = []
    y_new_samples = []
    start_n = 0
    for j in list(category.values()):
        sample_id = np.random.choice(j,n_per_datas)
        sample_id = [sample_index+start_n for sample_index in sample_id]
        for sample_idx in sample_id:
            X_new_samples.append(X_samples[sample_idx].tolist())
            y_new_samples.append(y_samples[sample_idx].tolist())
        start_n += j

    return torch.tensor(X_new_samples),torch.tensor(y_new_samples)

#批量生成器
def batch_data_loader(X_samples,y_samples,batch_size,class_weights,is_shuffle=True):
    num_samples = len(X_samples)
    indices = list(range(num_samples))
    if is_shuffle:
        np.random.shuffle(indices)
    else:
        pass
    for i in range(0,num_samples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_samples)])
        yield X_samples[batch_indices],y_samples[batch_indices],class_weights[batch_indices]

#模型生成器
def models_list(n_models,num_classes):
    #model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = LeNet(3,5)  #本机测试用的
    models = []
    models_request = []   #构建列表，存储resize，num_classes，in_channels数值
    model2 = AlexNet(5)
    models_request.append([(224,224),num_classes,3])
    models_request.append([(224, 224), num_classes, 3])
    models.append(model2)
    models.append(model2)
    for _ in range(0,n_models-2):
        models.append(model)
        models_request.append([(28,28),num_classes,3])
    return models , models_request

#bagging权值投票
"""权值只有0和1，且纯标签后处理，权值并非模型权值"""
def result_optimiszer_selective(workers_test_list,num_classes,class_weights):
    assert type(workers_test_list) == list and len(workers_test_list) > 0
    assert len(workers_test_list[0]) == num_classes
    work_list = np.zeros(torch.tensor(workers_test_list[0]).shape)
    final_class_weights = np.zeros(torch.tensor(workers_test_list[0]).shape)
    for i in range(len(class_weights)):
        for j in range(len(class_weights[i])):
            if class_weights[i][j] == 1:
                work_list[j] = workers_test_list[i][j]
                final_class_weights[j] = 1
            else:
                continue
    return work_list,final_class_weights

#bagging均权投票
def result_optimiszer_selective_bagging_Ada(workers_test_list,num_classes):
    assert type(workers_test_list) == list and len(workers_test_list) > 0
    probs = []
    add_list_dict = {}
    for i in range(len(workers_test_list)):
        max_length = max(len(sublist) for sublist in workers_test_list[i])
        for j in range(len(workers_test_list[i])):
            if len(workers_test_list[i][j]) < max_length:
                add_list_dict[f'{i},{j}'] = max_length - len(workers_test_list[i][j])
                workers_test_list[i][j] += [[0] * num_classes] * (max_length - len(workers_test_list[i][j]))

    for i in range(len(workers_test_list)):
        probs1 = []
        for j in range(len(workers_test_list[0])):
            probs2 = []
            for k in range((len(workers_test_list[0][0]))):
                max_category = np.argmax(np.array(workers_test_list[i][j][k]))
                probs_small = np.zeros(len(workers_test_list[i][j][k]),dtype=int)
                probs_small[max_category] = 1
                probs2.append(probs_small.tolist())
            probs1.append(probs2)
        probs.append(probs1)
    final_predict = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=0,
                                        arr=np.array(probs)).reshape(np.array(workers_test_list[0]).shape)
    return final_predict.tolist(),add_list_dict

#评估器
def final_accuacy(final_test_list,test_labels):
    assert  len(final_test_list) == len(test_labels)

    test_labels = torch.tensor(test_labels).float()
    final_test_list = torch.tensor(final_test_list).float()
    #_,truth = torch.max(test_labels,dim=1)
    correct = (final_test_list==test_labels).sum().item()
    total = len(test_labels)
    final_accuacy = correct/total
    return final_accuacy

def is_iterable(variable):
    try:
        iter(variable)
        return True
    except TypeError:
        return False

#迭代器
def data_iter_product(n_iters,data_dir_train,data_dir_test,models,models_request,samples_method='balance'):
    assert type(models) == list and  n_iters == len(models) == len(models_request)
    data_dir_train = data_dir_train
    data_dir_test = data_dir_test
    resize_list = []
    X_samples_dict = {}
    y_samples_dict = {}
    X_samples_test_dict = {}
    y_samples_test_dict = {}
    class_weights_train_dict = {}
    class_weights_test_dict = {}
    category_dict = {}
    X_samples_list = []
    y_samples_list = []
    X_samples_test_list = []
    y_samples_test_list = []
    class_weights_train_list = []
    class_weights_test_list = []
    for i in range(n_iters):
        resize_list.append(models_request[i][0])
    resize_list = set(resize_list)
    for resize in resize_list:
        X_samples_dict[f'{resize}'], y_samples_dict[f'{resize}'], class_weights_train_dict[f'{resize}'], category_dict[f'{resize}'] = data_loader(
            data_dir=data_dir_train,
            resize=resize,
            is_category=True)
        X_samples_test_dict[f'{resize}'], y_samples_test_dict[f'{resize}'], class_weights_test_dict[f'{resize}'], _ = data_loader(
            data_dir=data_dir_test,
            resize=resize,
            is_category=True)
    for model in models:
        if samples_method == 'balance':
            X_samples, y_samples = random_balance_samples(
                X_samples=X_samples_dict[f'{models_request[models.index(model)][0]}'],
                y_samples=y_samples_dict[f'{models_request[models.index(model)][0]}'],
                n_per_datas=10,
                category=category_dict[f'{models_request[models.index(model)][0]}']
            )
        elif samples_method == 'bootstrap':
            X_samples, y_samples = Bootstrap_samples(
                X_samples=X_samples_dict[f'{models_request[models.index(model)][0]}'],
                y_samples=y_samples_dict[f'{models_request[models.index(model)][0]}']
            )
        else:
            print('train data loader error!!!')
            break
        X_samples_list.append(X_samples)
        y_samples_list.append(y_samples)
        class_weights_train_list.append(class_weights_train_dict[f'{models_request[models.index(model)][0]}'])
        X_samples_test_list.append(X_samples_test_dict[f'{models_request[models.index(model)][0]}'])
        y_samples_test_list.append(y_samples_test_dict[f'{models_request[models.index(model)][0]}'])
        class_weights_test_list.append(class_weights_test_dict[f'{models_request[models.index(model)][0]}'])
    return X_samples_list,y_samples_list,class_weights_train_list,X_samples_test_list,y_samples_test_list,class_weights_test_list

def SMMAE_enhance(X,y,class_weights_train,X_test,y_test,class_weights_test,B,model,losser,optimizer,epochs,batch_size,cuda_ids=None):
    pred,class_test_weight_pred = Bagging_Adaboosting.SMMAE(
        X=X,
        y=y,
        class_weights_train=class_weights_train,
        X_test=X_test,
        y_test=y_test,
        class_weights_test=class_weights_test,
        B=B,
        model=model,
        losser=losser,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        cuda_ids=cuda_ids
    )
    return pred,class_test_weight_pred

def train_model(data_dir_train,data_dir_test,n_base_models,n_samples,epochs,batch_size,B,lr,weight_decay,num_classes,cuda_id=None):
    assert n_base_models == n_samples
    models,models_request = models_list(n_base_models,num_classes=num_classes)
    test_labels_list = []
    test_labels_class_weights = []
    for model in tqdm(models,desc='models'):
        print(' ')
        """
        wandb.log({
            'model':model,
            'model_index':models.index(model),
        })
        """
        model_index = models.index(model)
        test_one_class_weights = []
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        X_samples,y_samples,class_weights_train,X_samples_test,y_samples_test,class_weights_test = data_iter_product(
            n_iters=n_base_models,
            data_dir_train=data_dir_train,
            data_dir_test=data_dir_test,
            models=models,
            models_request=models_request,
            samples_method='bootstrap'
        )
        test_one_model,test_one_class_weights  = SMMAE_enhance(
                            X=X_samples[model_index],
                            y=y_samples[model_index],
                            class_weights_train=class_weights_train[model_index],
                            X_test=X_samples_test[model_index],
                            y_test=y_samples_test[model_index],
                            class_weights_test = class_weights_test[model_index],
                            B=B,
                            model=model,
                            losser=loss,
                            optimizer=optimizer,
                            epochs=epochs,
                            batch_size=batch_size,
                            cuda_ids=cuda_id
                            )

        test_labels_list.append(test_one_model)
        test_labels_class_weights.append(test_one_class_weights)
        print(torch.tensor(test_labels_list))
    return result_optimiszer_selective(
                                        test_labels_list,
                                        num_classes=len(test_labels_list[0]),
                                        class_weights=test_labels_class_weights
                                        )
