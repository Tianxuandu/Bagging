from classguided_sampler import class_count_guided_data_loader
from Boost_Ada_Bagging import (random_balance_samples,Bootstrap_samples,data_loader,
                               result_optimiszer_selective,
                               SMMAE_enhance)
from LeNet import LeNet
from AlexNet import AlexNet
import torch
import tqdm
import torch.nn as nn
import torchvision


#[2,3,4,8]
def MCBAB_data_iter_product(n_iters,bins,data_dir_train,data_dir_test,models,models_request,samples_method='balance'):
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
    X_samples = []
    y_samples = []
    class_weights = []
    for i in range(len(bins)):
        X_samples.append([])
        y_samples.append([])
        class_weights.append([])
    for i in range(n_iters):
        for j in range(len(bins)):
            resize_list.append(models_request[i][j][0])
    resize_list = set(resize_list)
    for resize in resize_list:
        X_samples_dict[f'{resize}'], y_samples_dict[f'{resize}'], class_weights_train_dict[f'{resize}'], category_dict[f'{resize}'] = class_count_guided_data_loader(
            path=data_dir_train,
            bins=bins,
            resize=resize,
            is_category=True)
        X_samples_test_dict[f'{resize}'], y_samples_test_dict[f'{resize}'], class_weights_test_dict[f'{resize}'], _ = data_loader(
            data_dir=data_dir_test,
            resize=resize,
            is_category=True)
    for model in models:
        for i in range(len(bins)):
            if samples_method == 'balance':
                X_samples[i], y_samples[i] = random_balance_samples(
                    X_samples=X_samples_dict[f'{models_request[models.index(model)][i][0]}'][i],
                    y_samples=y_samples_dict[f'{models_request[models.index(model)][i][0]}'][i],
                    n_per_datas=10,
                    category=category_dict[f'{models_request[models.index(model)][i][0]}']
                )
            elif samples_method == 'bootstrap':
                X_samples[i], y_samples[i] = Bootstrap_samples(
                    X_samples=X_samples_dict[f'{models_request[models.index(model)][i][0]}'][i],
                    y_samples=y_samples_dict[f'{models_request[models.index(model)][i][0]}'][i]
                )
            else:
                print('train data loader error!!!')
                break
            class_weights[i] = class_weights_train_dict[f'{models_request[models.index(model)][i][0]}'][i]
        X_samples_test_list.append(X_samples_test_dict[f'{models_request[models.index(model)][i][0]}'])
        y_samples_test_list.append(y_samples_test_dict[f'{models_request[models.index(model)][i][0]}'])
        class_weights_test_list.append(class_weights_test_dict[f'{models_request[models.index(model)][i][0]}'])
        X_samples_list.append(X_samples)
        y_samples_list.append(y_samples)
        class_weights_train_list.append(class_weights)
    return X_samples_list,y_samples_list,class_weights_train_list,X_samples_test_list,y_samples_test_list,class_weights_test_list

def MCBAB_models_list(n_models,bins,num_classes):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    models = []
    models_request = []  # 构建列表，存储resize，num_classes，in_channels数值
    #model = LeNet(3, num_classes)  # 本机测试用的
    #model2 = AlexNet(num_class=num_classes)
    for _ in range(n_models):
        models.append([])
        models_request.append([])# 构建列表，存储resize，num_classes，in_channels数值
    for i in range(n_models):
        for j in range(len(bins)):
            #models[i].append(model2)
            #models[i].append(model2)
            #models_request[i].append([(224, 224), num_classes, 3])
            #models_request[i].append([(224, 224), num_classes, 3])
            models[i].append(model)
            models_request[i].append([(224,224),num_classes,3])
    return models , models_request

def train_model(data_dir_train,data_dir_test,bins,n_base_models,n_samples,epochs,batch_size,B,lr,weight_decay,num_classes,cuda_id=None):
    assert n_base_models == n_samples
    models,models_request = MCBAB_models_list(n_base_models,bins=bins,num_classes=num_classes)
    subset_test_ensemble_list = []
    subset_weights_ensemble_list = []
    X_samples, y_samples, class_weights_train, X_samples_test, y_samples_test, class_weights_test = MCBAB_data_iter_product(
        n_iters=n_base_models,
        bins=bins,
        data_dir_train=data_dir_train,
        data_dir_test=data_dir_test,
        models=models,
        models_request=models_request,
        samples_method='bootstrap'
    )
    for base_model_list in models:
        """
        wandb.log({
            'model':model,
            'model_index':models.index(model),
        })
        """
        model_index = models.index(base_model_list)
        test_labels_list = []
        test_labels_class_weights = []
        for index,model in enumerate(base_model_list):
            test_one_class_weights = []
            loss = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            test_one_model,test_one_class_weights  = SMMAE_enhance(
                                X=X_samples[model_index][index],
                                y=y_samples[model_index][index],
                                class_weights_train=class_weights_train[model_index][index],
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

            test_labels_list.append(test_one_model.tolist())
            test_labels_class_weights.append(test_one_class_weights.tolist())
            print('test_labels_list',torch.tensor(test_labels_list))
            print('test_labels_class_weights',torch.tensor(test_labels_class_weights))
        one_test_labels,one_weights = result_optimiszer_selective(
            test_labels_list,
            num_classes=len(test_labels_list[0]),
            class_weights=test_labels_class_weights
        )
        subset_test_ensemble_list.append(one_test_labels)
        subset_weights_ensemble_list.append(one_weights)
        print('subset_test_ensemble_list',subset_test_ensemble_list)
        print('subset_weights_ensemble_list',subset_weights_ensemble_list)
    return result_optimiszer_selective(
        subset_test_ensemble_list,
        num_classes=len(subset_test_ensemble_list[0]),
        class_weights=subset_weights_ensemble_list
    )

