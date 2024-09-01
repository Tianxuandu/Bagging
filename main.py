import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from Boost_Ada_Bagging  import data_loader,final_accuacy
from MCBAB import train_model

def main(**kwargs):
    torch.multiprocessing.freeze_support()  # Linux系统不需要
    batch_size = 16
    n_base_model = 5
    n_samples = 5
    lr = 1e-4
    weight_decay = 1e-4
    cuda_id = [0]  # [0,1,2,3,4,5,6,7]
    max_epochs = 10
    num_classes = 5
    B = 2
    bins=[1,2,4,8]
    """
    wandb.init(
        project='Bagging',
        name='bagging10',
        config=wandb.config,
    )

    wandb.config.update({
        'n_base_model': n_base_model,
        'n_samples': n_samples,
        'max_lr': lr,
        'weight_decay': weight_decay,
        'cuda_id': cuda_id,
        'max_epochs': max_epochs,
    })
    """
    result,_ = train_model(
        data_dir_train='C:/Users/Dumin/Desktop/dataset/train/',
        data_dir_test='C:/Users/Dumin/Desktop/dataset/val/',
        bins=bins,
        n_base_models=n_base_model,
        n_samples=n_samples,
        epochs=max_epochs,
        batch_size=batch_size,
        B=B,
        lr=lr,
        weight_decay=weight_decay,
        num_classes=num_classes,
        cuda_id=cuda_id,
    )

    _, test_labels, _, _ = data_loader(
        data_dir='C:/Users/Dumin/Desktop/dataset/val/',
        resize=(28, 28),
        is_category=True
    )
    test_labels = test_labels.tolist()
    final_accuacy_test = final_accuacy(
        final_test_list=result,
        test_labels=test_labels,
    )
    print(final_accuacy_test)
    """
    wandb.log({
        'final_acc':final_accuacy_test
    })
    wandb.finish()
    #wandb.save('result.pkl')
    """


if __name__ == '__main__':
    main()