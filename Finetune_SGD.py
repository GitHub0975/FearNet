from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as pltD
import time
import copy
import os
import shutil
import sys
import datetime
import matplotlib.pyplot as plt
from NCM_calculate import NCM_result
from evaluate import accuracy_result
sys.path.append('General_utils')

from ImageFolderTrainVal import *
#from test_network import *
#from SGD_Training import *

# the weights of losses
COS_WEIGHT = 10.
EN_WEIGHT = 1.
L1_WEIGHT = 1.
MSE_WEIGHT = 1.
prototype = 2048

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=200):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    print('lr is '+str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
def fine_tune_SGD(dataset_path='',num_epochs=100,
        exp_dir='',model_path='',lr=0.0004, 
        batch_size=100, init_freeze=1, class_num = 0):
   
    print('lr is ' + str(lr))
    
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=0)
                for x in ['train_true', 'val_true']}
    dset_sizes = {x: len(dsets[x]) for x in ['train_true', 'val_true']}
    print(dset_sizes['train_true'])
    dset_classes = dsets['train_true'].classes
    
    use_gpu = torch.cuda.is_available()

    resume = exp_dir + '/epoch.pth.tar'

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
    if not os.path.isfile(model_path):
        model_ft = models.alexnet(pretrained=True)
       
    else:
        model_ft=torch.load(model_path)
    if not init_freeze:    
        num_ftrs = model_ft.classifier[6].in_features 
        model_ft.classifier._modules['6'] = nn.Linear(num_ftrs, len(dset_classes))    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()
    
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr, amsgrad = True)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr)
        
    
  
    model_ft = train_model(model = model_ft, optimizer = optimizer_ft,
        lr_scheduler = exp_lr_scheduler, lr = lr, 
        dset_loaders = dset_loaders, dset_sizes = dset_sizes, 
        use_gpu = use_gpu,num_epochs = num_epochs,
        exp_dir = exp_dir, resume = resume,batch_size=batch_size, class_num = class_num)
    
    return model_ft
    
def test_Acc(model, dset_loaders, use_gpu, dset_sizes, class_num):
    model.train(False)
    
    epoch_acc_acc = 0
    # Mean center test method
    #epoch_acc = accuracy_result(model, dset_loaders,
    #                dset_sizes, class_num)
    
    # classify test method
    total = 0
    correct = 0
    with torch.no_grad():# test時不需要反向傳播
        for inputs, labels in (dset_loaders):
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()).view(-1, 3, 32, 32), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            encoded, decoded, classify, features = model(inputs)
            
            _, predicted = torch.max(encoded[2].data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_acc = correct/total
    
    return epoch_acc
    
def train_model(model, optimizer, lr_scheduler,lr,
        dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',
        resume='',batch_size=100, class_num=2):
        
    print('dictoinary length'+str(len(dset_loaders)))
    
    print(model)
    since = datetime.datetime.now()

    #best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        #pdb.

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    
    #cos_history = []
    ce_history = []
    mse_history = []
    l1_history = []
    total_loss = []
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        optimizer = lr_scheduler(optimizer, epoch,lr)
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for step, (inputs, labels) in enumerate(dset_loaders['train_true']):
                
            inputs = inputs.view(-1, 3, 32, 32)
                
            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            model.zero_grad()
            # forward
            encoded, decoded, classify, features = model(inputs)
                

                
            # 為了計算l1_loss產生零張量
            Zeros_tensor = torch.zeros(encoded[2].size())
            Zeros_tensor = Zeros_tensor.to("cuda")
            loss_mse = nn.MSELoss()
            #loss_cos = nn.CosineEmbeddingLoss(reduction='sum')    # 預設是做64個距離的平均
            loss_en = nn.CrossEntropyLoss()
            loss_l1 = nn.L1Loss()    # 計算encoded和0張量的絕對值差
            labels = labels.to(device="cuda", dtype=torch.int64)
            #loss = EN_WEIGHT * loss_en(classify, labels) + MSE_WEIGHT * loss_mse(decoded, features) + L1_WEIGHT * loss_l1(encoded, Zeros_tensor)
            loss1 = 0
            #loss1 += loss_mse(decoded[2], features)
            #for i in range(len(encoded)-1):
            #    loss1 += loss_mse(encoded[i], decoded[1-i]) 
            #loss1 = MSE_WEIGHT * loss_mse(decoded, features) 
            loss2 = L1_WEIGHT * loss_l1(encoded[2], Zeros_tensor)
            loss3 = EN_WEIGHT * loss_en(encoded[2], labels)

            loss = loss3 + loss2
                
            _, preds = torch.max(encoded[2].data, 1)

            loss.backward()
            optimizer.step()

                # statistics
            running_loss += loss.item()
             
            running_corrects += torch.sum(preds == labels.data)
            
        # save mean and std information
        #NCM_result(model, dset_loaders['train_true'], batch_size, dset_sizes['train_true'], class_num)   
            
        # calculate acc
        epoch_loss = running_loss / dset_sizes['train_true']
        #儲存各loss值
        #epoch_acc = accuracy_result(model, dset_loaders['train_true'], dset_sizes['train_true'], class_num)
        #print(epoch_acc)

        epoch_acc = float(running_corrects) / dset_sizes['train_true']
        
        train_acc_history.append(epoch_acc)
        ce_history.append(loss_en(encoded[2], labels))
        #ce_history.append(loss_en(encoded[2], labels))
        #mse_history.append(loss1)
        #mse_history.append(loss_mse(decoded[2], features))
        l1_history.append(loss_l1(encoded[2], Zeros_tensor))
        total_loss.append(epoch_loss)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train_ture', epoch_loss, epoch_acc))
        test_epoch_acc = test_Acc(model, dset_loaders['val_true'], use_gpu, dset_sizes['val_true'], class_num)
        print('Testing Acc: {:.4f}'.format(test_epoch_acc))
                
    # 把softmax拿掉
    model.encoder3 = nn.Sequential(*list(model.encoder3.children())[:-1])

        
    # 固定編碼器不訓練
    for name, param in model.encoder1.named_parameters():
        param.requires_grad = False
    for name, param in model.encoder2.named_parameters():
        param.requires_grad = False
    for name, param in model.encoder3.named_parameters():
        param.requires_grad = False
        
    for epoch in range(10):
        print('Epoch {}/{}'.format(epoch, 9))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        optimizer = lr_scheduler(optimizer, epoch,lr)
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for step, (inputs, labels) in enumerate(dset_loaders['train_true']):
                
            inputs = inputs.view(-1, 3, 32, 32)
                
            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            model.zero_grad()
            # forward
            encoded, decoded, classify, features = model(inputs)
            
            loss_mse = nn.MSELoss()
            
            labels = labels.to(device="cuda", dtype=torch.int64)
            
            loss1 = 0
            loss1 += loss_mse(decoded[2], features)
            for i in range(len(encoded)-1):
                loss1 += loss_mse(encoded[i], decoded[1-i]) 

            loss = loss1
                
            _, preds = torch.max(encoded[2].data, 1)

            loss.backward()
            optimizer.step()

                # statistics
            running_loss += loss.item()
             
            running_corrects += torch.sum(preds == labels.data)
            
        # save mean and std information
        NCM_result(model, dset_loaders['train_true'], batch_size, dset_sizes['train_true'], class_num)   
            
        # calculate acc
        epoch_loss = running_loss / dset_sizes['train_true']
        #儲存各loss值
        #epoch_acc = accuracy_result(model, dset_loaders['train_true'], dset_sizes['train_true'], class_num)
        #print(epoch_acc)

        epoch_acc = float(running_corrects) / dset_sizes['train_true']
        train_acc_history.append(epoch_acc)
        mse_history.append(loss1)
        total_loss.append(epoch_loss)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train_ture', epoch_loss, epoch_acc))   
                
        #save related information
        
        print()
        
        test_epoch_acc = test_Acc(model, dset_loaders['val_true'], use_gpu, dset_sizes['val_true'], class_num)
        print('Testing Acc: {:.4f}'.format(test_epoch_acc))
        test_acc_history.append(test_epoch_acc)
        
    NCM_acc = accuracy_result(model, dset_loaders['val_true'], dset_sizes['val_true'], class_num)
    print('Testing Acc: {:.4f}'.format(NCM_acc))
    
    epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
    save_checkpoint({
            'epoch': num_epochs,
            'epoch_acc':epoch_acc,
            'arch': 'full_connection',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
    
    plot_result(total_loss, 'Model loss', 'Loss', 'Epoch', exp_dir + '/loss.png')
    plot_result(mse_history, 'mse loss', 'Loss', 'Epoch', exp_dir + '/loss_mse.png')
    #plot_result(cos_history, 'cos loss', 'Loss', 'Epoch', exp_dir + '/loss_cosine.png')
    plot_result(ce_history, 'ce loss', 'Loss', 'Epoch', exp_dir + '/loss_cosine.png')
    plot_result(l1_history, 'l1 loss', 'Loss', 'Epoch', exp_dir + '/loss_L1.png')
    plot_result(train_acc_history, 'accuracy', 'Accuracy', 'Epoch', exp_dir + '/accuracy_train.png')
    plot_result(test_acc_history, 'accuracy', 'Accuracy', 'Epoch', exp_dir + '/accuracy_test.png')
    #plot_result(acc_history[:len(acc_history)//2], 'accuracy', 'Accuracy', 'Epoch', 'accuracy.png')
    
    end = datetime.datetime.now()
    print('執行時間: {}'.format(str(end-since)))
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    
    return model
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)

    
def plot_result(list, title_name, ylabel_name, xlabel_name, save_name):
    plt.plot(list)
    plt.title(title_name)
    plt.ylabel(ylabel_name)
    plt.xlabel(xlabel_name)
    
    plt.savefig(save_name)
    plt.close()
    #plt.show()
