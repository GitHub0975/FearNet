from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from MNIST_Net import *
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader
from NCM_calculate_1 import NCM_result
from evaluate_1 import accuracy_result
from itertools import cycle
from Data_generate import *
from torch import autograd
from torch.utils.data import DataLoader as loader
#end of imports
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# the weights of losses
COS_WEIGHT = 10.
L1_WEIGHT = 1.
MSE_WEIGHT = 1.
EN_WEIGHT = 1.
prototype = 2048

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_Acc(model, use_gpu, class_num, oldData_loader, newData_loader, Not_Merge = True, classifier = False):
    model.train(False)
    model.cuda()
    
    running_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for step, data in enumerate(oldData_loader):
                
            # get the inputs
            old_input, old_label = data    # data_loader的image
                
            # old data label
            first_neuron_class = class_num//10*10
            if class_num % 10 == 0:
                first_neuron_class -= 10
                
            if Not_Merge:
                old_label = (old_label - first_neuron_class).float() + 1
                zero_class = torch.zeros(old_label.size())
                old_label = torch.where(old_label > 0.0, old_label, zero_class)
            
            # Old task shape
            old_input = old_input.view(-1, 3, 32, 32)
                
            if use_gpu:
                old_input, old_label = old_input.cuda(), old_label.cuda()

            encoded_array, _, old_classify, _ = model(old_input)
            test_encode = encoded_array[2]
            if Not_Merge:
                _, old_preds = torch.max(old_classify, 1)
            else:
                _, old_preds = torch.max(test_encode, 1)
            
            #print('===========================================')
            #print(old_preds)
            #print('-------------------------------------------')
            #print(old_label)

            # statistics
            running_corrects += torch.sum(old_preds == old_label.data)
            total_samples += old_preds.size()[0]
        print('Old Data Acc: {}'.format(running_corrects / total_samples))
        
        running_corrects1 = 0
        total_samples1 = 0
        for step, data in enumerate(newData_loader):
                
            # get the inputs
            new_input, new_label = data    # data_loader的image
                
            # new label production
            if Not_Merge:
                first_neuron_class = class_num//10*10
                if class_num % 10 == 0:
                    first_neuron_class -= 10
                new_label = (new_label - first_neuron_class).float() + 1

            # Old task shape
            new_input = new_input.view(-1, 3, 32, 32)
                
            if use_gpu:
                new_input, new_label = new_input.cuda(), new_label.cuda()

            encoded_array, _, new_classify, _ = model(new_input)
            test_encode = encoded_array[2]
            if Not_Merge:
                _, new_preds = torch.max(new_classify, 1)
            else:
                _, new_preds = torch.max(test_encode, 1)
            
            #print('===========================================')
            #print(new_preds)
            #print('-------------------------------------------')
            #print(new_label)

            # statistics
            running_corrects += torch.sum(new_preds == new_label.data)
            total_samples += new_preds.size()[0]
            
            running_corrects1 += torch.sum(new_preds == new_label.data)
            total_samples1 += new_preds.size()[0]
        print('New Data Acc: {}'.format(running_corrects1 / total_samples1))
    return running_corrects / total_samples
    
# task 2~n 用的train model
def train_Big_Merger(model, lr_scheduler,lr,dset_loaders, oldData_loader, 
        dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='', class_num=2, task = 1, reg_sets=[]):
    """Train a given model using MAS optimizer. The only unique thing is that it passes the importnace params to the optimizer"""

    print('dictoinary length'+str(len(dset_loaders)))
    since = time.time()
    
    optimizer_ft = optim.Adam(model.parameters(), lr, amsgrad = True)

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #print('load')
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print('Start_epoch: {}'.format(str(start_epoch)))
    #pdb.set_trace()
    cos_history = []
    mse_history = []
    l1_history = []
    total_loss = []
    acc_history = []
    regulizer = []
    
    for epoch in range(start_epoch, num_epochs):
    #for epoch in range(50):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
            
        # Iterate over data.
        for step, data in enumerate(oldData_loader['train_true']):
                
            # get the inputs
            old_input, old_label = data    # data_loader的image
            
            # Old task shape
            old_input = old_input.view(-1, 8192).float()    # 去掉三通道的第一個通道(因為是灰階)
                
            if use_gpu:
                old_input, old_label = old_input.cuda(), old_label.cuda()

            # zero the parameter gradients
            optimizer_ft.zero_grad()    #不然gradient會每個batch加總
                
            # Old task forward
            encoded1 = model.encoder1(old_input)
            encoded2 = model.encoder2(encoded1)
            encoded3 = model.encoder3(encoded2)
            
            decoded1 = model.decoder1(encoded3)
            decoded2 = model.decoder2(decoded1)
            decoded3 = model.decoder3(decoded2)
            #old_classify = model.classifier(encoded)
            #decoded = model.decoder(encoded)
                
            # 合併新舊資料
            loss_en = nn.CrossEntropyLoss()
            loss_l1 = nn.L1Loss()
                
            labels = old_label.to(device="cuda", dtype=torch.int64)
            #print(labels)
            Zeros_tensor = torch.zeros(encoded3.size()).to(device)
                
            loss = EN_WEIGHT * loss_en(encoded3, labels) + L1_WEIGHT * loss_l1(encoded3, Zeros_tensor)
                
            _, preds = torch.max(encoded3, 1)
            #print('=============================================')
            #print(preds)
            #print('---------------------------------------------')
            #print(labels)
                
                
            #with autograd.detect_anomaly():    #異常偵測
            loss.backward()
            optimizer_ft.step()

            # statistics
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data)
            total_samples = total_samples + preds.size()[0]
        # 更新各類別的Mean和std值
        #NCM_result(model, oldData_loader['train_true'], total_samples, class_num-5) 
            
        #epoch_acc = accuracy_result(model, dset_loaders['val_true'], dset_sizes['val_true'])
        epoch_loss = running_loss / dset_sizes['train_true']
        epoch_acc = running_corrects.float() / total_samples

        total_loss.append(loss)
        acc_history.append(epoch_acc)
            
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))
        del labels
        del old_input
        del old_label
        '''epoch_file_name=exp_dir+'/'+'Merge'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_ft.state_dict(),
                },epoch_file_name)'''
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    # 拿掉softmax
    model.encoder3 = nn.Sequential(*list(model.encoder3.children())[:-1])
    
    # 固定編碼器不訓練
    for name, param in model.encoder1.named_parameters():
        param.requires_grad = False
    for name, param in model.encoder2.named_parameters():
        param.requires_grad = False
    for name, param in model.encoder3.named_parameters():
        param.requires_grad = False
    print(model)
    
        
    for epoch in range(20):
        print('Epoch {}/{}'.format(epoch, 19))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
            
        # Iterate over data.
        for step, data in enumerate(oldData_loader['train_true']):
                
            # get the inputs
            old_input, old_label = data    # data_loader的image
            
            # Old task shape
            old_input = old_input.view(-1, 8192).float()    # 去掉三通道的第一個通道(因為是灰階)
                
            if use_gpu:
                old_input, old_label = old_input.cuda(), old_label.cuda()

            # zero the parameter gradients
            optimizer_ft.zero_grad()    #不然gradient會每個batch加總
                
            # Old task forward
            encoded1 = model.encoder1(old_input)
            encoded2 = model.encoder2(encoded1)
            encoded3 = model.encoder3(encoded2)
            
            decoded1 = model.decoder1(encoded3)
            decoded2 = model.decoder2(decoded1)
            decoded3 = model.decoder3(decoded2)
            #old_classify = model.classifier(encoded)
            #decoded = model.decoder(encoded)
                
            # 合併新舊資料
            loss_mse = nn.MSELoss()
                
            labels = old_label.to(device="cuda", dtype=torch.int64)
            
            loss1 = loss_mse(decoded3, old_input) + loss_mse(decoded2, encoded1) + loss_mse(decoded1, encoded2)
                
            loss = MSE_WEIGHT * loss1
                
            _, preds = torch.max(encoded3, 1)
            #print('=============================================')
            #print(preds)
            #print('---------------------------------------------')
            #print(labels)
                
                
            #with autograd.detect_anomaly():    #異常偵測
            loss.backward()
            optimizer_ft.step()

            # statistics
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data)
            total_samples = total_samples + preds.size()[0]
        # 更新各類別的Mean和std值
        NCM_result(model, oldData_loader['train_true'], total_samples, class_num-5, exp_dir) 
            
        #epoch_acc = accuracy_result(model, dset_loaders['val_true'], dset_sizes['val_true'])
        epoch_loss = running_loss / dset_sizes['train_true']
        epoch_acc = running_corrects.float() / total_samples

        total_loss.append(loss)
        acc_history.append(epoch_acc)
            
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))
        del labels
        del old_input
        del old_label
        epoch_file_name=exp_dir+'/'+'Merge'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_ft.state_dict(),
                },epoch_file_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
        
    
    #epoch_acc = test_Acc(model, use_gpu, class_num, oldData_loader['val_true'], dset_loaders['val_true'], Not_Merge = False)
    #print('Test Acc: {:.4f}'.format(epoch_acc))
    #epoch_acc, _ = accuracy_result(model, dset_loaders['val_true'], 1000)
    #print(' New class Test Acc: {:.4f}'.format(epoch_acc))
    #epoch_acc, _ = accuracy_result(model, oldData_loader['val_true'], 5000)
    #print(' old class Test Acc: {:.4f}'.format(epoch_acc))
    
    return model
#importance_dictionary: contains all the information needed for computing the w and omega

    
def train_Big_Separate(model, lr_scheduler,lr,dset_loaders, oldData_loader, 
        dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='', class_num=2, task = 1, reg_sets=[]):
    """Train a given model using MAS optimizer. The only unique thing is that it passes the importnace params to the optimizer"""

    print('dictoinary length'+str(len(dset_loaders)))
    since = time.time()
    
    #optimizer_ft = optim.SGD(model.parameters(), lr, momentum=0.9)
    #optimizer_ft = optim.Adam(model.parameters(), lr, amsgrad = True)
    #optimizer_ft = optim.Adam(model.parameters(), lr)
    optimizer_ft = optim.Adam(model.parameters(), lr, weight_decay=0.01)

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #print('load')
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    #start_epoch=0
    print('Start_epoch: {}'.format(str(start_epoch)))
    #pdb.set_trace()
    cos_history = []
    mse_history = []
    l1_history = []
    total_loss = []
    acc_history = []
    regulizer = []
    
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
      
        for step, data in enumerate(zip(oldData_loader['train_true'], cycle(dset_loaders['train_true']))):
                
            # get the inputs
            old_input = data[0][0]    # data_loader的image
            old_label = torch.zeros(data[0][1].size()[0])
           
            new_input = data[1][0]    # data_loader1的image   
            #print(old_input.size())
            # new label production
            first_neuron_class = class_num//10*10
            new_label = (data[1][1] - first_neuron_class).float() + 1
            #new_label = torch.ones(data[1][1].size()[0])

            # Old task shape
            old_input = old_input.view(-1, 8192).float()    # 去掉三通道的第一個通道(因為是灰階)
        
            # New task shape
            new_input = new_input.view(-1, 3, 32, 32)
            #old_input = old_input.view(-1, 3, 32, 32)
                
            if use_gpu:
                old_input, old_label, new_input, new_label = old_input.cuda(), old_label.cuda(), new_input.cuda(), new_label.cuda()

            # zero the parameter gradients
            optimizer_ft.zero_grad()    #不然gradient會每個batch加總

            # New task forward
            _, _, new_classify, _ = model(new_input)
            #print(new_classify.size())
                
            old_classify = model.classifier(old_input)
                
            # 合併新舊資料
            labels = torch.cat((old_label, new_label), 0)
            all_preds = torch.cat((old_classify, new_classify), 0)
            #print(labels.size())
            #print(all_preds.size())
                
            loss_en = nn.CrossEntropyLoss()
                
            labels = labels.to(device="cuda", dtype=torch.int64)
                
            loss = loss_en(all_preds, labels)
                
            _, old_preds = torch.max(old_classify, 1)
            _, new_preds = torch.max(new_classify, 1)
            preds = torch.cat((old_preds, new_preds), 0)
            #print('=============================================')
            #print(preds)
            #print('---------------------------------------------')
            #print(labels)
                
                
                # backward + optimize only if in training phase
                #if phase == 'train_true':
            #with autograd.detect_anomaly():    #異常偵測
            loss.backward()
            optimizer_ft.step()

            # statistics
            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data)
            total_samples = total_samples + old_preds.size()[0] + new_preds.size()[0]
            
            # calculate acc by class center
            #torch.save(model, 'model_tmp_normalize.pth.tar')
            
            ########################################################################################
        #NCM_result(model, dset_loaders['train_true'], dset_sizes['train_true'], class_num)   
            #epoch_acc = accuracy_result(model, dset_loaders['val_true'], dset_sizes['val_true'])
        epoch_loss = running_loss / dset_sizes['train_true']
        epoch_acc = running_corrects.float() / total_samples
            #print('正確的數量:{}'.format(running_corrects.float()))
            #print('總數:{}'.format(total_samples))
            #儲存各loss值
        total_loss.append(loss)
        acc_history.append(epoch_acc)
            
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))
        del labels
        del old_input
        del new_input
        del old_label
        del new_label
            #del loss
            #del preds
        
        
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_ft.state_dict(),
                },epoch_file_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    
    epoch_acc = test_Acc(model, use_gpu, class_num, oldData_loader['val_true'], dset_loaders['val_true'], classifier=True)
    
    print('Test Acc: {:.4f}'.format(epoch_acc))
    
    return model
    

def train_New(model, criterion, lr_scheduler,lr,dset_loaders, oldData_loader, 
        dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='', class_num=2, task = 1, reg_sets=[]):
    """Train a given model using MAS optimizer. The only unique thing is that it passes the importnace params to the optimizer"""

    '''optimizer = optim.Adam(model.parameters(), lr, amsgrad=True, weight_decay = 0.05)
    
    for epoch in range(20):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
        first_neuron_class = class_num//10*10
        if class_num % 10 == 0:
            first_neuron_class -= 10
                
        
        num = 0
        for i, data in enumerate(dset_loaders['train_true']):

            inputs, labels = data
            
            
            labels = (labels - first_neuron_class).float() + 1
            
            inputs = inputs.cuda().float().view(-1, 3, 32, 32)
            labels = labels.to(device="cuda", dtype=torch.int64)
            
            # 將梯度置零
            optimizer.zero_grad()
            
            #outputs = model(inputs)
            model = model.cuda()
            _, _, classify, _ = model(inputs)

            _, preds = torch.max(classify, 1)
            
            loss_en = nn.CrossEntropyLoss()    
            loss = loss_en(classify, labels).to(device)
            
            loss.backward()   # 梯度計算
            optimizer.step()  # 參數更新
            
            running_corrects += torch.sum(preds == labels.data)

            running_loss += loss.item()
            num += labels.size()[0]

        print('epoch: %d\t loss: %.6f\t Acc: %.6f' % (epoch + 1, running_loss / num, running_corrects/num))
        
        
    layer_names = []
    for param_tensor in model.classifier.state_dict():
        layer_names.append(param_tensor)
            #print(param_tensor, "\t", model_ft.classifier.state_dict()[param_tensor].size())
        
    last_weight2 = model.classifier.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層
    last_bias2 = model.classifier.state_dict()[layer_names[len(layer_names) - 1 ]]    # 倒數第二層(bias層)
    
    # 載入新task的資料到model中
    f = nn.Linear(8192, 11)
        
    # 載入之前的權重到前幾個task的神經元
    f.weight.data[1:11] = last_weight2[1:11]
    f.bias.data[1:11] = last_bias2[1:11]
        
    model.classifier = nn.Sequential(
            nn.ELU(inplace=False),
            nn.Dropout(0.5),
            f)'''
        
    
    print('dictoinary length'+str(len(dset_loaders)))
    since = time.time()
    
    #optimizer_ft = optim.SGD(model.parameters(), lr, momentum=0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr, amsgrad = True, weight_decay=0.01)
    #optimizer_ft = optim.Adam(model.parameters(), lr)
    
    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch=0
        print("=> no checkpoint found at '{}'".format(resume))
    
    print('Task: {}, Start_epoch: {}'.format(task, str(start_epoch)))
    #pdb.set_trace()
    cos_history = []
    mse_history = []
    l1_history = []
    total_loss = []
    acc_history = []
    regulizer = []

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        
        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
            
        # 因為是兩個Dataloader一起使用，怕數量會不一樣，因此都重新計算
        total_samples = 0
        
        num = 0
        for i, data in enumerate(oldData_loader['train_true']):

            inputs, labels = data
            
            inputs = inputs.cuda().float()#.view(-1, 3, 32, 32)
            labels = labels.to(device="cuda", dtype=torch.int64)
            
            # 將梯度置零
            optimizer_ft.zero_grad()
            
            #outputs = model(inputs)
            model = model.cuda()
            classify = model.classifier(inputs)

            _, preds = torch.max(classify, 1)
            
            loss_en = nn.CrossEntropyLoss()    
            loss = loss_en(classify, labels).to(device)
            
            loss.backward()   # 梯度計算
            optimizer_ft.step()  # 參數更新
            
            running_corrects += torch.sum(preds == labels.data)

            running_loss += loss.item()
            num += labels.size()[0]

        print('epoch: %d\t loss: %.6f\t Acc: %.6f' % (epoch + 1, running_loss / num, running_corrects/num))
        #NCM_result(model, dset_loaders['train_true'], dset_sizes['train_true'], class_num)
        
        #epoch_acc = test_Acc(model, use_gpu, class_num, oldData_loader['val_true'], dset_loaders['val_true'])
        #print('Test Acc: {:.4f}'.format(epoch_acc))
        
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_ft.state_dict(),
                },epoch_file_name)
                
                
    # 載入之前的權重到前幾個task的神經元
    '''f.weight.data[1:11] = last_weight2[1:11]
    f.bias.data[1:11] = last_bias2[1:11]
        
    model.classifier = nn.Sequential(
            nn.ELU(inplace=False),
            nn.Dropout(0.5),
            f)'''
    
    epoch_acc = test_Acc(model, use_gpu, class_num, oldData_loader['val_true'], dset_loaders['val_true'], classifier=True)
    
    print('Test Acc: {:.4f}'.format(epoch_acc))
        
    return model


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)

def random_shuffle(encoded_vec, label):

    tensor_size = encoded_vec.size()[0]
    order = torch.randperm(tensor_size)    # 順序表
    shuffle_encode = torch.empty(tensor_size, prototype)
    shuffle_label = torch.empty(tensor_size)
    
    # 按順序表填入相對應的tensor以及標籤
    for i in range(tensor_size):
        shuffle_encode[i] = encoded_vec[order[i]]
        shuffle_label[i] = label[order[i]]
        
    return shuffle_encode, shuffle_label
   
