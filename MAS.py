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
sys.path.append('General_utils')
sys.path.append('MAS_utils')
from ImageFolderTrainVal import *

from MAS_based_Training import *

from Data_generate import *
from Merge_generate import *
import pdb

import pandas as pd
from torch.utils.data import DataLoader as loader

from NCM_calculate_1 import NCM_result
from evaluate_1 import accuracy_result

prototype = 2048
feature_size = 8192

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=200):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    print('lr is '+str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
    
def Big_Merger(dataset_path, previous_pathes, previous_task_model_path, exp_dir, data_dirs, reg_sets, 
                reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200, weight_decay=1e-5, 
                b1=False, after_freeze=1, class_num = 1, task = 1, digits = [1,2]):
    """Call MAS on mainly a sequence of tasks with a head for each where between at each step it sees samples from all the previous tasks to approximate the importance weights 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    pevious_pathes:pathes of previous methods to use the previous heads in the importance weights computation. We assume that each task head is not changed in classification setup of different tasks
    reg_sets,data_dirs: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """
    # New training data
    end_class = class_num // 10 * 10
    start_class = end_class - 10
    class_split = []
    for i in range(start_class, end_class):
        class_split.append(i)
    print(class_split)
    dsets = {}
    dsets['train_true'] = CIFAR_Split('D:\\Brain_Scan_PCA', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
                       
    dsets['val_true'] = CIFAR_Split('D:\\Brain_Scan_PCA', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
                       
    dset_loaders = { 'train_true' : loader(dsets['train_true'], batch_size = batch_size, shuffle = True),
                     'val_true' : loader(dsets['val_true'], batch_size = batch_size, shuffle = False)}
                       
    dset_sizes = {x: len(dsets[x]) for x in ['train_true', 'val_true']}
    dset_classes = dsets['train_true'].classes
    
    
    use_gpu = torch.cuda.is_available()
    checkpoint = torch.load(previous_task_model_path)
    model_ft = checkpoint['model']

    train_encode = None
    train_label = None
    for step, data in enumerate(dset_loaders['train_true']):
        input, label = data
        input = input.view(-1, 3, 32, 32).cuda()
        
        encoded_array, decoded_array, _, features = model_ft(input)
        test_encode = encoded_array[1]
        #print(label)
        
        if train_encode is None:
            train_encode = features
            train_label = label
        else:
            train_encode = torch.cat((train_encode, features), 0)
            train_label = torch.cat((train_label, label), 0)
            
    save_checkpoint({
            'input': train_encode.to(cpu),
            'label': train_label,
            }, 'New_merge_encode.pth.tar')    # New data and new label
    
    

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    
    # 準備舊資料(training和testing要不一樣)
    train_total_sample = [(start_class) * 500] 
    train_class_num = [start_class]

    root = 'D:\\Brain_Scan_PCA'
    
    print('-------old class num: {}---------'.format(train_class_num))
    print('-------old training samples: {}---------'.format(train_total_sample))
    
    # produce old data features(implements decoder) and new train data
    train_dset = Prototype_merge_generate(train_total_sample, class_num = train_class_num, root = root, digits = digits, model = model_ft)
    
    layer_names = []
    for param_tensor in model_ft.encoder3.state_dict():
        layer_names.append(param_tensor)
    #    print(param_tensor, "\t", model_ft.encoder3.state_dict()[param_tensor].size())
    
    last_weight = model_ft.encoder3.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層
    last_bias = model_ft.encoder3.state_dict()[layer_names[len(layer_names) - 1 ]]    # 倒數第二層(bias層)
    
    # 載入之前的權重到前幾個task的神經元
    f = nn.Linear(prototype, class_num - 5)
    f.weight.data[0:class_num - 15] = last_weight
    f.bias.data[0:class_num - 15] = last_bias
    
    
    # 分類器分class_num類
    model_ft.encoder3 = nn.Sequential(
        nn.ELU(inplace = False),
        f,
        nn.Softmax(dim=1))
        
    layer_names = []
    for param_tensor in model_ft.decoder1.state_dict():
        layer_names.append(param_tensor)
    #    print(param_tensor, "\t", model_ft.decoder1.state_dict()[param_tensor].size())
    
    last_weight1 = model_ft.decoder1.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層
    last_bias1 = model_ft.decoder1.state_dict()[layer_names[len(layer_names) - 1 ]]    # 倒數第二層(bias層)
    #print(last_weight.size())
    # 載入之前的權重到前幾個task的神經元
    f1 = nn.Linear(class_num - 5, prototype)
    #print(f.bias.data.size())
    f1.weight.data[:,0:class_num - 15] = last_weight1
    f1.bias.data = last_bias1
        
    model_ft.decoder1 = nn.Sequential(
        f1)
        
    
        
    print(model_ft)
    
    # 訓練編碼器和解碼器
    for name, param in model_ft.encoder1.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.encoder2.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.encoder3.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.decoder1.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.decoder2.named_parameters():
        param.requires_grad = True
    for name, param in model_ft.decoder3.named_parameters():
        param.requires_grad = True
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    
    class_split = []
    # 只丟舊任務(不包括現在的10個任務)
    for i in range(start_class):
        class_split.append(i)

    test_dset =  CIFAR_Split('D:\\Brain_Scan_PCA', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
    
    oldData_loader = { 'train_true' : loader(train_dset, batch_size = batch_size, shuffle = True),
                       'val_true' : loader(test_dset, batch_size = batch_size, shuffle = True)}
                       
    # MAS_based_Training的train_model
    model_ft = train_Big_Merger(model_ft, exp_lr_scheduler, lr, dset_loaders, oldData_loader, 
            dset_sizes, use_gpu, num_epochs, exp_dir, resume, class_num, task = task, reg_sets = reg_sets)
    
    
    return model_ft#, all_task_acc   


def Big_Separate(dataset_path, previous_pathes, previous_task_model_path, exp_dir, data_dirs, reg_sets, 
                reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200, weight_decay=1e-5, 
                b1=False, after_freeze=1, class_num = 1, task = 1, digits = [1,2]):
    """Call MAS on mainly a sequence of tasks with a head for each where between at each step it sees samples from all the previous tasks to approximate the importance weights 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    pevious_pathes:pathes of previous methods to use the previous heads in the importance weights computation. We assume that each task head is not changed in classification setup of different tasks
    reg_sets,data_dirs: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """

    print('Dataset Path: {}'.format(dataset_path))
    dsets = torch.load(dataset_path)    # dictionary dataset(train and test data), new data
    
    # New training data
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                           shuffle=True, num_workers=0)
                for x in ['train_true', 'val_true']}
    dset_sizes = {x: len(dsets[x]) for x in ['train_true', 'val_true']}
    dset_classes = dsets['train_true'].classes

    use_gpu = torch.cuda.is_available()
    checkpoint = torch.load(previous_task_model_path)
    model_ft = checkpoint['model']

    # 分類器一律變成兩個神經元
    model_ft.classifier = nn.Sequential(
        #nn.ELU(inplace = False),
        #nn.Dropout(0.5),
        #nn.Linear(feature_size, 4096),
        nn.ELU(inplace = False),
        nn.Dropout(0.5),
        nn.Linear(feature_size, 6))
    
    # 固定編碼以及解碼器不訓練
    for name, param in model_ft.encoder1.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.encoder2.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.encoder3.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder1.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder2.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder3.named_parameters():
        param.requires_grad = False
    print(model_ft)
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    
    # 準備舊資料(training和testing要不一樣)
    train_total_sample = [(class_num - 5) * 100] # 這裡為不平衡資料(訓練效果較好?)
    train_class_num = [class_num - 5]

    root = 'D:\\Brain_Scan_PCA'
    
    print('-------old class num: {}---------'.format(train_class_num))
    print('-------old training samples: {}---------'.format(train_total_sample))
    
    train_dset = Prototype_generate(train_total_sample, class_num = train_class_num, root = root, digits = digits, model=model_ft, New_data = False)

    class_split = []
    # 只丟舊任務(不包括現在的任務)
    for i in range(class_num-5):
        class_split.append(i)
    print(class_split)

    test_dset =  CIFAR_Split('D:\\Brain_Scan_PCA', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
                       
   
    oldData_loader = { 'train_true' : loader(train_dset, batch_size = batch_size, shuffle = True),
                       'val_true' : loader(test_dset, batch_size = batch_size, shuffle = True)}
    
    # MAS_based_Training的train_model
    model_ft = train_Big_Separate(model_ft, exp_lr_scheduler, lr, dset_loaders, oldData_loader, 
            dset_sizes, use_gpu, num_epochs, exp_dir, resume, class_num, task = task, reg_sets = reg_sets)

    
    #-----------------Test accuracy(包含前面的task)---------------------------
    # 紀錄每個任務的準確率
    all_task_acc=[]
    for t in range(len(reg_sets)):
        
        dset = torch.load(reg_sets[t])
        dset_loader= torch.utils.data.DataLoader(dset['val_true'], batch_size=150,
                                               shuffle=False, num_workers=0)
             
        #print(dset_loader)
        dset_sizes = len(dset['val_true'])
        dset_classes = dset['val_true'].classes
        #print("dset_classes--------------------------------------------")
        #print(dset_classes)
        print('------------task {:} acc ---------------'.format(t + 1))
        print(reg_sets[t])
        
        # 判斷用舊任務分類還是新任務
        
        
        
        epoch_acc,_ = accuracy_result(model_ft, dset_loader, dset_sizes, class_num)
            
        #epoch_acc = float(running_corrects) / len(dset['val_true'])

        print('{} {:} {} Acc: {:.4f}'.format('Task', t + 1, 'val', epoch_acc))
        all_task_acc.append(epoch_acc)
    
    
    
    print("------------task current acc ---------------")
    dset_loaders = torch.utils.data.DataLoader(dsets['val_true'], batch_size=150,
                                           shuffle=False, num_workers=0)
                
    dset_sizes = len(dsets['val_true'])
    dset_classes = dsets['val_true'].classes
    

    epoch_acc,_ = accuracy_result(model_ft, dset_loaders, dset_sizes, class_num)
    
    
    print('{} Acc: {:.4f}'.format('val', epoch_acc))
    all_task_acc.append(epoch_acc)
    #print(model_ft.classifier)

    
    return model_ft, all_task_acc        
    
def Train_New(dataset_path, previous_pathes, previous_task_model_path, exp_dir, data_dirs, 
        reg_sets, reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200, weight_decay=1e-5, b1=False, 
        after_freeze=1, class_num = 1, task = 1, digits = [1,2]):
    """Call MAS on mainly a sequence of tasks with a head for each where between at each step it sees samples from all the previous tasks to approximate the importance weights 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    pevious_pathes:pathes of previous methods to use the previous heads in the importance weights computation. We assume that each task head is not changed in classification setup of different tasks
    reg_sets,data_dirs: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """
    #print("~~~~~~~~~~~~~~~~")
    print('DataPath: {}'.format(dataset_path))
    dsets = torch.load(dataset_path)    # dictionary dataset(train and test data), new data
    
    # New training data
    start_class = class_num // 10 * 10
    if class_num % 10 == 0:
        start_class -= 10
    
    class_split = []
    for i in range(start_class, class_num):
        class_split.append(i)
    print(class_split)
        
    dsets = {}
    dsets['train_true'] = CIFAR_Split('D:\\Brain_Scan_PCA', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
                       
    dsets['val_true'] = CIFAR_Split('D:\\Brain_Scan_PCA', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
                       
    dset_loaders = { 'train_true' : loader(dsets['train_true'], batch_size = batch_size, shuffle = True),
                     'val_true' : loader(dsets['val_true'], batch_size = batch_size, shuffle = False)}
                       
    dset_sizes = {x: len(dsets[x]) for x in ['train_true', 'val_true']}
    dset_classes = dsets['train_true'].classes
    

    use_gpu = torch.cuda.is_available()
    
    checkpoint = torch.load(previous_task_model_path)
    model_ft = checkpoint['model']
   

    # 固定編碼以及解碼器不訓練
    for name, param in model_ft.encoder1.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.encoder2.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.encoder3.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder1.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder2.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder3.named_parameters():
        param.requires_grad = False
    print(model_ft)
    
    criterion = nn.CrossEntropyLoss()

    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    
    # 準備舊資料(training和testing要不一樣),每個task有500個data
    # task == 20, 30, 40...
    if task%2 == 0:
        nueron_num = 10
    else:
        nueron_num = task - task//10*10
    
    if class_num%10 == 0:
        first_nueron_class = (class_num - 1)//10*10
        nueron_num = 11
    else:
        first_nueron_class = class_num//10*10
        
    print('-Nueron number: {}---------'.format(nueron_num))
    print('-First nueron number: {}---------'.format(first_nueron_class))

    #print(nueron_num)
    # old data samples and classes
    train_total_sample = [first_nueron_class * 100]    # 每個task有100個Data
    train_class_num = [start_class]  # 每個nueron有幾個類別
    
    print('-------Old data classes: {}---------'.format(train_class_num))
    print('-------Old data samples: {}---------'.format(train_total_sample))
    
    
    train_encode = None
    train_label = None
    for step, data in enumerate(dset_loaders['train_true']):
        input, label = data
        input = input.view(-1, 3, 32, 32).cuda()
        
        encoded_array, _, _, features = model_ft(input)
        test_encode = encoded_array[1]
            
        first_neuron_class = class_num//10*10
        if class_num % 10 == 0:
            first_neuron_class -= 10
        label = (label - first_neuron_class).float() + 1    #label轉成十個神經元的
        #print(label)
        
        if train_encode is None:
            train_encode = features
            train_label = label
        else:
            train_encode = torch.cat((train_encode, features), 0)
            train_label = torch.cat((train_label, label), 0)
            
    save_checkpoint({
            'input': train_encode.to(cpu),
            'label': train_label,
            }, 'New_training_encode.pth.tar')
    
    
    train_dset = Prototype_generate(train_total_sample, class_num = train_class_num, 
        root = 'D:\\Brain_Scan_PCA', digits = digits, model = model_ft, New_data = True)

    # old data的test有真正資料，要拿真正資料做評估
    
    class_split = []
    # 只丟舊任務(不包括現在的任務，-10 class)
    for i in range(class_num-10):
        #print(np.arange(i,i+10))
        class_split.append(i)
    print(class_split)
    
    test_dset =  CIFAR_Split('D:\\Brain_Scan_PCA', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                       ]),digits=class_split)
    
    oldData_loader = { 'train_true' : loader(train_dset, batch_size = 128, shuffle = True),
                       'val_true' : loader(test_dset, batch_size = batch_size, shuffle = False)}
    
    layer_names = []
    for param_tensor in model_ft.classifier.state_dict():
        layer_names.append(param_tensor)
        #print(param_tensor, "\t", model_ft.classifier.state_dict()[param_tensor].size())
        
    #model_ft.classifier = nn.Sequential(*list(model_ft.classifier.children())[:-2])
        
    #last_weight1 = model_ft.classifier.state_dict()[layer_names[len(layer_names) - 4 ]]   #倒數第3層
    #last_bias1 = model_ft.classifier.state_dict()[layer_names[len(layer_names) - 3 ]]    # 倒數第4層(bias層)
    
    last_weight2 = model_ft.classifier.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層
    last_bias2 = model_ft.classifier.state_dict()[layer_names[len(layer_names) - 1 ]]    # 倒數第二層(bias層)
    
    # 載入之前的權重到前幾個task的神經元
    #f1 = nn.Linear(feature_size, 4096)
    #f1.weight.data = last_weight1
    #f1.bias.data = last_bias1
    
    f2 = nn.Linear(feature_size, nueron_num)
    f2.weight.data[0:6] = last_weight2
    f2.bias.data[0:6] = last_bias2
    
    # MAS_based_Training的train_model
    model_ft.classifier = nn.Sequential(
                    nn.ELU(inplace=False),
                    nn.Dropout(0.5),
                    #*list(model_ft.classifier.children())[:-2],
                    f2)
                    
    '''model_ft.classifier = nn.Sequential(
        nn.ELU(inplace = False),
        nn.Dropout(0.5),
        nn.Linear(feature_size, 11))'''
                    
    print(model_ft._modules)

    model_ft = train_New(model_ft, criterion, exp_lr_scheduler, lr, dset_loaders, oldData_loader, dset_sizes, use_gpu, num_epochs, exp_dir, resume, class_num, task = task, reg_sets = reg_sets)
    
    #-----------------Test accuracy(包含前面的task)---------------------------
    # 準確率寫在檔案中
    all_task_acc=[]
    for t in range(len(reg_sets)):
        
        dset = torch.load(reg_sets[t])
        dset_loader= torch.utils.data.DataLoader(dset['val_true'], batch_size=150,
                                               shuffle=False, num_workers=0)
             
        #print(dset_loader)
        dset_sizes = len(dset['val_true'])
        dset_classes = dset['val_true'].classes
        #print("dset_classes--------------------------------------------")
        #print(dset_classes)
        print('------------task {:} acc at {}---------------'.format(t + 1,reg_sets[t]))
        print(reg_sets[t])
        
        # 判斷用舊任務分類還是新任務
        epoch_acc, _ = accuracy_result(model_ft, dset_loader, dset_sizes, class_num)
            
        #epoch_acc = float(running_corrects) / len(dset['val_true'])

        print('{} {:} {} Acc: {:.4f}'.format('Task', t + 1, 'val', epoch_acc))
        all_task_acc.append(epoch_acc)
    
    
    print("------------task current acc ---------------")
    dset_loaders = torch.utils.data.DataLoader(dsets['val_true'], batch_size=150,
                                           shuffle=False, num_workers=0)
                
    dset_sizes = len(dsets['val_true'])
    dset_classes = dsets['val_true'].classes
    

    epoch_acc, _ = accuracy_result(model_ft, dset_loaders, dset_sizes, class_num)
    
    
    print('{} Acc: {:.4f}'.format('val', epoch_acc))
    all_task_acc.append(epoch_acc)
    #print(model_ft.classifier)
    
    return model_ft, all_task_acc    


def update_weights_params(data_dir,reg_sets,model_ft,batch_size,norm='L2'):
    """update the importance weights based on the samples included in the reg_set. Assume starting from zero omega
    
       model_ft: the model trained on the previous task 
    """
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    

    # prepare the dataset
    dset_loaders=[]
    for data_path in reg_sets:
    
        # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset=torch.load(data_path)
            dset=dset['train']
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
        dset_loaders.append(dset_loader)    # 每個任務的data的loaders

    
    use_gpu = torch.cuda.is_available()

    #inialize the importance params,omega, to zero
    reg_params=initialize_reg_params(model_ft)    # return初始參數以及初始importance(全為0)
    model_ft.reg_params=reg_params
    #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
    optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
    
    if norm=='L2':
        print('********************MAS with L2 norm***************')
        #compute the imporance params
        model_ft = compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        if norm=='vector':
            optimizer_ft=MAS_Omega_Vector_Grad_update(model_ft.parameters(), lr=0.0001, momentum=0.9)

            model_ft = compute_importance_gradient_vector(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)

        else:
            model_ft = compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)

   
    return model_ft#
def update_sequence_MAS_weights(data_dirs,reg_sets,previous_models,model_ft,batch_size,norm='L2'):
    """updates a task in a sequence while computing omega from scratch each time on the previous tasks
       previous_models: to use their heads for compute the importance params
    """
    data_transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t=0    
    #last_layer_index=str(len(model_ft.classifier._modules)-1)    # 取得最後一層的索引值
    for model_path in previous_models:
        pre_model=torch.load(model_path)
        #get previous task head(取得最後一層參數)
        # 好像不需要，因為是autoencoder，不是分類的
        #model_ft.classifier._modules[last_layer_index] = pre_model.classifier._modules[last_layer_index]

        # if data_dirs is None then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dirs is not None:
            dset = ImageFolderTrainVal(data_dirs[t], reg_sets[t], data_transform)
        else:
            dset=torch.load(reg_sets[t])    # 一次一個舊task
            dset=dset['train_true']
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=0)
       
    #=============================================================================

        use_gpu = torch.cuda.is_available()
        
        # freeze layer
        freeze = ['decoder.0.weight', 'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias', 'decoder.4.weight', 'decoder.4.bias']
       
        if t==0:
            
            #initialize to zero
            reg_params=initialize_reg_params(model_ft, freeze)
        else:
            #store previous task param
            reg_params=initialize_store_reg_params(model_ft, freeze)    # prev_omega, current_omega(prev_omega=current_omega), weights

        model_ft.reg_params=reg_params    # 幫model_ft加一個屬性(reg_params)

        #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
        optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
        
        #legacy code
        dset_loaders=[dset_loader]    # 對不同的舊任務都開一個dataloader(其實length都是1)
        #compute param importance
        if norm=='L2':
            print('********************objective with L2 norm***************')
            model_ft = compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
        else:
            model_ft = compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
        if t>0:    # 非第一個task
            reg_params=accumelate_reg_params(model_ft)    # 把之前的任務所算的omega加起來
        
        model_ft.reg_params=reg_params
        t=t+1
    sanitycheck(model_ft)   
    return model_ft
    
def Big_Merger_10(dataset_path, previous_pathes, previous_task_model_path, exp_dir, data_dirs, reg_sets, reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200, weight_decay=1e-5, b1=False, after_freeze=1, class_num = 1, task = 1):
    """Call MAS on mainly a sequence of tasks with a head for each where between at each step it sees samples from all the previous tasks to approximate the importance weights 
    dataset_path=new dataset path
    exp_dir=where to save the new model
    previous_task_model_path: previous task in the sequence model path to start from 
    pevious_pathes:pathes of previous methods to use the previous heads in the importance weights computation. We assume that each task head is not changed in classification setup of different tasks
    reg_sets,data_dirs: sets of examples used to compute omega. Here the default is the training set of the last task
    b1=to mimic online importance weight computation, batch size=1
    reg_lambda= regulizer hyper parameter. In object recognition it was set to 1.
    norm=the norm used to compute the gradient of the learned function
    """
    #print("~~~~~~~~~~~~~~~~")
    print('Dataset Path: {}'.format(dataset_path))
    dsets = torch.load(dataset_path)    # dictionary dataset(train and test data), new data
    
    # New training data
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                           shuffle=True, num_workers=0)
                for x in ['train_true', 'val_true']}
    dset_sizes = {x: len(dsets[x]) for x in ['train_true', 'val_true']}    # train和test的總張數
    dset_classes = dsets['train_true'].classes

    use_gpu = torch.cuda.is_available()
   
    model_ft = torch.load(previous_task_model_path)    # load previous trained model

    # 分類器一律變成11個神經元
    model_ft.classifier = nn.Sequential(
        nn.ReLU(inplace = False),
        nn.Linear(prototype, 11))
    
    # 固定編碼以及解碼器不訓練
    for name, param in model_ft.encoder.named_parameters():
        param.requires_grad = False
    for name, param in model_ft.decoder.named_parameters():
        param.requires_grad = False
    print(model_ft._modules)
    
    criterion = nn.CrossEntropyLoss()    # 冗餘參數
    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    #our optimizer(沒什麼效...)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr, amsgrad = True)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    
    # 準備舊資料(training和testing要不一樣)
    train_total_sample = [(class_num - 11) * 500] # 這裡為不平衡資料(訓練效果較好?)
    train_class_num = [class_num - 11]
    #test_total_sample = [100]
    #test_class_num = [class_num - 1]
    root = './General_utils/'
    
    print('-------{}---------'.format(train_class_num))
    print('-------{}---------'.format(train_total_sample))
    
    train_dset = Prototype_generate(train_total_sample, class_num = train_class_num, root = root, digits = digits)
    # old data的test有真正資料，要拿真正資料做評估
    old_Data = torch.load(reg_sets[-1])    # 一次只能拿一個task當測試....這樣的測試是非常不準的(會跟最後的測試結果不一樣)
    #print('------Test data data path: {}----'.format(reg_sets[-1]))
    test_dset = old_Data['val_true'] # 拿前一個task的舊資料來測試準確率
    #dset_classes = dsets['val_true'].classes
    #print(dset_classes)
    #print(dset_sizes['train_true'])
    #print(dset_sizes['val_true'])
    print(len(old_Data['val_true']))
    #train_dset = old_Data['train_true']
    #test_dset = Prototype_generate(test_total_sample, class_num = test_class_num, root = root)
    oldData_loader = { 'train_true' : loader(train_dset, batch_size = batch_size, shuffle = True),
                       'val_true' : loader(test_dset, batch_size = batch_size, shuffle = True)}
    
    # MAS_based_Training的train_model
    model_ft = train_Big_Merger(model_ft, criterion, exp_lr_scheduler, lr, dset_loaders, oldData_loader, dset_sizes, use_gpu, num_epochs, exp_dir, resume, class_num, task = task, reg_sets = reg_sets)
    
    
    # 儲存model
    #torch.save(model_ft, os.path.join(exp_dir, 'D://Brain_Scan_Architecture/Final_model.pth.tar'))
    
    
    #-----------------Test accuracy(包含前面的task)---------------------------
    # 紀錄每個任務的準確率
    all_task_acc=[]
    correct_num = []
    for t in range(len(reg_sets)):
        
        dset = torch.load(reg_sets[t])
        dset_loader= torch.utils.data.DataLoader(dset['val_true'], batch_size=150,
                                               shuffle=False, num_workers=0)
             
        #print(dset_loader)
        dset_sizes = len(dset['val_true'])
        dset_classes = dset['val_true'].classes
        #print("dset_classes--------------------------------------------")
        #print(dset_classes)
        print('------------task {:} acc ---------------'.format(t + 1))
        print(reg_sets[t])
        
        # 判斷用舊任務分類還是新任務
        
        
        
        epoch_acc, total_corrects = accuracy_result(model_ft, dset_loader, dset_sizes, class_num)
        correct_num.append(total_corrects)
            
        #epoch_acc = float(running_corrects) / len(dset['val_true'])

        print('{} {:} {} Acc: {:.4f}'.format('Task', t + 1, 'val', epoch_acc))
        all_task_acc.append(epoch_acc)
    
    
    
    print("------------task current acc ---------------")
    dset_loaders = torch.utils.data.DataLoader(dsets['val_true'], batch_size=150,
                                           shuffle=False, num_workers=0)
                
    dset_sizes = len(dsets['val_true'])
    dset_classes = dsets['val_true'].classes
    

    epoch_acc, total_corrects = accuracy_result(model_ft, dset_loaders, dset_sizes, class_num)
    
    # 更新prev_mean和prev_std
    shutil.copyfile('./General_utils/current_feature_mean.npy', './General_utils/prev_feature_mean.npy')
    shutil.copyfile('./General_utils/current_feature_cov.npy', './General_utils/prev_feature_cov.npy')
    #print('{} {:} {} Acc: {:.4f}'.format('Task', t + 1, 'val', epoch_acc))

    print('{} Acc: {:.4f}'.format('val', epoch_acc))
    all_task_acc.append(epoch_acc)
    correct_num.append(total_corrects)   # 儲存正確張數(evaluate所有類別)
    #print(model_ft.classifier)
    
    
    return model_ft, all_task_acc    
    
def accumulate_MAS_weights(data_dir,reg_sets,model_ft,batch_size,norm='L2'):
    """accumelate the importance params: stores the previously computed omega, compute omega on the last previous task
            and accumelate omega resulting on  importance params for all the previous tasks
       reg_sets:either a list of files containing the samples used for computing the importance param like train or train and test
                or pytorch dataset, then train set is used
       data_dir:
    """
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    

    #prepare the dataset
    dset_loaders=[]
    for data_path in reg_sets:
    
        # if data_dir is not None then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset=torch.load(data_path)
            dset=dset['train']
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
        dset_loaders.append(dset_loader)
    #=============================================================================
    
    use_gpu = torch.cuda.is_available()

    #store the previous omega, set values to zero
    reg_params=initialize_store_reg_params(model_ft)
    model_ft.reg_params=reg_params
    #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
    optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
   
    if norm=='L2':
        print('********************objective with L2 norm***************')
        model_ft =compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        model_ft =compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    #accumelate the new importance params  with the prviously stored ones (previous omega)
    reg_params=accumelate_reg_params(model_ft)
    model_ft.reg_params=reg_params    # 只有importance和weights
    sanitycheck(model_ft)   
    return model_ft

def sanitycheck(model):
    for name, param in model.named_parameters():
           
        #print (name)
        if param in model.reg_params:
            
            reg_param=model.reg_params.get(param)
            omega=reg_param.get('omega')
            
            #print('omega max is',omega.max())
            #print('omega min is',omega.min())
            #print('omega mean is',omega.mean())
            
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)