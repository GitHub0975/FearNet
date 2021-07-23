
from __future__ import print_function, division

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import sys
import shutil
import pdb

from PIL import Image
import os
import os.path
import errno
import pickle as pk

import codecs
import numpy as np
from torch.utils.data import Dataset
from CIFAR_split import *
from torch.utils.data import DataLoader as loader
from torch.distributions.multivariate_normal import MultivariateNormal

TASK1_CLASS = 50
TASKN_CLASS = 10
prototype = 2048
device = torch.device("cuda")
cpu = torch.device("cpu")


class Prototype_generate(datasets.CIFAR100):    # 自定義資料集
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
  
    

    def __init__(self, total_sample, class_num, root='', train=True, transform=None, target_transform=None, 
                    download=False,digits=[1,2], real_label = True, model=None, New_data = False):
        #super(Prototype_generate, self).__init__(total_sample, class_num, root, train, transform, target_transform, download)
        super(Prototype_generate, self).__init__(root, train, transform, target_transform, download)
        """
            total_sample: (class per task list) [50, 10, 10]
            class_num: (samples per task list) [5000, 500, 500]
        """
        all_mean = np.load('./General_utils/current_feature_mean.npy')
        all_std = np.load('./General_utils/current_feature_cov.npy')
        
        self.target_transform = target_transform

        # 對每一個類別做sample
        self.all_data = None
        self.all_labels = None
        self.pca_transform_data = None
        task_split = True
        print('class_num:{}'.format(class_num))
        #print(model)
        #print(New_data)
        
        s = 0
        e = 0
        #print(torch.rand((1,)))
        # 有幾個task的舊資料要產生
        for c in range(len(class_num)): 

            e += class_num[c]
            # 每個class有幾個sample
            samples = int(total_sample[c] / class_num[c])# + 1
            #print(samples)
            
            for i in range(s, e): # 一個task有多個class
                mean = torch.from_numpy(all_mean[i])
                std = torch.from_numpy(all_std[i])
                
                np.set_printoptions(threshold=np.inf)
                #print(mean)
                n = MultivariateNormal(mean, std)
                instance = n.sample((samples, ))
                instance = instance.to(device).float()
                
                instance = model.decoder1(instance)
                instance = model.decoder2(instance)
                instance = model.decoder3(instance).to(cpu)
                #n = torch.distributions.Normal(mean, std)
                '''if New_data:
                    instance = instance.to(device).float()
                    instance = model.decoder1(instance).to(cpu)'''
                
                #print(instance.size())
                if self.all_data is None:
                    self.all_data = instance
                else:
                    self.all_data = torch.cat((self.all_data, instance), 0)
                    
            # 一顆神經元一個label
            labels = torch.zeros((total_sample[c], )) + c
            if self.all_labels is None:
                self.all_labels = labels
            else:
                self.all_labels = torch.cat((self.all_labels, labels), 0)
                    
            torch.set_printoptions(threshold=np.inf)
            
            s = e # e是新任務
        #print('======================all labels=====================================')
        #print(self.all_labels)
        print('Virtual data: {}'.format(self.all_data.size()))   # 總共產生幾筆虛擬資料    
        torch.set_printoptions(threshold=np.inf)

        #pca = pk.load(open('pca.pkl','rb'))
        #self.pca_transform_data = pca.inverse_transform(self.all_data.cpu().numpy())
        #self.pca_transform_data = torch.from_numpy(self.pca_transform_data)
        #print('Virtual data after pca inverse_transform: {}'.format(self.pca_transform_data.size()))
            
        #Big Separate的時候不做
        if New_data:
            New_data = torch.load('New_training_encode.pth.tar')
            New_input = New_data['input']#.cuda()
            New_label = New_data['label']#.cuda()
        
            self.all_data = torch.cat((self.all_data, New_input), 0)
            self.all_labels = torch.cat((self.all_labels, New_label), 0)
            
        #print(self.all_labels)
        
        
        '''for digit in digits:
            
            digit_mask=torch.eq(self.train_labels , digit) # 判斷train_label中的類別是否是要的，要=True, 不要=False
            digit_index=torch.nonzero(digit_mask)  # 取出類別非零的數值的索引值(第0個類別不要)
            digit_index=digit_index.view(-1)    # 轉成一維tensor(digit_index原本就是一維陣列)
            this_digit_data=self.train_data[digit_index]    # 取出要的data
            this_digit_labels=self.train_labels[digit_mask]    # 取出要的label值(創造出足夠的sample空間)
            #print(this_digit_labels)
            # training data要重新定義label
            if not real_label:
                this_digit_labels.fill_(digits.index(digit))    # 重新定義label值(從0開始)
            
            this_digit_labels = torch.zeros((this_digit_labels.size(0), )) + len(class_num)
            if self.pca_transform_data is None:
                self.pca_transform_data=this_digit_data.clone()    # 複製張量(梯度狀態一樣複製但數值無法流入)
                self.all_labels=this_digit_labels.clone()
            else:
                self.pca_transform_data=torch.cat((self.all_data,this_digit_data),0)
                self.all_labels=torch.cat((self.all_labels,this_digit_labels),0)'''

        print('Total data after pca inverse_transform: {}'.format(self.all_data.size()))
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        #img, target = self.digit_data[index], self.digit_labels[index]
        #prototype, target = self.pca_transform_data[index], self.all_labels[index]
        prototype, target = self.all_data[index], self.all_labels[index]


        if self.target_transform is not None:
            target = self.target_transform(target)
        
        #img=img.view(-1, 3, 32, 32) # 轉成一維tensor(直接丟入全連層)
        
       
        return prototype, target

    def __len__(self):
        return(self.all_labels.size()[0])  # 總data數量

  