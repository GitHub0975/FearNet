
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
#prototype = 2048
device = torch.device("cuda")
cpu = torch.device("cpu")


class Prototype_merge_generate(datasets.CIFAR100):    # 自定義資料集
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
    

    def __init__(self, total_sample, class_num, root='', train=True, transform=None, target_transform=None, download=False,digits=[1,2], real_label = True, model=None):
        #super(Prototype_generate, self).__init__(total_sample, class_num, root, train, transform, target_transform, download)
        super(Prototype_merge_generate, self).__init__(root, train, transform, target_transform, download)
        """
            total_sample: (class per task list) [50, 10, 10]
            class_num: (samples per task list) [5000, 500, 500]
        """
        all_mean = np.load('./General_utils/current_feature_mean.npy')
        all_std = np.load('./General_utils/current_feature_cov.npy')
        
        self.target_transform = target_transform
        print('Virtual data generating...')
        # 對每一個類別做sample
        self.all_data = None
        self.all_labels = None
        self.pca_transform_data = None
        task_split = True
        
        s = 0
        e = 0
        #print(torch.rand((1,)))
        # 有幾個task的舊資料要產生
        for c in range(len(class_num)): 

            e += class_num[c]
            # 每個class有幾個sample
            samples = int(total_sample[c] / class_num[c])# + 1
            #print(samples)
            
            # 如果神經元的個數按照task下去分的話(5個task就5個神經元)            
            for i in range(s, e): # 一個task有多個class
                mean = torch.from_numpy(all_mean[i])
                std = torch.from_numpy(all_std[i])
            
                np.set_printoptions(threshold=np.inf)
                #print(mean)
                #n = torch.distributions.Normal(mean, std)
                n = MultivariateNormal(mean, std)
                instance = n.sample((samples, )).to(device).float()
                
                instance = model.decoder3(model.decoder2(model.decoder1(instance))).to(cpu)
                
                #print(instance.size())
                if self.all_data is None:
                    self.all_data = instance
                else:
                    self.all_data = torch.cat((self.all_data, instance), 0)
                
                labels = torch.zeros((instance.size()[0], )) + i
                if self.all_labels is None:
                    self.all_labels = labels
                else:
                    self.all_labels = torch.cat((self.all_labels, labels), 0)
                           
            torch.set_printoptions(threshold=np.inf)
            
            s = e # e是新任務
            
        New_data = torch.load('New_merge_encode.pth.tar')
        New_input = New_data['input']#.cuda()
        New_label = New_data['label']#.cuda()
        
        self.all_data = torch.cat((self.all_data, New_input), 0)
        self.all_labels = torch.cat((self.all_labels, New_label), 0)
        self.all_labels = self.all_labels.int()
        #print('======================all labels=====================================')
        #print(self.all_labels)
        print('Virtual data: {}'.format(self.all_data.size()))   # 總共產生幾筆虛擬資料    
        torch.set_printoptions(threshold=np.inf)
        
    
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

  