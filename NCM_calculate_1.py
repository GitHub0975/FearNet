import os
import numpy as np
 
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as Data
import pickle as pk
import datetime

from sklearn.decomposition import PCA
from torchvision.datasets import ImageFolder

from PIL import Image

#prototype = 50
pca_prototype = 1000

def sort_data(img, label):    #因為train_data是沒有順序的
    argsort_label = torch.argsort(label)
    sorted_img = torch.empty(img.size())
    sorted_label = torch.empty(label.size())
    for i in range(len(argsort_label)):
        sorted_img[i] = img[argsort_label[i]]
        sorted_label[i] = label[argsort_label[i]]
    
    return sorted_img.numpy(), sorted_label.numpy()

def NCM_result(autoencoder, train_loaders, TRAIN_SAMPLE, TRAIN_CLASS, exp_dir):
    
    #autoencoder = torch.load(model_name)
    autoencoder = autoencoder.eval()
    device = torch.device("cuda")
    autoencoder.to(device)
    prototype = TRAIN_CLASS
    #model = make_model()
    print('Train_samples: {}'.format(TRAIN_SAMPLE))
    print('Train_classes: {}'.format(TRAIN_CLASS))

    
    with torch.no_grad():
        
        all_features = torch.empty(TRAIN_SAMPLE, prototype)
        all_labels = torch.empty(TRAIN_SAMPLE, )
        accu_data = 0
        data_per_class = np.zeros(TRAIN_CLASS)
        
        for step, (x, b_label) in enumerate(train_loaders):
            
            #features = extract_feature(model, x).view((BATCH_SIZE, 8192 )).to(device)
            #print(features.size())
            torch.cuda.empty_cache()
            x = x.view(-1, 8192).to(device)
            train_encode = autoencoder.encoder3(autoencoder.encoder2(autoencoder.encoder1(x)))
            
            for j in range(train_encode.size()[0]):
                #print(train_encode[j])
                all_features[accu_data + j] = train_encode[j]
                all_labels[accu_data + j] = b_label[j]
                #print(b_label[j])
                
                # 累加各類別的張數分別有多少張
                data_per_class[b_label[j]] += 1
            # 目前有多少筆data
            accu_data += train_encode.size()[0]
                

    # 把訓練資料按照類別排好
    sort_feature, sort_label = sort_data(all_features, all_labels)
    
    
    mean_feature = np.zeros([TRAIN_CLASS, prototype])
    cov_feature = np.zeros([TRAIN_CLASS, prototype, prototype])
    

    init = 0
    for i in range(TRAIN_CLASS):
        # 計算目標索引值
        des = int(init + data_per_class[i])
        
        mean_feature[i] = np.mean(sort_feature[init:des], axis = 0)
        #cov_feature[i] = np.std(sort_feature[init:des], axis = 0)
        cov_feature[i] = np.cov(sort_feature[init:des].T)
        # 更新初始索引值
        init += int(data_per_class[i])
    
    np.set_printoptions(threshold=np.inf)
    #print(mean_feature[0])
    #print(cov_feature[0])

    '''np.save(exp_dir + '/prev_feature_mean', mean_feature)
    np.save(exp_dir + '/prev_feature_cov', cov_feature)
    np.save(exp_dir + '/current_feature_mean', mean_feature)
    np.save(exp_dir + '/current_feature_cov', cov_feature)
    np.save(exp_dir + '/prev_feature_mean_copy', mean_feature)
    np.save(exp_dir + '/prev_feature_cov_copy', cov_feature)'''
    
    np.save('./General_utils/prev_feature_mean', mean_feature)
    np.save('./General_utils/prev_feature_cov', cov_feature)
    np.save('./General_utils/current_feature_mean', mean_feature)
    np.save('./General_utils/current_feature_cov', cov_feature)
    np.save('./General_utils/prev_feature_mean_copy', mean_feature)
    np.save('./General_utils/prev_feature_cov_copy', cov_feature)
    
    '''np.save('./General_utils/pca_prev_feature_mean', pca_mean_feature)
    np.save('./General_utils/pca_prev_feature_cov', pca_cov_feature)
    np.save('./General_utils/pca_current_feature_mean', pca_mean_feature)
    np.save('./General_utils/pca_current_feature_cov', pca_cov_feature)'''

    
if __name__ == '__main__':
    NCM_result('autoencoder.pkl')