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
import datetime
import pickle as pk
from torchvision.datasets import ImageFolder
from scipy.stats import multinomial
from numpy.random import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.decomposition import PCA

from PIL import Image

EPOCH = 1
LR = 1e-4      # learning rate
DOWNLOAD_DATA = False
prototype = 50

def sort_data(img, label):    #因為train_data是沒有順序的
    argsort_label = torch.argsort(label)
    sorted_img = torch.empty(img.size())
    sorted_label = torch.empty(label.size())
    for i in range(len(argsort_label)):
        sorted_img[i] = img[argsort_label[i]]
        sorted_label[i] = label[argsort_label[i]]
    
    return sorted_img.numpy(), sorted_label.numpy()

def NCM_result(autoencoder, train_loaders, BATCH_SIZE, TRAIN_SAMPLE, TRAIN_CLASS):
    
    #autoencoder = torch.load(model_name)
    autoencoder = autoencoder.eval()
    device = torch.device("cuda")
    autoencoder.to(device)
    #model = make_model()

    
    with torch.no_grad():
        
        all_features = torch.empty(TRAIN_SAMPLE, prototype)
        all_labels = torch.empty(TRAIN_SAMPLE, )
        accu_data = 0
        data_per_class = np.zeros(TRAIN_CLASS)
        
        for step, (x, b_label) in enumerate(train_loaders):
            
            #features = extract_feature(model, x).view((BATCH_SIZE, 8192 )).to(device)
            #print(features.size())
            torch.cuda.empty_cache()
            x = x.view(-1, 3, 32, 32).to(device)
            encoded_array, _,_,_ = autoencoder(x)
            #train_encode, _,_,_ = autoencoder(x)
            train_encode = encoded_array[2]
            
            for j in range(train_encode.size()[0]):
                #print(train_encode[j])
                all_features[accu_data + j] = train_encode[j]
                all_labels[accu_data + j] = b_label[j]
                
                # 累加各類別的張數分別有多少張
                data_per_class[b_label[j]] += 1
            # 目前有多少筆data
            accu_data += train_encode.size()[0]
        
        torch.set_printoptions(threshold=np.inf)

        
    

    # 把訓練資料按照類別排好
    sort_feature, sort_label = sort_data(all_features, all_labels)
    
    # 降維
    '''start = datetime.datetime.now()
    pca = PCA(n_components=1000)
    pca.fit(sort_feature)
    pca_feature = pca.transform(sort_feature)
    with open('pca_50.pkl', 'wb') as pickle_file:
        pk.dump(pca, pickle_file)
    end = datetime.datetime.now()
    print('pca執行時間: {}'.format(str(end-start)))
    
    red_dim = 1000
    # calculate mean of encodes
    pca_mean_feature = np.zeros([TRAIN_CLASS, red_dim])
    #cov_feature = np.zeros([TRAIN_CLASS, prototype, prototype])
    pca_cov_feature = np.zeros([TRAIN_CLASS, red_dim])'''
    
    mean_feature = np.zeros([TRAIN_CLASS, prototype])
    cov_feature = np.zeros([TRAIN_CLASS, prototype, prototype])
    

    init = 0
    for i in range(TRAIN_CLASS):
        # 計算目標索引值
        des = int(init + data_per_class[i])
        # 在計算平均之前先做正規化
        #Min_num, Max_num = sort_feature[init:des].min(), sort_feature[init:des].max()
        # (矩阵元素-最小值)/(最大值-最小值)
        #normalize_feature = (sort_feature[init:des] - Min_num)/(Max_num - Min_num))
        #mean_feature[i] = np.mean(normalize_feature, axis = 0)
        
        #pca_mean_feature[i] = np.mean(pca_feature[init:des], axis = 0)
        #pca_cov_feature[i] = np.std(pca_feature[init:des], axis = 0)
        mean_feature[i] = np.mean(sort_feature[init:des], axis = 0)
        #cov_feature[i] = np.std(sort_feature[init:des], axis = 0)

        cov_feature[i] = np.cov(sort_feature[init:des].T)

        # 更新初始索引值
        init += int(data_per_class[i])
    
    np.set_printoptions(threshold=np.inf)
    #print(mean_feature[0])
    #print(cov_feature[0])

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