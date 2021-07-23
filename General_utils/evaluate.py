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
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal

EPOCH = 1
BATCH_SIZE = 100
LR = 1e-4      # learning rate
DOWNLOAD_DATA = True
TEST_SAMPLE = 5000
#TEST_CLASS = 2
yourPath = './divide_cifar50_test'


def accuracy_result(autoencoder, test_loaders, TEST_SAMPLE, TEST_CLASS):
    #TEST_CLASS = 2

    autoencoder = autoencoder.eval()
    device = torch.device("cuda")
    cpu = torch.device("cpu")
    autoencoder.to(device)

    mean_feature = np.load('./General_utils/prev_feature_mean.npy')
    cov_feature = np.load('./General_utils/prev_feature_cov.npy')
    mean_feature = torch.from_numpy(mean_feature).to(device)
    cov_feature = torch.from_numpy(cov_feature).to(device)
    
    with torch.no_grad():
        
        predict_class = torch.zeros(TEST_SAMPLE).to(device)
        all_labels = torch.empty(TEST_SAMPLE, ).to(device)
        #predict_class = np.empty(TEST_SAMPLE)
        #all_labels = np.empty(TEST_SAMPLE, )
        acc_data = 0
        start = datetime.datetime.now()
        
        
        
        for step, (x, b_label) in enumerate(test_loaders):
            #print(step)
            b_start = datetime.datetime.now()
            x = x.view(-1, 3, 32, 32).to(device)

   
            
            #test_encode, _, _, _ = autoencoder(x)
            encoded_array, _, _, _ = autoencoder(x)
            test_encode = encoded_array[2]
            #test_encode = encoded_array[1]
            #test_encode = test_encode.to(cpu)
            #print(test_encode.size())
            data_size = test_encode.size()[0]
            
            
            
            # sort_feature做PCA轉換
            '''test_encode = test_encode.to(cpu).numpy()
            pca = pk.load(open("pca.pkl",'rb'))
            pca_encode = pca.transform(test_encode)
            pca_encode = torch.from_numpy(pca_encode).to(device)
            
            #distribution = np.empty((TEST_CLASS, data_size))'''
            
            
            '''all_labels[acc_data:acc_data+data_size] = b_label#.numpy()
            for i in range(TEST_CLASS):
                
                
                m = MultivariateNormal(mean_feature[i], cov_feature[i])
                distribution[i] = m.log_prob(test_encode)
                # save probability
                #distribution[i] = multivariate_normal.pdf(test_encode.numpy(), mean=mean_feature[i], cov = cov_feature[i], allow_singular = True)
            
            predict_class[acc_data:acc_data + data_size] = torch.argmax(distribution, dim = 0)
            torch.set_printoptions(threshold=np.inf)
            print(predict_class)'''
            distribution = torch.empty((TEST_CLASS, data_size))#.to(device)
            all_labels[acc_data:acc_data+data_size] = b_label
            #distribution = np.empty(TEST_CLASS,data_size)
            for i in range(TEST_CLASS):
                m = MultivariateNormal(mean_feature[i], cov_feature[i])
                distribution[i] = m.log_prob(test_encode)
                #distribution[i] = torch.norm(torch.div(torch.pow((test_encode - mean_feature[i]), 2), cov_feature[i]), dim = 1)
                #distribution[i] = torch.norm(torch.div(torch.pow((test_encode - mean_feature[i]), 2), cov_feature[i]), dim = 1)
                
                #cos_dis[i] = torch.cosine_similarity(test_encode[j], mean_feature[i], dim = 0)
                
            predict_class[acc_data:acc_data + data_size] = torch.argmax(distribution, dim=0)
            
            
            acc_data += data_size
            
            b_end = datetime.datetime.now()
            #print('batch執行時間: {}'.format(str(b_end-b_start)))
            
        end = datetime.datetime.now()
        print('執行時間: {}'.format(str(end-start)))

        
        #print('################################ test result ####################################')
        #print(accuracy_score(all_labels.numpy(), predict_class.numpy()))
        #print(accuracy_score(all_labels.numpy(), predict_class.numpy(), normalize = False))
        return accuracy_score(all_labels.to(cpu).numpy(), predict_class.to(cpu).numpy())
        #return accuracy_score(all_labels, predict_class)

    
if __name__ == '__main__':
    accuracy_result('autoencoder.pkl')