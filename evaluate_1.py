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
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score

def accuracy_result(autoencoder, test_loaders, TEST_SAMPLE, TEST_CLASS):
    #print("~~~~~~~~~~~~~~~~TEST CLASS~~~~~~~~~~~~~~~~~")
    #print(TEST_CLASS-1)
    TEST_CLASS -= 1
    
    #autoencoder = torch.load(model_name)
    autoencoder = autoencoder.eval()
    device = torch.device("cuda")
    cpu = torch.device("cpu")
    autoencoder.to(device)

    mean_feature = np.load('./General_utils/current_feature_mean.npy')
    mean_feature = torch.from_numpy(mean_feature).to(device)
    
    cov_feature = np.load('./General_utils/current_feature_cov.npy')
    cov_feature = torch.from_numpy(cov_feature).to(device)
    #print("feature_size")
    #print(mean_feature.size())
    
    with torch.no_grad():
        
        predict_class = torch.zeros(TEST_SAMPLE)
        predict_class_classify = torch.zeros(TEST_SAMPLE)
        all_labels = torch.zeros(TEST_SAMPLE, )
        acc_data = 0
        numbers = 0
        
        # 判斷新任務和舊任務的分界
        class_bound = TEST_CLASS // 10 * 10 - 1
        print(TEST_CLASS)
        print('### Class_bound : {} ###'.format(class_bound))
        TEST_CLASS = int(len(mean_feature))
        
        m = [MultivariateNormal(mean_feature[0], cov_feature[0])]
        for i in range(1, TEST_CLASS):
            m.append(MultivariateNormal(mean_feature[i], cov_feature[i]))
        
        
        for step, (x, b_label) in enumerate(test_loaders):
            #print(b_label)
            #print("------------------------")
            x = x.view(-1, 3, 32, 32).to(device)   # batch x, shape (batch, 28*28)
            
            # 階層一分類(New task or Old task)
            #test_encode, test_decode, classify, _ = autoencoder(x)
            encoded_array, decoded_array, classify, _ = autoencoder(x)
            test_encode = encoded_array[2]

            _, preds = torch.max(classify, 1)
            
            data_size = test_encode.size()[0]    # 該batch的數量
            
            all_labels[acc_data:acc_data+data_size] = b_label
            
            
            for j in range(data_size):
                # 儲存ground truth的值
               
                
                if preds[j] > 0:    # new task
                    predict_class[acc_data + j] = class_bound + preds[j]
                    predict_class_classify[acc_data + j] = class_bound + preds[j]
                else:    # old task, using old classifier to classify
                    distribution = torch.zeros(TEST_CLASS)
                    numbers += 1
                    predict_class_classify[acc_data + j] = torch.argmax(test_encode[j])
                    for i in range(TEST_CLASS):
                        #m = MultivariateNormal(mean_feature[i], cov_feature[i])
                        distribution[i] = m[i].log_prob(test_encode[j])
                    predict_class[acc_data + j] = torch.argmax(distribution)
            
            acc_data += data_size
            
        #print(predict_class)
        torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None)
      
        running_corrects = torch.sum(predict_class == all_labels.data)
        #print('判斷正確的張數: {}'.format(running_corrects))
        classifier_acc = accuracy_score(all_labels.numpy(), predict_class_classify.numpy())
        NCM_acc = accuracy_score(all_labels.numpy(), predict_class.numpy())
        Final_acc = 0
        if classifier_acc > NCM_acc:
            Final_acc = classifier_acc
        else:
            Final_acc = NCM_acc
        
        print('classifier acc: {}'.format(accuracy_score(all_labels.numpy(), predict_class_classify.numpy())))
        print('NCM acc: {}'.format(accuracy_score(all_labels.numpy(), predict_class.numpy())))
        #print('################################ test result ####################################')

        return Final_acc, running_corrects

    
if __name__ == '__main__':
    accuracy_result('autoencoder.pkl')