
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
import numpy as np
class MNIST_Net(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256):
        super(MNIST_Net, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
class MNIST_Net_buttelNeck(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256,buttel_neck=10):
        super(MNIST_Net_buttelNeck, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, buttel_neck),
            nn.ReLU(inplace=True),
            nn.Linear(buttel_neck, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x
        
class AutoEncoder(nn.Module):
    def __init__(self,num_classes=2048,hidden_size=2048):    # hidden layer用256會梯度爆炸
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(8192, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, num_classes))
        
        self.decoder = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 8192))
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded
        
class VGG19_AutoEncoder(nn.Module):
    def __init__(self,num_classes=2048,hidden_size=2048):    # hidden layer用256會梯度爆炸
        super(VGG19_AutoEncoder, self).__init__()
        
        self.VGG_feature = models.vgg19(pretrained=True).features[:21]
        for name, value in self.VGG_feature.named_parameters():
            value.requires_grad = False
        self.encoder2 = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ELU(inplace=False))
            
        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 2048))
            
        self.encoder = nn.Sequential(
            nn.ELU(inplace=False),
            nn.Linear(2048, 50))
        
        self.decoder = nn.Sequential(
            nn.Linear(50, 2048),
            nn.ELU(inplace=False),
            nn.Linear(2048, 4096),
            nn.ELU(inplace=False),
            nn.Linear(4096, 8192))
            
            
        self.classifier = nn.Sequential(
            #nn.Linear(num_classes, 1000),
            nn.ELU(inplace=False),    # 負值為0，會減少神經元輸出(改成ELU試試)
            #nn.Dropout(0.5),
            nn.Linear(2048, 2))
        
        '''self.training = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Linear(num_classes, 1)'''
            
    def forward(self, x):
        features = self.VGG_feature(x).view(-1, 8192)
        encoded = self.encoder2(features)
        encoded1 = self.encoder1(encoded)
        encoded = self.encoder(encoded1)
        decoded = self.decoder(encoded)
        classify = self.classifier(encoded1)
        
        return encoded, decoded, classify, features
        
class Fearnet(nn.Module):
    def __init__(self,num_classes=50,hidden_size=2048):    # hidden layer用256會梯度爆炸
        super(Fearnet, self).__init__()
        
        self.VGG_feature = models.vgg19(pretrained=True).features[:21]
        for name, value in self.VGG_feature.named_parameters():
            value.requires_grad = False
            
        self.encoder1 = nn.Sequential(
            nn.Linear(8192, 4096))
            
        self.encoder2 = nn.Sequential(
            nn.ELU(inplace = False),
            nn.Linear(4096, hidden_size))
            
        self.encoder3 = nn.Sequential(
            nn.ELU(inplace = False),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim = 1))
            
        self.decoder1 = nn.Sequential(
            nn.Linear(num_classes, hidden_size))
            
        self.decoder2 = nn.Sequential(
            nn.ELU(inplace = False),
            nn.Linear(hidden_size, 4096))
            
        self.decoder3 = nn.Sequential(
            nn.ELU(inplace = False),
            nn.Linear(4096, 8192))
            
            
        self.classifier = nn.Sequential(
            #nn.Linear(num_classes, 1000),
            nn.ELU(inplace=False),    # 負值為0，會減少神經元輸出(改成ELU試試)
            #nn.Dropout(0.5),
            nn.Linear(8192, 2))
        
            
    def forward(self, x):
        encoded = []
        decoded = []
        features = self.VGG_feature(x).view(-1, 8192)
        
        feature_max, _ = torch.max(features, 1)
        feature_max = feature_max.unsqueeze(dim=1)
        feature_min, _ = torch.min(features, 1)
        feature_min = feature_min.unsqueeze(dim=1)
        features_nor = (features - feature_min) / (feature_max - feature_min)
        
        
        
        encoded1 = self.encoder1(features_nor)
        encoded.append(encoded1)
        
        encoded2 = self.encoder2(encoded1)
        encoded.append(encoded2)
        
        encoded3 = self.encoder3(encoded2)
        encoded.append(encoded3)
        
        decoded1 = self.decoder1(encoded3)
        decoded.append(decoded1)
        
        decoded2 = self.decoder2(decoded1)
        decoded.append(decoded2)
        
        decoded3 = self.decoder3(decoded2)
        decoded.append(decoded3)
        
        classify = self.classifier(features_nor)
        
        return encoded, decoded, classify, features_nor