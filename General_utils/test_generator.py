import torch
import torch.nn
from torch.utils.data import DataLoader as loader
from Data_generate import *
from itertools import cycle


total_sample = [500,500, 500, 500]
class_num = [50, 10, 10, 10]
#total_sample = [500]
#class_num = [50]
root=''

dataset = Prototype_generate(total_sample=total_sample, class_num=class_num, root=root)

data_loader = loader(dataset, batch_size = 64, shuffle = True, num_workers = 0)

data_loader1 = loader(dataset, batch_size = 200, shuffle = True, num_workers = 0)

for step, data in enumerate(zip(data_loader, cycle(data_loader1))):
    print("-----------------")
    print(data[0][0])   # data_loader的image
    print(data[0][1])   # data_loader的label
    print()
    print(data[1][0])   # data_loader1的image
    print(data[1][1])   # data_loader1的label