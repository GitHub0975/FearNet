import torch
import matplotlib.pyplot as plt
from numpy.random import normal, uniform, exponential, randint

mean = torch.zeros((2, ))
std = torch.zeros((2,)) + 1
print(mean)
print(std)
n = torch.distributions.Normal(mean,std)
instance = n.sample((100, ))
#print(instance)

#plt.plot(instance)

#plt.show()

sample = []
for i in range(100):
    vertex = torch.normal(mean,std)
    sample.append(vertex.numpy().tolist())
    
#plt.plot(sample)
#plt.show()

sample1 = []
for i in range(100):
    x = normal()
    print(x)
    sample1.append(x)
    
plt.plot(sample1)
plt.show()