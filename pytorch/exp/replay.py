#!/usr/bin/env python

import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import time
import numpy as np

from net import Net

model= torch.load("models/exp.pth")

shape = (1,1)
x_np = np.linspace(0,1,100).astype("float32")

times=[]

outs =np.array([])
start = time.time()
for x in x_np:
    x_= Variable(torch.from_numpy(x.astype(np.float32).reshape(1,1,1)))
    output = model(x_)
    outs=np.append(outs,output[0])

elasped_time=time.time() - start
print(elasped_time)

plt.plot(np.exp(x_np),"b")
plt.plot(outs,"r")
plt.show()

