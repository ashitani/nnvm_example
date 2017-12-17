import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx

from net import Net

#from matplotlib import pyplot as plt
import numpy as np

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y

model = Net()
optimizer = optim.Adam(model.parameters())

losses =[]
for i in range(3000):
    x,y = get_batch(100)
    x_= Variable(torch.from_numpy(x.astype(np.float32).reshape(100,1,1)))
    t_= Variable(torch.from_numpy(y.astype(np.float32).reshape(100,1,1)))

    optimizer.zero_grad()
    output = model(x_)
    mseloss = nn.MSELoss()
    loss=mseloss(output,t_)
    print(loss.data[0])
    losses.append(loss.data[0])
    loss.backward()
    optimizer.step()


# export

x_= Variable(torch.randn(1,1),requires_grad=True)
model.train(False)

torch.save(model, './models/exp.pth')


torch_out = torch.onnx._export(model,
                               x_,                       # model input (or a tuple for multiple inputs)
                               "./models/exp.onnx",              # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file


