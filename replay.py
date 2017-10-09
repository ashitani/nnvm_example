#!/usr/bin/env python

import mxnet as mx
import numpy as np

from matplotlib import pyplot as plt

from net import *

import time

model=Net()
model.load_params("model",mx.cpu(0))

eval_data=np.linspace(0,1,100)

outs=np.array([])
times=[]
for x in eval_data:
    start = time.time()
    out = model(mx.nd.array([x])).asnumpy()
    elasped_time=time.time() - start
    times.append(elasped_time)
    outs=np.append(outs,out)


print np.average(times)

plt.plot(np.exp(eval_data))
plt.plot(outs,"r")

plt.show()

