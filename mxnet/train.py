
import mxnet as mx
from mxnet import autograd
import logging
logging.getLogger().setLevel(logging.DEBUG)

import numpy as np
from net import *

model=Net()
model.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y

trainer = gluon.Trainer(model.collect_params(), 'adam')

for i in range(10000):
    with autograd.record():
        x,y = get_batch(100)
        data = mx.nd.array(x).reshape((100,1))
        label = mx.nd.array(y).reshape((100,1))
        output = model(data)
        L = gluon.loss.L2Loss(batch_axis=1) #?!
        loss = L(output, label)
        print loss.asnumpy()
        loss.backward()

    trainer.step(data.shape[0])

model.save_params('model')
