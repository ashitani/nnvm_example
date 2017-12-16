#!/usr/bin/env python

from mxnet import gluon
from mxnet.gluon import nn

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(64)
            self.fc2 = nn.Dense(256)
            self.fc3 = nn.Dense(1)

    def hybrid_forward(self, F, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y
