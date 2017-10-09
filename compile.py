#!/usr/bin/env python

import mxnet as mx
import numpy as np

from net import *

model=Net()
model.load_params("model",mx.cpu(0))

import nnvm
import nnvm.compiler
import tvm
from tvm.contrib import graph_runtime, util

sym, params = nnvm.frontend.from_mxnet(model)

target = 'llvm'
shape_dict = {'data': (1,1)}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params, dtype="float32")
module = graph_runtime.create(graph, lib, tvm.cpu(0))

lib.export_library("deploy.dylib")
with open("deploy.json", "w") as fo:
    fo.write(graph.json())
with open("deploy.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

