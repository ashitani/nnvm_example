#!/usr/bin/env python

import numpy as np
import nnvm.compiler
import tvm
from tvm.contrib import graph_runtime, util

from matplotlib import pyplot as plt
import time

def replay_nnvm(target):

    if target=="llvm":
        # llvm
        basename="./models/deploy_llvm"
        ctx=tvm.cpu(0)
    elif target == "opencl":
        # opencl
        basename="./models/deploy_opencl"
        ctx=tvm.context("opencl",0)

    loaded_lib = tvm.module.load(basename+".dylib")
    loaded_json = open(basename+".json").read()
    loaded_params = bytearray(open(basename+".params", "rb").read())

    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    params = nnvm.compiler.load_param_dict(loaded_params)
    module.load_params(loaded_params)

    shape = (1,1)
    x_np = np.linspace(0,1,100).astype("float32")

    times=[]

    outs =np.array([])
    start = time.time()
    for x in x_np:
        module.set_input('input_0', tvm.nd.array(np.array([x]).astype('float32')))
        module.run()
        out=module.get_output(0, out=tvm.nd.empty(shape)).asnumpy()[0]
        outs=np.append(outs,out)

    elasped_time=time.time() - start
    print(elasped_time)

    plt.plot(np.exp(x_np),"b")
    plt.plot(outs,"r")
    plt.show()

if __name__ == '__main__':
    replay_nnvm("llvm")
    replay_nnvm("opencl")


