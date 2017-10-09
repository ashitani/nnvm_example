#!/usr/bin/env python

import numpy as np
import nnvm.compiler
import tvm
import numpy as np
from tvm.contrib import graph_runtime, util

from matplotlib import pyplot as plt
import numpy as np

import time


loaded_lib = tvm.module.load("deploy.dylib")
loaded_json = open("deploy.json").read()
loaded_params = bytearray(open("deploy.params", "rb").read())

module = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu(0))
params = nnvm.compiler.load_param_dict(loaded_params)
module.load_params(loaded_params)

shape = (1,1)
x_np = np.linspace(0,1,100).astype("float32")

times=[]

outs =np.array([])
for x in x_np:
    start = time.time()
    module.run(data=np.array([x]))
    elasped_time=time.time() - start
    times.append(elasped_time)
    out=module.get_output(0, out=tvm.nd.empty(shape)).asnumpy()[0]
    outs=np.append(outs,out)

print np.average(times)


plt.plot(np.exp(x_np),"b")
plt.hold(True)
plt.plot(outs,"r")
plt.show()


