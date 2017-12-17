import numpy as np
import nnvm.compiler
import tvm
from tvm.contrib import util
from tvm.contrib import util, rpc
from tvm.contrib import graph_runtime as runtime

##---------------------------------
## Under debugging
##---------------------------------

import cv2
import time

inshape = (1,3,224,224)
outshape = (1,1000)

basename="models/alexnet/deploy_rasp"
loaded_params = bytearray(open(basename+".params", "rb").read())

# connect the server
remote = rpc.connect("raspberrypi.local", 9090)

# upload the library to remote device and load it
lib_fname='models/alexnet/deploy_rasp.o'
print("uploading")
remote.upload(lib_fname)
print("loading module")
rlib = remote.load_module("deploy_rasp.o")
print("loading graph")
graph = open(basename+".json").read()

ctx = remote.cpu(0)
# upload the parameter
print("loading paramdict")
params = nnvm.compiler.load_param_dict(loaded_params)
print("converting paramdict")
rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

# create the remote runtime module
print("creating module")
module = runtime.create(graph, rlib, ctx)

# load parameters, and graph

print("loading params")
module.load_params(loaded_params)
print("done")

import cv2
img = cv2.imread("orange.jpg")
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rgb= cv2.resize(rgb,(224,224))
print(np.shape(rgb))
x=rgb.transpose(2,0,1).reshape(inshape)

# set parameter
print("setting params")
module.set_input(**rparams)
# set input data
print("input data")
module.set_input('input_0', tvm.nd.array(x.astype('float32')))
# run
print("running")
module.run()
print("getting output")
# get output
out = module.get_output(0, tvm.nd.empty(outshape, ctx=ctx))
# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))

