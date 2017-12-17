# -*- coding: utf-8 -*-

import nnvm
import tvm
import onnx
import numpy as np
import nnvm.compiler
from tvm.contrib import graph_runtime, util

import os

def compile(model_name, framework):

    out_folder="models/"+model_name
    onnx_file_name=out_folder+"/"+model_name+".onnx"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    onnx_graph = onnx.load(onnx_file_name)
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    shape=(1,3,224,224)

    if framework=="llvm":
        target="llvm"
        basename=out_folder+"/deploy_llvm"
    elif framework=="opencl":
        target="opencl"
        basename=out_folder+"/deploy_opencl"
    elif framework=="llvm_rasp":
#        target="llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon" # RPi 3
        target="llvm -target=armv6-none-linux-gnueabihf -mcpu=arm1176jzf-s -mattr=+neon" # RPi zero W
        basename=out_folder+"/deploy_rasp"

    shape_dict = {'input_0': shape}

    if framework=="llvm_rasp":

        with tvm.target.rasp():
            deploy_graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        lib.save(basename+".o")
    else:
        deploy_graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        lib.export_library(basename+".dylib")


    with open(basename+".json", "w") as fo:
        fo.write(deploy_graph.json())
    with open(basename+".params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

if __name__ == '__main__':
    from common_lib import *

    if (len(sys.argv)!=3):
        print("Usage: python {} [framework] [network]".format(sys.argv[0]))
        exit(-1)

    framework = sys.argv[1]
    model_name = sys.argv[2]

    print("building :", framework, model_name)
    compile(framework,model_name)
    print("finished")
