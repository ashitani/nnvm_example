# -*- coding: utf-8 -*-

import nnvm
import tvm
import onnx
import numpy as np
import nnvm.compiler
from tvm.contrib import graph_runtime, util

import os

def link_shared_arm(file_name, files):
    print(file_name)
    print(files)

def compile(model_name, target_name):

    out_folder="models"
    onnx_file_name=out_folder+"/"+model_name+".onnx"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    onnx_graph = onnx.load(onnx_file_name)
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    shape=(1,1)


    if target_name=="llvm":
        target="llvm"
        basename=out_folder+"/deploy_llvm"
    elif target_name=="opencl":
        target="opencl"
        basename=out_folder+"/deploy_opencl"
    elif target_name=="llvm_rasp":
#        target="llvm -device=rasp -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"
        target="llvm -device=rasp -mtriple=armv6l-none-linux-gnueabihf -mcpu=arm1176jzf-s -mattr=+neon" #RPi zero W
#        target=tvm.target.rasp("-target=armv6-none-linux-gnueabihf -mcpu=arm1176jzf-s  -mattr=+neon") #RPi zero W
    #    target="llvm" # -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"
        basename=out_folder+"/deploy_rasp"

#target = tvm.target.rasp("--system-lib -target=armv7l-hisiv600-linux-gnueabi -mcpu=cortex-a17 -mfloat-abi=soft -mattr=+neon")

    shape_dict = {'input_0': shape}

    if target_name=="llvm_rasp":
        #with tvm.target.rasp():
        deploy_graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        lib.save(basename+".o")
        lib.export_library(basename+".so",fcompile=link_shared_arm)
    else:
        deploy_graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        lib.export_library(basename+".dylib")

    # with tvm.target.rasp():
    #     print("1-")
    #     deploy_graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
    #     print("2-")
    #     lib.save(basename+".o")

    with open(basename+".json", "w") as fo:
        fo.write(deploy_graph.json())
    with open(basename+".params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

if __name__ == '__main__':
    #from common_lib import *

    compile("exp","llvm")
    compile("exp","opencl")
    #compile("exp","llvm_rasp")

