from common_lib import *

import sys

def normalize(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for c in range(3):
        img[:,:,c] =  (img[:,:,c]-mean[c])/std[c]
    return img

def img2dat(img):
    inshape=(1,3,224,224)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    rgb = normalize(rgb)
    rgb = cv2.resize(rgb,(224,224))
    dat = rgb.transpose(2,0,1)
    dat = dat.reshape(inshape)

    return dat

def picture_demo(framework, model_name, img):

    if framework=="pytorch":

        torch_model =model_dict[model_name](pretrained=True)
        inshape=(1,3,224,224)

        dat=torch.from_numpy(img2dat(img))

        start_time = time.time()

        x = Variable(dat)

        out=torch_model(x)
        result=class_dict[np.argmax(out.data[0])]

        elapsed_time = time.time() - start_time

        return result,elapsed_time

    elif framework=="llvm" or framework=="opencl":
        import nnvm.compiler
        import tvm
        from tvm.contrib import graph_runtime, util

        model_folder="models/"+model_name+"/"

        if framework=="llvm":
            basename=model_folder+"deploy_llvm"
            ctx=tvm.cpu(0)
        elif framework=="opencl":
            basename=model_folder+"deploy_opencl"
            ctx=tvm.context("opencl",0)

        loaded_lib = tvm.module.load(basename+".dylib")
        loaded_json = open(basename+".json").read()
        loaded_params = bytearray(open(basename+".params", "rb").read())

        module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        params = nnvm.compiler.load_param_dict(loaded_params)
        module.load_params(loaded_params)
        inshape = (1,3,224,224)
        outshape = (1,1000)

        dat=img2dat(img)

        start_time = time.time()

        module.set_input('input_0', tvm.nd.array(dat))
        module.run()
        out=module.get_output(0, out=tvm.nd.empty(outshape))
        out=out.asnumpy()
        result=class_dict[np.argmax(out)]

        elapsed_time = time.time() - start_time

        return result, elapsed_time


if __name__ == '__main__':

    if (len(sys.argv)!=4):
        print("Usage: python {} [image_filename] [framework] [network]".format(sys.argv[0]))
        exit(-1)

    image_name = sys.argv[1]
    framework = sys.argv[2]
    model_name = sys.argv[3]

    img = cv2.imread(image_name)

    result,elapsed_time = picture_demo(framework, model_name,img)
    print()
    print("configuration: "+framework+", ",model_name)
    print("image_file: ",image_name)
    print("result: ",result)
    print("elasped_time: {}".format(elapsed_time))
    print()
