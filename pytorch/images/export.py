from common_lib import *

def export_from_pytorch(model_name):

    out_folder="models/"+model_name
    onnx_file_name=model_name+".onnx"
    target_path=out_folder+"/"+onnx_file_name

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    torch_model =model_dict[model_name](pretrained=True)

    torch_model.train(False)
    batch_size=1
    x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)

    torch_out = torch.onnx._export(torch_model,
                               x,
                               target_path,
                               export_params=True)

if __name__ == '__main__':

    if (len(sys.argv)!=2):
        print("Usage: python {} [network]".format(sys.argv[0]))
        exit(-1)

    model_name = sys.argv[1]
    print("exporting : models/"+model_name+"/"+model_name+".onnx")
    export_from_pytorch(model_name)
