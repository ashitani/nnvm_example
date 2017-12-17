# nnvm_example

NNVM Example.
For detail, please read the [Qiita Entry for MXNet](https://qiita.com/ashitani/items/e85231297247ec036128) or [the entry for pyTorch](TBD)

# Requirements

- NNVM and TVM
- MXNet or PyTorch

# Usage(PyTorch/images)

## export.py

Export pretrained pytorch model to the onnx file in the "models" folder.

## compile.py

Compile onnx files to the TVM dynamic link library.

## replay.py

Infer the models.

# Usage(MXNet)

## train.py

Train net and save to the file "model".

## replay.py

Infer the function exp() using MXNet and the file "model".

## compile.py

Convert "model" to the dylib and parameters using NNVM.
Output files are "deploy.dylib", "deploy.json" and "deploy.params".

## replay_nnvm.py

Infer the function exp() using "deploy.\*" files.

