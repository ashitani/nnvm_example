# nnvm_example

NNVM Example.
For detail, please read the [Qiita Entry](https://qiita.com/ashitani/items/e85231297247ec036128).

# Requirements

- NNVM and TVM
- MXNet

# Usage

## train.py

Train net and save to the file "model".

## replay.py

Infer the function exp() using MXNet and the file "model".

## compile.py

Convert "model" to the dylib and parameters using NNVM.
Output files are "deploy.dylib", "deploy.json" and "deploy.params".

## replay_nnvm.py

Infer the function exp() using "deploy.*" files.

