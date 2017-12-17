# Usage(MXNet)

Example to infer exp() function.

## train.py

Train net and save to the file "model".

## replay.py

Infer the function exp() using MXNet and the file "model".

## compile.py

Convert "model" to the dylib and parameters using NNVM.
Output files are "deploy.dylib", "deploy.json" and "deploy.params".

## replay_nnvm.py

Infer the function exp() using "deploy.\*" files.

