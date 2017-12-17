# Usage(PyTorch/images)

## export.py

Export pretrained pytorch model to the onnx file in the "models" folder.

Usage:

```
python export.py [network]
```

Example:

```
python export.py vgg16
```

## compile.py

Compile onnx files to the TVM dynamic link library.

Usage:

```
python compile.py [framework] [network]
```

Example:

```
python compile.py opencl vgg16
```

## replay.py

Infer the models.

Usage:

```
python replay.py [image_filename] [framework] [network]
```

Example:

```
python raplay.py orange.jpg opencl vgg16
```

