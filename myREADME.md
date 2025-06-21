## Setup
#### Tested with 
 - CUDA 12.1
 - Python 3.10.12
 - Torch 2.1.2
 - environment.yml (my conda env)
 - requirements.txt (modified)

#### Follow README.md with

```
# instead of
git clone https://github.com/Turoad/clrnet
# do
git clone https://github.com/AshwinDisa/CLRNet.git
```

```
# instead of
conda create -n clrnet python=3.8 -y
# do
conda env create -f environment.yml
```

```
# instead of
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# do
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch
```

```
# instead of
python setup.py build develop
# do
pip install -e .
```

## Data and model weights

#### Download test images from [here]() and save as
```
extras/test_images/P3scene7/*.jpg
```
For testing on custom lane dataset create a similar folder in /test_images

#### Download ResNet18 trained on CULane from [here](https://github.com/turoad/CLRNet/releases) and save as
```
models/culane_r18.pth
```
#### Download ONNX and TensortRT engine from [here](https://drive.google.com/drive/folders/1NzC4dBfGmrLrnzQavO6uOUHhdecE-g5I?usp=sharing) and save as
```
models/culane_r18_b1.onnx
models/culane_r18_fp16_b1.trt
```

### Model conversion (optional)
#### .pth to ONNX
```
python main.py configs/clrnet/clr_resnet18_culane.py --convert_to_onnx --gpus 0
```

#### ONNX to trt
```
python3 extras/convert_to_trt.py --onnx models/culane_r18.onnx --engine models/culane_r18_fp16_b1.trt --precision fp16 --workspace 16
```
Code reference from [here](https://medium.com/@maxme006/how-to-create-a-tensorrt-engine-version-10-4-0-ec705013da7c
)

## Inference

#### PyTorch

```
python main.py configs/clrnet/clr_resnet18_culane.py --infer pytorch --model models/culane_r18.pth --image_dir extras/test_images/P3scene7/ --gpus 0
```

#### ONNX CPU
```
python main.py configs/clrnet/clr_resnet18_culane.py --infer onnx_python_cpu --model models/culane_r18_b1.onnx --image_dir extras/test_images/P3scene7/ --gpus 0
```

#### ONNX GPU
```
python main.py configs/clrnet/clr_resnet18_culane.py --infer onnx_python_gpu --model models/culane_r18_b1.onnx --image_dir extras/test_images/P3scene7/ --gpus 0
```

#### TensorRT
```
python main.py configs/clrnet/clr_resnet18_culane.py --infer tensorrt --model models/culane_r18_fp16_b1.trt --image_dir extras/test_images/P3scene7/ --gpus 0
```

Note: ONNX and TensorRT models do not support dynamic batch size for now. Infer only on fixed batch size (1).

#### for arg info
```
python3 main.py -h
```