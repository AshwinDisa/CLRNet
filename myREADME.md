### Prerequisites
Tested with 
 - CUDA 12.1
 - Python 3.10.12
 - Torch 2.1.2
 - modified requirements.txt 

### Modified commands, everything else remains same

```
instead of
git clone https://github.com/Turoad/clrnet
do
git clone https://github.com/AshwinDisa/CLRNet.git
```

```
instead of
conda create -n clrnet python=3.8 -y
do
conda env create -f environment.yml
```

```
instead of
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
do
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch
```

```
instead of
python setup.py build develop
do
pip install -e .
```

### Download pretrained weights and save as
```
models/culane_r18.pth
```

### .pth to ONNX
```
python main.py configs/clrnet/clr_resnet18_culane.py --convert_to_onnx --gpus 0
```

### Infer Pytorch
```
python main.py configs/clrnet/clr_resnet18_culane.py --infer pytorch --model models/culane_r18.pth --image_dir extras/test_images/scene2/ --gpus 0
```

### Infer ONNX Python CPU
```
python main.py configs/clrnet/clr_resnet18_culane.py --infer onnx_python_cpu --model models/culane_r18.onnx --image_dir extras/test_images/scene2/ --gpus 0
```

### Infer ONNX Python GPU
```
python main.py configs/clrnet/clr_resnet18_culane.py --infer onnx_python_gpu --model models/culane_r18.onnx --image_dir extras/test_images/scene2/ --gpus 0
```
