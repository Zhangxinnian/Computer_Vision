# Image_classification
Image classification implemented using resnet18 network built by mxnet.
For mxnet documentation, please visit [Mxnet](https://cv.gluon.ai/contents.html) .
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
Ubuntu18.04
CUDA
Python3.6

Please install the correct version of cuda and python according to your own situation, cuda10.0 and python3.6.8 are used here.
### Installing
mxnet
gluoncv
opencv_python

Please refer to the version of CUDA to install the GPU version of mxnet. The operating environment required by the project has been exported to requirements.txt. Please use 
```
pip install -r requirements.txt
```
 to install.
## Running and Tests
 Explain how to run training scripts and test scripts.
 ### Training your own model on ImageNet
```
python ./train_ImageNet.py
```
### Training yuour own model on cifar10
```
python train_cifar10.py --num-epochs 240 --mode hybrid --num-gpus 1 -j 8 --batch-size 128 --wd 0.0001  --lr 0.1  --lr-decay-epoch 80, 160 --model cifar_resnet20_v1
```
### Training your dataset
```
python train_your_dataset.py --num-epochs 40  --num-gpus 1 -j 8 --batch-size 32 --wd 0.0001  --lr 0.0001  --lr-decay-epoch 10,20,30 --model resnet18_v2
```
### Test
#### Test by loading the json file 
Take ImageNet's pre-training model as an example, using json files and param files for prediction.
```
python ./test_load_json.py
```
#### Test by loading the params file
Use the params file after training on the minc-2500 dataset as an example.
```
python ./test_params.py
```
## Tools
Split the data set and export the json file of the network.
### Split dataset
Take the minc-2500 data set as an example to make labeling and segmentation data sets.
```
python ./tools/split_your_dataset.py --data (your own dataset path) --split 1
```
### Export json file
Here is an example of the model trained on cifar10. The comment in the file contains the part that was exported and trained on its own data set.
```
python ./tools/export_json.py
```
## Convert to Ncnn
Please compile Ncnn according to the requirements of [Ncnn](https://github.com/Tencent/ncnn)  official website.
```
cd ncnn/build/tools/mxnet
./mxnet2ncnn your_model.json your_model.params model.param model.bin
```
Use ncnnoptimize to optimize the model.
```
cd ncnn/build/tools
./ncnnoptimize model.param model.bin model_opt.param model_opt.bin 1
```
The 0 at the end refers to fp32, and 1 refers to fp16.









