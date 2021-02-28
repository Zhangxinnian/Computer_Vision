# Target Detection 
Target Detection implemented using yolov3_mobilnet network built by mxnet.
[Articles](https://zhuanlan.zhihu.com/p/141971141) referenced at 2ncnn.
My sincere thanks to the author for my guidance in asking questions.
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
Ubuntu18.04
mxnet
CUDA
Ncnn
Python3.6

Please install the correct version of cuda  and python according to your own situation, cuda10.0 and python3.6.8 are used here.
### Installing
Ncnn
mxnet
gluoncv
opencv_python

Please refer to ncnn official website official website for installation.
[Ncnn](https://github.com/Tencent/ncnn)
```
pip install -r requirements.txt
```

## Running and Tests
Explain how to run training scripts and test scripts.
### Doenload datasets
Here use PASCAL VOC datasets as an example.
```
python download_dataset.py
```
if you already have the above files sitting on your disk, you can set --download-dir to ponit to them.
For expamle, assuming the files are saved in ~/Vocdevkit/,you can run:
```
python download_dataset.py --download-dir ~/Vocdevkit
```

### Training
Here we take yolov3_mobilnetv1 as an example, and the data uses VOC.
```
python train_yolov3 --network mobilenet1.0 --dataset voc --log-interval 100 --lr-decay-epoch 100,200,300,400 --epochs 500
--syncbn --warmup-epochs 4 --batch-size 8
--data-shape 416 --no-random-shape
--dataset voc
--gpus 0 -j 4 --lr 0.000125
```
### Test
Load the model directly from the model library for testing (coco data set).
```
python example_demo.py
```
Use the trained model to make predictions.
```
python predict.py
```

### 2ncnn
Take yolo3_mobilenet1.0_coco as an example.
#### Export  json and params
```
python ./tools/export_Symbol.py
```
#### Filter the Symbol parameter and remove the prior box
```
python ./tools/prior_box.py
```
#### Modify the source code of yolo3.py under gluoncv
Location python installation directory/Lib/site-packages/gluoncv/model_zoo/yolo/yolo3.py Find the following code block:
```
 # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i + 1]
            x = F.concat(F.slice_like(upsample, route_now * 0, axes=(2, 3)), route_now, dim=1)
```
Replace the above code block with the following code block.
```
# upsample feature map reverse to shallow layers
upsample = _upsample(F, x, stride=2)
route_now = routes[::-1][i + 1]
if autograd.is_training():
    sliceLike = F.slice_like(upsample, route_now * 0, axes=(2, 3))
else:
    sliceLike = upsample
x = F.concat(sliceLike, route_now, dim = 1);
```
Modify the _upsample function in the corresponding file.
```text
def _upsample(F, x, stride=2):
    """Simple upsampling layer by stack pixel alongside horizontal and vertical directions.
    Parameters
    ----------
    x : mxnet.nd.NDArray or mxnet.symbol.Symbol
        The input array.
    stride : int, default is 2
        Upsampling stride
    """
    #return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)
    return F.UpSampling(x, scale = stride, sample_type="nearest")
```
Re-export.
#### Conversion and optimization
Please compile Ncnn according to the requirements of [Ncnn](https://github.com/Tencent/ncnn)  official website.
```
cd ncnn/build/tools/mxnet
./mxnet2ncnn your_model.json your_model.params model.param model.bin
```
Use ncnnoptimize to optimize the model.
```
cd ncnn/build/tools
./ncnnoptimize model.param model.bin model_opt.param model_opt.bin 65536
```
#### Modify the model_opt.param file
Add one to the two numbers enclosed by #.
We need to add an output layer manually, so the number of layers and blob in the second row must be increased by one.

```
7767517
#######
#59 63
########
60  64
```
Build an output layer at the end.
```
Yolov3DetectionOutput detection_out    3 1 yolov30_yolooutputv30_conv0_fwd yolov30_yolooutputv31_conv0_fwd yolov30_yolooutputv32_conv0_fwd detection_out 0=80 1=3 2=0.80000 3=0.450000 -23304=18,10.000000,13.000000,16.000000,30.000000,33.000000,23.000000,30.000000,61.000000,62.000000,45.000000,59.000000,119.000000,116.000000,90.000000,156.000000,198.000000,373.000000,326.000000 -23305=9,6.000000,7.000000,8.000000,3.000000,4.000000,5.000000,0.000000,1.000000,2.000000 -23306=3,32.000000,16.000000,8.000000 7=2
```
If you don’t understand, you can understand the meaning of the param file parameters in ncnn.

Please refer to the param file you got, set the input layer and output layer names of detection_out, here is the coco data set as an example, so it is 80 classification ，0=80.







