# Target Detection 
Target Detection implemented using yolov5 network built by ncnn.
Refer to the article by the author of ncnn, [Nihui](https://github.com/nihui). 
If you are interested, please visit the original [article](https://zhuanlan.zhihu.com/p/275989233).
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
Ubuntu18.04
Opencv
Ncnn
Pytorch

Please install the correct version of ncnn and opencv according to your own situation, opencv3.2 are used here.
### Installing
Ncnn
Opencv
Pytorch

Please refer to ncnn official website and opencv official website for installation.
[Ncnn](https://github.com/Tencent/ncnn)
[Opencv](https://github.com/opencv/opencv)
```
pip install torch
```
Here, yolov5 is used as an example, please refer to the readme.md file of yolov5 to install the dependent libraries.

## Pytorch2onnx
### Pytorch Test
```
python detect.py --source inference/images --weights yolov5s.pt --conf 0.25
```
### 2onnx
```
python models/export.py --weights yolov5s.pt --img 640 --batch 1
pip install onnx-simplifier
python -m onnxsim yolov5s.onnx yolov5s-sim.onnx
```
### onnx2ncnn
```
cd ncnn/build/tools/onnx
./onnx2ncnn yolov5s-sim.onnx yolov5s.param yolov5s.bin
```
Converted to ncnn model, will output a lot of Unsupported slice step, this is the error report of the focus module conversion.
Quoting the picture from the original article.
Open yolov5s.param，Make the following changes.
-   ![avatar](https://github.com/Zhangxinnian/Computer_Vision/blob/main/Target%20Detection/yolov2ncnn/test.jpg)
Find the input and output blob names and connect them with a custom layer YoloV5Focus.
In the second line at the beginning of param, layer_count should be modified accordingly, but blob_count only needs to be greater than or equal to the actual number.
After modification, use the ncnnoptimize tool to automatically modify it to the actual blob_count.

### ncnnoptimize
```
cd ncnn/build/tools/
./ncnnoptimize yolov5s.param yolov5s.bin yolov5s-opt.param yolov5s-opt.bin 65536
```
Error message will appear：'layer YoloV5Focus not exists or registered'
Replace your custom layer YoloV5Focus with a known builtin layer, such as Exp.  
Run ncnnoptimize.
Change Exp back to YoloV5Focus in the optimized param.
### Modify the reshape layer
Finally, according to the meaning of the ncnn Reshape parameter, change the number of hard-coded output grids to -1 to be adaptive.










