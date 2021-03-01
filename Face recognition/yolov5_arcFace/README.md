# Face Recognition
Use yolov5_arcface for face recognition, and add silent live detection.
Please refer to yolov5 in Target Detection for installation, and run yolov5 successfully.
The project implementation draws on an [article](https://blog.csdn.net/weixin_41809530/article/details/107313752) , interested can visit the original article, and [code](https://github.com/ooooxianyu/yoloV5-arcface_forlearn).

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
Ubuntu18.04
CUDA
Python3.6
Pytorch
Ncnn

Please install the correct version of cuda and python according to your own situation, cuda10.0 and python3.6.8 are used here.
### Installing
Ncnn
Pytorch
opencv_python
onnx

[Ncnn](https://github.com/Tencent/ncnn)
Please refer to the version of CUDA to install the GPU version of Pytorch. The operating environment required by the project has been exported to requirements.txt. Please use 
```
pip install -r requirements.txt
```
 to install.
## Running and Tests
 Explain how to run training scripts and test scripts.
### Making train and val dataset
 Here is an example using celebA data set.
```
python ./tools/split_dataset.py
```
### Making label
Create new directories for labels and photos under one directory, named "/labels" and "/images" respectively, which are used to store image data and label data.
```
python ./tools/make_face_label.py
```
### Making face.yaml
```
cd ./data
touch face.yaml
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: /media/cj1/data/CelebA/images/train/
val: /media/cj1/data/CelebA/images/test/
# number of classes
nc: 1
# class names
names: ['face']
```
### Training face model
```
python ./train.py --img 640 --batch 16 --epochs 10 --data ./data/face.yaml --weights ./weights/yolov5s.pt
```
### Test 
The test is divided into two parts, one is to test face detection, and the other is to test face recognition.
The face identified here is Sun Zhenni.

#### Test face detection
```
python ./detect.py --source ./data/images/bus.jpg --weights ./runs/train/exp/weights/best.pt
```
#### Test face recognition
The face feature library is in the form of pictures and stored in a folder(image_feature).
```
python ./detect_face.py --source ./data/images/sun.jpg --weights ./weights/best.pt
```
### SilentFace  Detection
Please use '--open_rf' to control whether to turn on silent detection.
```
python ./detect_face.py --source ./data/images/sun.jpg --weights ./weights/best.pt
--open_rf 1
```
## 2Ncnn
For the same process as yolov5, please visit[yolov2ncnn](https://github.com/Zhangxinnian/Computer_Vision/tree/main/Target%20Detection/yolov2ncnn).
This model is a face detection model.
The recognition version of the Ncnn framework has not yet been done, if you are interested, please try to do it yourself.
