# Face Detection
Face Detection implemented using yolov5 network built by ncnn.
The code is consistent with [yolov5s_ncnn](https://github.com/Zhangxinnian/Computer_Vision/tree/main/Target%20Detection/yolov5s_ncnn).
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
Ubuntu18.04
Opencv
Ncnn

Please install the correct version of ncnn and opencv according to your own situation, opencv3.2 are used here.
### Installing
Ncnn
Opencv

Please refer to ncnn official website and opencv official website for installation.
[Ncnn](https://github.com/Tencent/ncnn)
[Opencv](https://github.com/opencv/opencv)

## CmakeLists
Here copy the include and lib under install under compiled ncnn to the current directory.

If you don't want to copy, you can change the find_ncnn part of cmakelist.

```
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND ")
endif()
```
## Run
Please change the path of the imagepath and the path of the model. If you want to use parameter passing, please see the comment section.











