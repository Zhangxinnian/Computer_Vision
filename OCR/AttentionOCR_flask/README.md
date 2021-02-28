# OCR
Flask deployment version of AttentionOCR.
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites
Ubuntu18.04
CUDA
Python3.6
Flask
Tensorflow

Please install the correct version of cuda and python according to your own situation, cuda10.0 and python3.6.8 are used here.
### Installing
Tensorflow
opencv_python
Flask

Please refer to the version of CUDA to install the GPU version of Tensorflow. The operating environment required by the project has been exported to requirements.txt. Please use 
```
pip install -r requirements.txt
```
 to install.
## Running and Tests
 Explain how to run training scripts and test scripts.
 ### Running and Return the json string version
```
python ./ceshi.py
```
### Running and Server-side drawing version
```
python ./flaskapp.py
```







