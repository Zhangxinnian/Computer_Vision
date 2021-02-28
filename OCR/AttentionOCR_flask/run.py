from rundemo import python_model
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

model = python_model(model_path='./ICDAR_0.7.pb')
# 读取数据
#f = np.load('../model/data/mnist.npz')
src = cv2.imread("timg.jpg")
#x_test, y_test = f['x_test'], f['y_test']
#x_test = np.reshape(x_test, [-1, 784])
output = model.inference(src)
#print(output.astype(np.int32))
#print(output)

new_im = MatrixToImage(output)
plt.imshow(output, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
new_im.save('demo.bmp')




#output=model.inference(x_test)
#print(output.astype(np.int32))