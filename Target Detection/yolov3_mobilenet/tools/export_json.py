import gluoncv
from gluoncv.model_zoo import get_model
import mxnet as mx
model_name = 'yolo3_mobilenet_v3_large_voc'
net = get_model(model_name, pretrained=False, pretrained_base=False)
net.load_parameters('./yolo3_mobilenet_v3_large_voc_best.params')

net.hybridize()
img = mx.image.imread('./biking.jpg')
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img, resize_short=512)
output = net(transformed_img)

net.export('./json_yolo3_mobilenet_v3_large_voc', epoch=0)

