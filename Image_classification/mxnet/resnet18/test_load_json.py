'''
Take ImageNet's pre-training model as an example, using json files and param files for prediction.
'''
import mxnet
from collections import namedtuple
import cv2
import numpy as np

#download model
def download_model():
    model_url = 'http://data.mxnet.io/models/imagenet/'
    mxnet.test_utils.download(model_url + 'resnet/18-layers/resnet-18-0000.params')
    mxnet.test_utils.download(model_url + 'resnet/18-layers/resnet-18-symbol.json')
    mxnet.test_utils.download(model_url + 'synset.txt')

#load label.txt
def get_label_names(label_path):
    with open(label_path, 'r') as rf:
        labels = [line_info.rstrip() for line_info in rf]
        return labels

#get model
def get_mod(model_str, ctx, data_shape):
    _vec = model_str.split(",")
    prefix = _vec[0]
    epoch = int(_vec[1])
    sym, arg_params, aux_params = mxnet.model.load_checkpoint(prefix, epoch)
    mod = mxnet.mod.Module(symbol=sym, context=ctx, label_names=None)
    #ImageNet--data_shape(224,224,3)
    mod.bind(for_training=False, data_shapes=[("data", data_shape)],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod

#Picture preprocessing
#opencv is used here, and the Image in test_params.py can also be used to load pictures
def preprocess_img(img_path, data_shape, ctx):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (data_shape[2],data_shape[3]))
    img = img[:,:,::-1]
    nd_img = mxnet.nd.array(img, ctx=ctx).transpose((2,0,1))
    #Convert the image format to NCHW
    nd_img = mxnet.nd.expand_dims(nd_img, axis=0)
    return nd_img

#Predict
def predict(model_str, ctx, data_shape, img_path, label_path):
    label_names = get_label_names(label_path)
    Batch = namedtuple("Batch",["data"])
    mod = get_mod(model_str, ctx, data_shape)
    nd_img = preprocess_img(img_path, data_shape, ctx)
    mod.forward(Batch([nd_img]))
    prob = mod.get_outputs()[0].asnumpy()
    #top5
    prob = np.squeeze(prob)
    sort_prob = np.argsort(prob)[::-1]
    for i in sort_prob[:5]:
        print("label name=%s, prob=%f"%(label_names[i],prob[i]))

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Test a model for image classification.')
    parser.add_argument('--model_str', type=str, default='resnet-18,0',
                        help='model name and epoch to use.')
    parser.add_argument('--data_shape', type=tuple, default=(1,3,224,224),
                        help='input data_shape.')
    parser.add_argument('--label_path', type=str, default='synset.txt',
                        help='the label path.')
    parser.add_argument('--img_path', type=str, default='image/bus.jpg',
                        help='the img path.')
    opt = parser.parse_args()
    return opt

ctx = mxnet.cpu()
opt = parse_args()
download_model()
predict(opt.model_str, ctx, opt.data_shape, opt.img_path, opt.label_path)

