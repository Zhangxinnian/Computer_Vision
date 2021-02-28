import os
import argparse
import mxnet as mx
import gluoncv as gcv
import mxnet.ndarray as nd
import numpy as np
from gluoncv.data.transforms import presets
import matplotlib.pyplot as plt
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network',type=str, default='yolo3_mobilenet1.0_coco', help='Base network name')
    parser.add_argument('--model',type=str, default='',help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ctx = mx.cpu()
    args = parse_args()
    net = gcv.model_zoo.get_model(args.network, pretrained=True)
    if args.model:
        net.load_parameters(args.model)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx=ctx)
    net.hybridize()
    cvImg = cv2.imread('../biking.jpg')
    srcH, srcW = cvImg.shape[:2]
    netH, netW = 416, 416
    scale = 0
    npImg = np.zeros((416,416,3), dtype=np.float32)
    resizeWidth = 0
    resizeHeight = 0
    if srcH * 1.0 / srcW > 1:
        scale = netH * 1.0 / srcH
        resizeHeight = 416
        resizeWidth = int(srcW * scale)
    else:
        scale = netW * 1.0 / srcW
        resizeWidth = 416
        resizeHeight = int(srcH * scale)
    cvImgResize = cv2.resize(cvImg, (resizeWidth, resizeHeight))
    npImg[0:resizeHeight,0:resizeWidth,:]=cvImgResize
    npImgshow = npImg.copy()
    npImg[:,:,0] = (npImg[:,:,0] / 255.0 -0.485) / 0.229
    npImg[:,:,1] = (npImg[:,:,1] / 255.0 -0.456) / 0.224
    npImg[:,:,2] = (npImg[:,:,2] / 255.0 -0.406) / 0.225

    x = nd.array(npImg, dtype=np.float32).transpose((2, 0, 1)).expand_dims(0)
    x = x.as_in_context(ctx)
    ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
    net.export(args.model)
    ax = None
    ax = gcv.utils.viz.plot_bbox(npImg, bboxes, scores, ids, thresh=0.5, class_names=net.classes, ax=ax)
    plt.show()