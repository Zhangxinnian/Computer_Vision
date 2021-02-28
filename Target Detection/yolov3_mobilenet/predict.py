import os
import argparse
import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.transforms import presets
from  matplotlib import  pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_mobilenet1.0_voc', help='Base network name')
    parser.add_argument('--images', type=str, default='',help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='', help='Training with GPUS, you can specify 1,3 for example')
    parser.add_argument('--pretrained', type=str, default='True', help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    if not args.images.strip():
        image_list = ['./biking.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip]

    # if args.pretrained.lower() in ['true', '1', 'yes', 't']:
    #     net = gcv.model_zoo.get_model(args.network,pretrained=True)
    # else:
    net = gcv.model_zoo.get_model(args.network, pretrained=True, pretrained_base=False)
    #net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
        #net.load_parameters(args.pretrained)
    #net.load_parameters('./yolo3.params')

    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx=ctx)

    for image in image_list:
        ax = None
        x, img = presets.yolo.load_test(image, short=144)
        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]

        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh, class_names=net.classes, ax=ax)
        plt.show()
