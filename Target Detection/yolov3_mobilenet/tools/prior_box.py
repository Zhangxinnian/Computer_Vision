# import  numpy as np
# import mxnet as mx
# import mxnet.gluon as gluon
# import cv2
# import argparse
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Test with YOLO networks.')
#     parser.add_argument('--model',type=str, default='yolo3_mobilenet1.0_coco', help='Load weights from previously saved parameters.')
#     parser.add_argument('--epoch', type=int, default=0)
#     args = parser.parse_args()
#
#     return args
#
# if __name__ == '__main__':
#     args = parse_args()
#     ctx = mx.cpu()
#     #load model symbol and params
#     path = '/home/cj1/VOCdevkit/mxnet2ncnn/'
#     model = path+args.model
#
#     sym, arg_params, aux_params = mx.model.load_checkpoint(model, args.epoch)
#     needDeletedKeys = []
#     for key in arg_params.keys():
#         if key.find('offset') != -1 or key.find('anchor') != -1:
#             needDeletedKeys.append(key)
#     for key in  needDeletedKeys:
#         del arg_params[key]
#
#     internals = sym.get_internals()
#     layerNames = sym.get_internals().list_outputs() #获得所有中间输出
#     print(layerNames)
#     outputName1 = internals['yolov30_yolooutputv30_conv0_fwd_output']
#     outputName2 = internals['yolov30_yolooutputv31_conv0_fwd_output']
#     outputName3 = internals['yolov30_yolooutputv32_conv0_fwd_output']
#
#     outputSymbol = mx.symbol.Group([outputName3, outputName2, outputName1])
#     mod = mx.mod.Module(symbol=outputSymbol, context=ctx, label_names=None)
#     mod.bind(for_training=False, data_shapes=[('data',(1,3,416,416))],label_shapes=mod._label_shapes)
#     mod.set_params(arg_params,aux_params, allow_missing=False)
#     mx.model.save_checkpoint('truncate_' + args.model, 0, outputSymbol, arg_params, aux_params)

import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--model', type=str, default='yolo3_mobilenet1.0_coco',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    ctx = mx.cpu()
    # load model symbol and params
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model, args.epoch)
    needDeletedKeys = []
    for key in arg_params.keys():
        if key.find("offset") != -1 or key.find("anchor") != -1:
            needDeletedKeys.append(key)
    for key in needDeletedKeys:
        del arg_params[key]

    internals = sym.get_internals()
    layerNames = sym.get_internals().list_outputs() #获得所有中间输出
    print(layerNames)
    outputName1 = internals["yolov30_yolooutputv30_conv0_fwd_output"]
    outputName2 = internals["yolov30_yolooutputv31_conv0_fwd_output"]
    outputName3 = internals["yolov30_yolooutputv32_conv0_fwd_output"]

    outputSymbol = mx.symbol.Group([outputName3, outputName2, outputName1])
    mod = mx.mod.Module(symbol=outputSymbol, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 416, 416))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=False)
    mx.model.save_checkpoint("truncate_" + args.model, 0, outputSymbol, arg_params, aux_params)