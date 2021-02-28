import gluoncv as gcv
from gluoncv.data import VOCDetection

train_dataset = VOCDetection(splits=[(2007,'trainval'),(2012,'trainval')])
val_dataset = VOCDetection(splits=[(2007,'test')])
print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

train_image, train_label = train_dataset[1]
# bboxes = train_label[:,:4]
# cids = train_label[:,4:5]
print('image:', train_image.shape)
# print('bboxes:', bboxes.shape, 'class ids:', cids.shape)
#
from matplotlib import  pyplot as plt
from gluoncv.utils import  viz
# ax = viz.plot_bbox(train_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
# plt.show()

from gluoncv.data.transforms import  presets
from gluoncv import  utils
from mxnet import nd
width, height = 416 , 416
train_transform = presets.yolo.YOLO3DefaultTrainTransform(width, height)
val_transform = presets.yolo.YOLO3DefaultValTransform(width, height)
utils.random.seed(123)

train_image2, train_label2 = train_transform(train_image, train_label)
print('tensor shape:', train_image2.shape)

train_image2 = train_image2.transpose((1,2,0)) * nd.array((0.229,0.224,0.225)) + nd.array((0.485, 0.456, 0.406))
train_image2 = (train_image2 * 255).clip(0,255)
# ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:,:4], labels=train_label2[:,4:5],class_names=train_dataset.classes)
# plt.show()

from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader

batch_size = 2
num_workers = 0

batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True, batchify_fn=batchify_fn, last_batch='rollover',
                          num_workers=num_workers)
val_loader = DataLoader(val_dataset.transform(val_transform), batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep',
                        num_workers=num_workers)

for ib, batch in enumerate(train_loader):
    if ib > 3:
        break
    print('data 0:', batch[0][0].shape, 'label 0:', batch[1][0].shape)
    print('data 1:', batch[0][1].shape, 'label 1:', batch[1][1].shape)

from gluoncv import model_zoo
net = model_zoo.get_model('yolo3_mobilenet_v3_large_voc', pretrained_base=False)
print(net)
import mxnet as mx
x = mx.nd.zeros(shape=(1, 3, 416))
net.initialize()
cids, scores, bboxes = net(x)

loss = gcv.loss.YOLOV3Loss()
print(net._loss)

from mxnet import autograd
train_transform = presets.yolo.YOLO3DefaultTrainTransform(width, height, net)
batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
for ib, batch in enumerate(train_loader):
    if ib > 0:
        break
    print('data:', batch[0][0].shape)
    print('label:', batch[6][0].shape)
    with autograd.record():
        input_order = [0, 6, 1, 2, 3, 4, 5]
        obj_loss, center_loss, scale_loss, cls_loss = net(*[batch[o] for o in  input_order])

