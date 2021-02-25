import gluoncv
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from mxnet import image
#model_name = 'resnet18_v2'
#net = get_model(model_name,  pretrained=True)
model_name = 'cifar_resnet20_v1'
net = get_model(model_name, classes=10, pretrained=True)
# if you modify the network and save the parameters locally, you need to use the modified model program to load the net
#net.load_parameters('../params/0.7551-finetune-resnet18_v2-38-best.params')
#if you use HybridSequential when defining the network, you need to use hybridize to activate to export it into a json format model file.
net.hybridize()

#im_fname = '../image/mirror.jpeg'
im_fname = '../image/cat.jpeg'
img = image.imread(im_fname)
#transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img, resize_short=224)
transformed_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
])
transformed_img = transformed_fn(img)
#output = net(transformed_img)
output = net(transformed_img.expand_dims(axis=0))

net.export('../model_json/cifar_resnet',epoch=0)


