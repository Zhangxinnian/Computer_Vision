import matplotlib.pyplot as plt
import mxnet
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model

#im_fname = './image/sky.jpg'
im_fname = './image/mirror.jpeg'
img = image.imread(im_fname)
# plt.imshow(img.asnumpy())
# plt.show()

jitter_param = 0.4
lighting_param = 0.1
transform_fn = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = transform_fn(img)
plt.imshow(nd.transpose(img, (1,2,0)).asnumpy())
plt.show()

#num_gpus = 1
#context = [mxnet.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mxnet.cpu()]
net = get_model('resnet18_v2', pretrained=True)
#net.load_parameters('./params/0.7551-finetune-resnet18_v2-38-best.params', ctx=context)
net.load_parameters('./params/0.7551-finetune-resnet18_v2-38-best.params')
#net.collect_params().reset_ctx(context)


pred = net(img.expand_dims(axis=0))
class_names = ['brick','carpet','ceramic','fabric','foliage','food','glass','hair','leather','metal',
               'mirror','other','painted','paper','plastic','polishedstone','skin','sky','stone','tile',
               'wallpaper','water','wood']

ind = nd.argmax(pred, axis=1).astype('int')
print('The input picture is classified as [%s], with probability %.3f.'%
      (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))