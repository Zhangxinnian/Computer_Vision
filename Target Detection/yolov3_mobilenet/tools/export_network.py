import gluoncv as gcv
from gluoncv.utils import export_block

net = gcv.model_zoo.get_model('yolo3_mobilenet1.0_coco',pretrained=True)
export_block('yolo3_mobilenet1.0_coco', net,preprocess=True, layout='HWC')
print('done!')