from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.get_model("yolo3_mobilenet1.0_coco", pretrained=True)

im_fname = "./biking.jpg"
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print("shape of pre-processed image:", x.shape)
ids, score, bboxes = net(x)

ax = utils.viz.plot_bbox(img, bboxes[0], score[0], ids[0], class_names=net.classes)
plt.show()