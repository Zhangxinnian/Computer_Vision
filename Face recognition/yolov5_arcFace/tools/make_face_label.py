######## make labels #############3
from PIL import  Image, ImageDraw
anno_box_path = r'/media/cj1/data/CelebA/Anno-20210129T055314Z-001/Anno/list_bbox_celeba.txt'
train_label_dir = '/media/cj1/data/CelebA/labels/train'
valid_label_dir = '/media/cj1/data/CelebA/labels/valid'
test_label_dir = '/media/cj1/data/CelebA/labels/test'
img_dir = '/media/cj1/data/CelebA/Img/img_celeba.7z/img_celeba'
count = 0
epoch = 1
box_file = open(anno_box_path,"r")

def save_label_txt(line,label_dir):
    img_strs = line.split()
    x1, y1, w, h = int(img_strs[1]), int(img_strs[2]), int(img_strs[3]), int(img_strs[4])
    x2, y2 = x1 + w, y1 + h

    img = Image.open(f"{img_dir}/{img_strs[0]}")
    img_w, img_h = img.size

    dw = 1. / (int(img_w))
    dh = 1. / (int(img_h))
    x = ((x1 + x2) / 2.0 - 1) * dw
    y = ((y1 + y2) / 2.0 - 1) * dh
    w = (x2 - x1) * dw
    h = (y2 - y1) * dh
    label_txt = open(f"{label_dir}/{imgname}.txt", "w")
    label_txt.write(f"0 {x} {y} {w} {h}\n")
    label_txt.flush()
    label_txt.close()

NUM_EXAMPLES = 202599
TRAIN_STOP = 162770
VALID_STOP = 182637

i = 0

for line in box_file:
    if i < 2:
        i += 1
        continue
    i += 1
    print(line)

    imgname = line[0:6]
    print(imgname)
    #print(int(imgname))
    if int(line[0:6]) <= TRAIN_STOP:
        save_label_txt(line, train_label_dir)

    if int(line[0:6]) > TRAIN_STOP and int(line[0:6]) <= VALID_STOP:
        save_label_txt(line, valid_label_dir)

    if int(line[0:6]) > VALID_STOP and int(line[0:6]) <= NUM_EXAMPLES:
        save_label_txt(line, test_label_dir)

    if i == 202600:
        exit()

