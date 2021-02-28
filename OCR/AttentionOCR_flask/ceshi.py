from  __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, os.path
import re
import sys
import tarfile
import copy
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


import textwrap
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from text_recognition import TextRecognition
from text_detection import TextDetection

######################初始化模型###############################################
def init_ocr_model():
    detection_pb = './checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    recognition_pb = './checkpoint/text_recognition_5435.pb' #
    with tf.device('/gpu:0'):
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True)
        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
        recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    label_dict = np.load('./reverse_label_dict_with_rects.npy',allow_pickle=True)[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict
ocr_detection_model, ocr_recognition_model, ocr_label_dict = init_ocr_model()

from flask import Flask,jsonify,flash,Response
from flask import make_response
from flask import request,render_template
from flask_bootstrap import Bootstrap
from flask import redirect,url_for
from flask import send_from_directory
from werkzeug import secure_filename
from subprocess import call
import time
import logging
import base64
app = Flask(__name__)
'''
class Request(object):
    def __init__(self):
        self.method = 'GET'
        self.path = ''
        self.query = {}
        self.body = ''
    def form(self):
        body = urllib.parse.unquote(self.body)
        args = body.split('&')
        f = {}
        for arg in args:
            k,v = arg.split('=')
            f[k] = v
        return f
'''
@app.route('/',methods=['GET','POST'])
def get_frame():
    #file_obj = request.files.get('file')
    #file_content = file_obj.read()
    #ip = request.remote_addr
    #logging.debug(ip)
   # print(ip)

    #print(file_content)
    #img = cv2.imdecode(np.frombuffer(file_content,np.uint8),cv2.IMREAD_COLOR)
    #cv2.imshow('demo',img)
    #rgb_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
   # print(rgb_img)
    #print(img)

    #bgr_image = cv2.imdecode(np.frombuffer(file_content,np.uint8),cv2.IMREAD_COLOR)
    # img_path ='./uploads'
    # for path,dirs,files in os.walk('./uploads'):
    # for path,dirs,files in os.walk(r'./image'):
    #   for file in files:
    #      img_path = os.path.join(path,file)
    # img_path ='./image/0.jpg'
    # for i in range(100):
    # i+=1
    # image = detection(img_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
    ALLOWED_EXTENSIONS = set([ 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'])
    img_name = str(request.form['image_name'])
    img_name = img_name.split('.')
    if img_name[-1] not in ALLOWED_EXTENSIONS:
        return jsonify('The file is not image!')
    else:
        img = base64.b64decode(str(request.form['image']))
        image_data = np.fromstring(img,np.uint8)
        bgr_image = cv2.imdecode(image_data,cv2.IMREAD_COLOR)
        result = detection(bgr_image,ocr_detection_model,ocr_recognition_model,ocr_label_dict)
    #print(result,'\n')
    #results = " , ".join(result)
        return jsonify(result)



from functools import reduce
import operator
import math
import cv2
from util import *
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from skimage import draw

def order_points(pts):
    def centroidpython(pts):
        x,y =zip(*pts)
        l = len(x)
        return sum(x)/l,sum(y)/l
    centroid_x,centroid_y = centroidpython(pts)
    pts_sorted = sorted(pts, key=lambda x: math.atan2((x[1] - centroid_y), (x[0] - centroid_x)))
    return pts_sorted
'''
def draw_annotation(image, points, label, horizon=True, vis_color=(30,255,255)):#(30,255,255)
    points = np.asarray(points)
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)

    image = Image.fromarray(image)
    width, height = image.size
    fond_size = int(max(height, width)*0.03)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    points = order_points(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT)
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)
    return image
'''
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox

#def detection(img_path, detection_model, recognition_model, label_dict):
def detection(bgr_image, detection_model, recognition_model, label_dict):
    #bgr_image = cv2.imread(img_path)
    print(bgr_image.shape)
    vis_image = copy.deepcopy(bgr_image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    r_boxes, polygons, scores = detection_model.predict(bgr_image)

    result2txt = []
    boxes2txt = []
    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)
        print(image_padded.shape)

        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        #print(''.join(results))
        #print(probs)
        ###################写入文本测试###############
        result2txt.append(str(''.join(results)))
        #boxes2txt.append(r_boxes)
        #print(results2txt)
       # with open('result.txt','a') as file_handle:
            #file_handle.write(results2txt)
            #file_handle.write('\n')
        ############################################
       # ccw_polygon = orient(Polygon(polygon.tolist()).simplify(5, preserve_topology=True), sign=1.0)
        #pts = list(ccw_polygon.exterior.coords)[:-1]
        #vis_image = draw_annotation(vis_image, pts, ''.join(results))
        # if height >= width:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results), False)
        # else:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results))

    #return vis_image
    return result2txt#,boxes2txt
#import time
#start = time.time()
#file_content = get_frame()
#bgr_image = cv2.imdecode(np.frombuffer(file_content,np.uint8),cv2.IMREAD_COLOR)
#img_path ='./uploads'
#for path,dirs,files in os.walk('./uploads'):
#for path,dirs,files in os.walk(r'./image'):
 #   for file in files:
  #      img_path = os.path.join(path,file)
#img_path ='./image/0.jpg'
#for i in range(100):
#image = detection(img_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
#result = detection(bgr_image,ocr_detection_model,ocr_recognition_model,ocr_label_dict)
#print(result,'\n')
    #i+=1
#end = time.time()
#final = end - start
#print(final)
if __name__ =='__main__':
    app.run(host='0.0.0.0',port=8080,threaded=True,debug=True)

































