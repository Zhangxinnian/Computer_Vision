# -*- coding: utf-8 -*-
# Created by 储朱涛
import cv2

camera = cv2.VideoCapture('rtsp://admin:a1234567@125.120.86.96:1055/Streaming/Channels/1')
while True:
    _,img = camera.read()
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(1)
