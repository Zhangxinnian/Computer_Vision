# -*- coding: utf-8 -*-
# Created by 储朱涛
            
import cv2
# source = "rtsp://admin:a1234567@172.16.20.3/Streaming/Channels/1"
source = "rtsp://admin:a1234567@125.120.86.96:1055/Streaming/Channels/1"
camera = cv2.VideoCapture(source)
while True:
    success,img = camera.read()
    cv2.namedWindow('1',0)
    cv2.imshow('1',img)
    # print(cv2.waitKey(1))
    if cv2.waitKey(1) == 27:
        break