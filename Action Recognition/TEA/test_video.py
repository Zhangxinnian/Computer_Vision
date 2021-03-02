import cv2

camrea = cv2.VideoCapture(0)
while True:
    _,img = camrea.read()
    cv2.imshow("img",img)
    cv2.waitKey(1)