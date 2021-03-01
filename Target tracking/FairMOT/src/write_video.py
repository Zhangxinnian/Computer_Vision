import cv2
cap = cv2.VideoCapture(0)
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
vout = cv2.VideoWriter()
vout.open('./video/output.mp4',fourcc,sz,True)
cnt = 0
while cnt<20:
    cnt += 1
    print(cnt)
    _,frame = cap.read()
    cv2.putText(frame, str(cnt), (10,10), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1,cv2.LINE_AA)
    vout.write(frame)
vout.release()
cap.release()