from imutils.video import VideoStream
import time
import cv2
from face_processing import face_processing
face_pro=face_processing()
print("[INFO] starting video stream...")
video = VideoStream(src=0).start()     # có thể chọn cam bằng cách thay đổi src
writer = None
print("[INFO] Opening...")
time.sleep(2.)
model="lfw_model.pkl"
print("Mời bạn hãy mở khó bằng khuôn mặt")
while True:
    frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("[INFO] recognizing faces...")
    name=face_pro.predict_face(rgb,model)
    if name!="UNKNOWN":
        print("Chào mừng",name)
        break
    elif name=="UNKNOWN":
        print("Khuôn mặt không khớp!")
        break
    
    