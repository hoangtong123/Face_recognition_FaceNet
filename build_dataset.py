import cv2 
import os
import shutil
from face_processing import face_processing

face_pro=face_processing()
def dem_so_thu_muc(path):
    count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            count += 1
    return count

deleted_file=0
video = cv2.VideoCapture(0)
total = 0
id=input("Nhập tên người dùng:")
path="dataset/"+id+"/"

"""check đã quá giới hạn người nhập hay chưa"""
if dem_so_thu_muc("dataset")>4:
    print("Đã quá giới hạn người dùng")
    delete=input("Nhập tên người muốn xóa:")
    while not os.path.exists( "dataset/"+delete+"/"):
        print("thư mục không tồn tại")
        delete=input("Nhập lại tên người muốn xóa:")
    shutil.rmtree("dataset/"+delete)
    print("Đã xóa tên người dùng")
        
"""check đã nhập thông tin hay chưa"""
while  os.path.exists(path):
# if  os.path.exists(path):
    print("Đã nhập thông tin cho người này")
    id_2=input("Nhập tên người dùng:")
    path="dataset/"+id_2+"/"
os.makedirs(path)

print("Vui lòng chụp nhiều góc khác nhau ")
while True:
    ret, img = video.read()
    img = cv2.flip(img,1)
    cv2.imshow("video", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        p = path+str(total)+".jpg"    
        cv2.imwrite(p, img)
        total += 1
	# nhấn q để thoát
    if key == ord("q")  or total ==15:
	    break        
 
print("[INFO] {} face images stored".format(total))
video.release()
cv2.destroyAllWindows()
print("[INFO] Entering Data......")
face_pro.train_dataset("dataset")

