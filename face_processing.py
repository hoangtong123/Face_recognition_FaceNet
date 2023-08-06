from sklearn.model_selection import train_test_split
from face_recognition import FaceRecognition
import os
import glob
import pandas as pd
import cv2
from tqdm import tqdm
from pprint import pprint
import cv2
class face_processing:
    global fr
    fr = FaceRecognition()
    
    """Hàm predict khuôn mặt video(kết quả trả về sẽ là một list)"""
    def predict_face(self,img,model):
        Model=model
        fr.load(Model)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = fr.predict(img, threshold=0.6)
        return result["predictions"][0]["person"]
    
    """Hàm train file dataset"""
    def train_dataset(self,dataset):
        ROOT_FOLDER=dataset
        MODEL_PATH = "lfw_model.pkl"
        dataset = []
        for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
            person= path.split("\\")[-2]
            dataset.append({"person":person, "path": path})
        dataset = pd.DataFrame(dataset)
        fr.fit_from_dataframe(dataset)
        fr.save(MODEL_PATH)
        
    """Hàm xử lý dataset"""
    def crawl_dataset(self,dataset):
        train, test = train_test_split(dataset, test_size=0.1, random_state=0)
        print("Train:",len(train))
        print("Test:",len(test))