from face_recognition import FaceRecognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
import cv2
import base64
from tqdm import tqdm
import requests
from pprint import pprint

ROOT_FOLDER=r"F:\face_recognizer-master\face-recognition-master\face-recognition-master\dataset"
MODEL_PATH = "lfw_model.pkl"
dataset = []
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
    person= path.split("\\")[-2]
    dataset.append({"person":person, "path": path})
dataset = pd.DataFrame(dataset)
# train, test = train_test_split(dataset, test_size=0.1, random_state=0)
# print("Train:",len(train))
# print("Test:",len(test))
fr = FaceRecognition()
fr.fit_from_dataframe(dataset)
fr.save(MODEL_PATH)
