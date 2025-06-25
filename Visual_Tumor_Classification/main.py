import numpy as np
import pandas as pd
from skimage.io import imread #type:ignore
from skimage.transform import resize #type:ignore
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns #type:ignore
import os

def label_encode(df_train,df_test):
    encoder=LabelEncoder()
    df_train=encoder.fit_transform(df_train)
    df_test=encoder.transform(df_test)
    return df_train, df_test

BASE_DIR='/Users/damia/Desktop/Machine Learning/LinearPerformance/Visual_Tumor_Classification'
categories=['Training','Testing']
labels=['glioma','meningioma','notumor','pituitary']

x_train, x_test, y_train, y_test=[],[],[],[]
for category in categories:
    for label in labels:
        path=f"{BASE_DIR}/{category}/{label}"
        for file in os.listdir(path):
            img_path=os.path.join(path,file)
            image=imread(img_path)
            image=resize(image,(16,16))
            if category=="Training":
                x_train.append(image.flatten())
                y_train.append(label)
            else:
                x_test.append(image.flatten())
                y_test.append(label)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train,y_test=label_encode(y_train,y_test)