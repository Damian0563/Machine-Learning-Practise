from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread #type:ignore
import tensorflow as tf #type:ignore
from skimage.transform import resize #type:ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #type:ignore
import os

classifications=['normal','stone']
normal,stones=[],[]
for option in classifications:
    path=f"{os.getcwd()}/{option}"
    for file in os.listdir(path):
        file=os.path.join(path,file)
        image=imread(file)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # grayscale to RGB
        elif image.shape[2] == 4:
                image = image[:, :, :3]  # RGBA to RGB
        image=resize(image,(16,16),anti_aliasing=True)
        image=image.astype(np.float32)
        if image.shape == (16,16,3):
            if option=="normal":
                normal.append(image.flatten())
            else:
                stones.append(image.flatten())
x_train,x_test, y_train, y_test =[],[],[],[]
for train in normal[:4000]:
    x_train.append(train)
    y_train.append(0)
for test in normal[4000:]:
    x_test.append(test)
    y_test.append(0)
for train in stones[:4000]:
    x_train.append(train)
    y_train.append(1)
for test in stones[4000:]:
    x_test.append(test)
    y_test.append(1)

