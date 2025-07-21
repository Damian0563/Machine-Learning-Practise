from sklearn.model_selection import train_test_split
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
        image=resize(image,(24,24),anti_aliasing=True)
        print(image.ndim)
        os._exit(1)
        if option=='normal': normal.append(image)
        else: stones.append(image)
