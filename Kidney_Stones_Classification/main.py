from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #type:ignore
from skimage.io import imread #type:ignore
import tensorflow as tf #type:ignore
from skimage.transform import resize #type:ignore
import joblib #type:ignore
import numpy as np #type:ignore
import pandas as pd #type:ignore
import matplotlib.pyplot as plt #type:ignore
import seaborn as sns #type:ignore
import os

classifications=['normal','stone']
stone_itr,normal_itr =0,0
x_train,x_test, y_train, y_test =[],[],[],[]
print("Starting preprocessing data.")
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
                if normal_itr<=4000:
                    x_train.append(image.flatten())
                    y_train.append(0)
                else:
                    x_test.append(image.flatten)
                    y_test.append(0)
                normal_itr+=1
            else:
                if stone_itr<=4000:
                    x_train.append(image.flatten())
                    y_train.append(1)
                else:
                    x_test.append(image.flatten())
                    y_test.append(1)
                stone_itr+=1
print("Concluded preprocessing data images and training-testing splits.")
model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation="sigmoid",input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation="sigmoid")
]
)
loss_function=tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer="adam",loss=loss_function,metrics=['accuracy'])
print("Training model...")
model.fit(x_train,y_train, epochs=5)
evaluation=model.evaluate(x_test,y_test,verbose=2)
loss,accuracy=evaluation[0],evaluation[1]
print(f"Model loss: {loss}, model accuracy: {accuracy}")
joblib.dump(model,"trained")
model=joblib.load("trained")
