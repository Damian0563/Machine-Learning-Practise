import numpy as np
import pandas as pd
from skimage.io import imread #type:ignore
from skimage.transform import resize #type:ignore
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns #type:ignore
import joblib
import os

def label_encode(df_train,df_test):
    encoder=LabelEncoder()
    df_train=encoder.fit_transform(df_train)
    df_test=encoder.transform(df_test)
    return df_train, df_test

BASE_DIR='/Users/damia/Desktop/Machine Learning/Visual_Tumor_Classification'
categories=['Training','Testing']
labels=['glioma','meningioma','notumor','pituitary']
print('Preprocessing images and splitting...')
x_train, x_test, y_train, y_test=[],[],[],[]
for category in categories:
    for label in labels:
        path=f"{BASE_DIR}/{category}/{label}"
        for file in os.listdir(path):
            img_path=os.path.join(path,file)
            image=imread(img_path)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)  # grayscale to RGB
            elif image.shape[2] == 4:
                image = image[:, :, :3]  # RGBA to RGB
            image=resize(image,(16,16),anti_aliasing=True)
            image=image.astype(np.float32)
            if image.shape == (16,16,3):
                if category=="Training":
                    x_train.append(image.flatten())
                    y_train.append(label)
                else:
                    x_test.append(image.flatten())
                    y_test.append(label)
print('Split done...')
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train,y_test=label_encode(y_train,y_test)
print('Done encoding...')
weighted=make_scorer(f1_score,average='weighted')
est=SVC()
params={
    'C':[0.1,1,10],
    'kernel':['rbf','linear'],
    'gamma':['auto','scale']
}
grid=GridSearchCV(
    estimator=est,
    param_grid=params,
    scoring=weighted,
    cv=5,
    n_jobs=-1,
    verbose=2
)
print('Optimising for the most efficient model parameters...')
model=grid.fit(x_train,y_train)
print(grid.best_params_) #==>> {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
joblib.dump(model,'optimized')
model=joblib.load('optimized')

y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))

plt.figure(figsize=(10,6))
plt.title('Confussion matrix')
ax=sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cbar=True, fmt='d',cmap='Blues')
ax.set_xticklabels(['glioma','meningioma','No Tumor','Pituitary'])
ax.set_yticklabels(['glioma','meningioma','No Tumor','Pituitary'])
plt.show()
