import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from sklearn import tree
import matplotlib.pyplot as plt


file=pd.read_csv('dataset_med.csv')
data=pd.DataFrame(file)

print(data.head)
le=LabelEncoder()
data['gender']=le.fit_transform(data['gender'])