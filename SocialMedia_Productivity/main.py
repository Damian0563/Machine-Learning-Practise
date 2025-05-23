import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report

file=pd.read_csv('social_media_vs_productivity.csv')
data=pd.DataFrame(file)

print(data.info())
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

oh=OneHotEncoder()
gender_enc=oh.fit_transform(data[['gender']]).toarray()
temp_gender=pd.DataFrame(gender_enc,columns=oh.get_feature_names_out(['gender']))
data=pd.concat([data.drop('gender', axis=1), temp_gender], axis=1)



