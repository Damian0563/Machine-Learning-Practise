import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import GridSearchCV
from sklearn.svm import SVC

file=pd.read_csv('sentiment_data.csv')
data=pd.DataFrame(file)
data.drop(columns=['Id'],inplace=True)
data=data.dropna()
data=data.drop_duplicates()
print(data.shape)

x=data['Comment']
y=data['Sentiment']
vectorizer=CountVectorizer()
new_x=vectorizer.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=0.15,random_state=1)

model=SVC()
params={
    "":[]
}
grid_search=GridSearchCV(

)



