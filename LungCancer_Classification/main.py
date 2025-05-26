import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


file=pd.read_csv('dataset_med.csv')
data=pd.DataFrame(file)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

print(data.head)
le_gender=LabelEncoder()
le_family=LabelEncoder()
le_country=LabelEncoder()
data['gender']=le_gender.fit_transform(data['gender'])
data['family_history']=le_family.fit_transform(data['family_history'])
data['country']=le_country.fit_transform(data['country'])
def one_hot_encode(df,column):
    oh=OneHotEncoder(sparse_output=True)
    encoded=oh.fit_transform(df[[column]]).toarray()
    encoded_df=pd.DataFrame(encoded,columns=oh.get_feature_names_out([column]),index=df.index)
    df=pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
    return df

data=one_hot_encode(data,'cancer_stage')
data=one_hot_encode(data,'treatment_type')
data=one_hot_encode(data,'smoking_status')
def standardize_date(val):
    return datetime.fromisoformat(val).timestamp()
data['diagnosis_date']=data['diagnosis_date'].apply(standardize_date)
data['end_treatment_date']=data['end_treatment_date'].apply(standardize_date)
data['treatment_length'] = data['end_treatment_date'] - data['diagnosis_date']
data=data.drop(columns=['diagnosis_date','end_treatment_date'])
print(data.info)

x=data.drop(columns=['survived'])
y=data['survived']
scaler=StandardScaler().fit(x)
x_scaled=scaler.transform(x)



x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,random_state=1,test_size=0.2)
model = XGBClassifier(scale_pos_weight=3, random_state=1) 
model.fit(x_test,y_test)
y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))