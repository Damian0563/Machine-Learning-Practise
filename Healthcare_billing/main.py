import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from datetime import datetime

file=pd.read_csv('healthcare_dataset.csv')
data=pd.DataFrame(file)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.drop(columns=['Name','Doctor','Hospital','Room Number','Medication'])

def standardize(date):
    return datetime.fromisoformat(str(date)).timestamp()
data["Date of Admission"]=data['Date of Admission'].apply(standardize)
data["Discharge Date"]=data['Discharge Date'].apply(standardize)
data['Treatment Time']=(data['Discharge Date']-data['Date of Admission'])//(60*60*24)