import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier #type: ignore
from sklearn import tree
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,random_state=1,test_size=0.15)
model = XGBClassifier(scale_pos_weight=3, random_state=1,max_depth=10) 
model.fit(x_test,y_test)
probs = model.predict_proba(x_test)[:, 1]
y_pred = (probs >= 0.54).astype(int)

print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Died', 'Survived'])
plt.yticks(tick_marks, ['Died', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.tight_layout()
plt.show()



fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
avg_precision = average_precision_score(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()