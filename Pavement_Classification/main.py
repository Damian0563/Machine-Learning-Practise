import numpy as np
import seaborn as sns #type: ignore
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier #type:ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

file=pd.read_csv('pavement.csv')
data=pd.DataFrame(file)
data.dropna(inplace=True)
data=data.drop(columns=['Segment ID',"Rutting"])
data.drop_duplicates(inplace=True)


def one_hot_encode(df,column):
    oh=OneHotEncoder(sparse_output=True)
    encoded=oh.fit_transform(df[[column]]).toarray()
    encoded_df=pd.DataFrame(encoded,columns=oh.get_feature_names_out([column]),index=df.index)
    df = pd.concat([df.drop(column,axis=1),encoded_df], axis=1)
    return df
data=one_hot_encode(data,'Asphalt Type')
def standardize(val):
    if val=="Primary": return 3
    elif val=="Secondary": return 2
    elif val=="Tertiary": return 1
data['Road Type']=data['Road Type'].apply(standardize)
def label_encode(data,column):
    le=LabelEncoder()
    data[column]=le.fit_transform(data[column])

x=data.drop(columns=['Needs Maintenance'])
current_year = datetime.datetime.now().year
x['Years Since Maintenance'] = current_year - x['Last Maintenance']
x['AADT']=x['AADT']/max(x['AADT'])
x.drop('Last Maintenance', axis=1, inplace=True)
y=data['Needs Maintenance']

plt.figure(figsize=(15,6))
sns.set_theme(font_scale=0.65)
ax=sns.heatmap(data.corr(),annot=True,fmt=".1f")
ax.xaxis.tick_top()
plt.show()

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
scaler = MinMaxScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)  

model=XGBClassifier()
params = {
    "n_estimators": [100, 150,200],
    "scale_pos_weight":[1,3,5],
    "max_depth":[3,5,7,9],
    "learning_rate":[0.01,0.05,0.1]
}
grid=GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=3,
    n_jobs=-1,
    scoring='f1',
    verbose=2
)
print(x.head())
# optimized=grid.fit(x_train_scaled,y_train)
# print(optimized.best_params_)
# joblib.dump(optimized.best_estimator_,'optimized')
#{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'scale_pos_weight': 1}
model=joblib.load('optimized')
print(model.feature_importances_)
feature_names = [
    'PCI',                     # 0
    'Road Type',               # 1
    'AADT',                    # 2
    'Average Rainfall',        # 3
    'IRI',                     # 4
    'Asphalt Type_Asphalt',    # 5
    'Asphalt Type_Concrete',   # 6
    'Years Since Maintenance'  # 7
]
plt.figure(figsize=(10,6))
temp=pd.DataFrame()
temp['Feature_names']=feature_names
temp['Impact_on_final_prediction']=model.feature_importances_
sns.barplot(temp, x="Feature_names", y="Impact_on_final_prediction")
plt.show()

y_pred=model.predict(x_test_scaled)
y_proba=model.predict_proba(x_test)[:, 1]
print(classification_report(y_test,y_pred))

plt.figure(figsize=(8,6))
sns.set_theme(font_scale=1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

results = x_test.copy()
results['True Label'] = y_test.values
results['Predicted'] = y_pred
results['Value']=y_proba
# False Positives
false_positives = results[(results['True Label'] == 0) & (results['Predicted'] == 1)]
print("False Positives:")
print(false_positives)
# False Negatives
false_negatives = results[(results['True Label'] == 1) & (results['Predicted'] == 0)]
print("False Negatives:")
print(false_negatives)

