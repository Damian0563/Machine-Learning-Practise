import numpy as np
import seaborn as sns #type: ignore
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

file=pd.read_csv('pavement.csv')
data=pd.DataFrame(file)
data.dropna(inplace=True)
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
label_encode(data,'Segment ID')

x=data.drop(columns=['Needs Maintenance'])
y=data['Needs Maintenance']
scaler=MinMaxScaler().fit(x)
x_scaled=scaler.transform(x)   

model=KNeighborsClassifier(algorithm='kd_tree')
params={
    "n_neighbors":[3,5,7],
    "weights": ['uniform', 'distance'],
    "p":[2]
}
grid=GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=3,
    n_jobs=-1,
    scoring='f1',
    verbose=2
)
x_train, x_test, y_train, y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=1)
optimized=grid.fit(x_train,y_train)
print(optimized.best_params_)