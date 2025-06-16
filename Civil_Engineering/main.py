import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
import joblib
from xgboost import XGBClassifier #type:ignore
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

file=pd.read_csv('cv_engine.csv')
data=pd.DataFrame(file)
print(data.columns)
print(data.info())
data=data.drop_duplicates()
data=data.dropna()
data=data.drop(columns=['Project ID'])

x=data.drop(columns=['is_good'])
y=data['is_good']


plt.figure(figsize=(20,10))
sns.set_theme(font_scale=0.65)
ax=sns.heatmap(data.corr(),annot=True,fmt=".1f")
ax.xaxis.tick_top()
plt.show()

x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=1,test_size=0.2)
scaler=StandardScaler().fit(x_train)
scaled_x=scaler.transform(x_train)

scaler=StandardScaler().fit(x_test)
scaled_test=scaler.transform(x_test)

mod=XGBClassifier()
params={
    "n_estimators":[100,150,200],
    "learning_rate":[0.01,0.05,0.1],
    "max_depth":[3,6,9],
    "scale_pos_weight":[1,3,5],
    "subsample":[0.5,1],
}
grid = GridSearchCV(
    estimator=mod,
    param_grid=params,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    cv=3
)
# grid.fit(scaled_x,y_train)
# print(grid.best_params_)
#{'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 100, 'scale_pos_weight': 1, 'subsample': 0.5}
# joblib.dump(grid,'optimized')
model=joblib.load('optimized')
y_pred=model.predict(scaled_test)
print(classification_report(y_pred,y_test))

plt.figure(figsize=(8,6))
sns.set_theme(font_scale=1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()