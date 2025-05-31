import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt

file=pd.read_csv("personality_dataset.csv")
data=pd.DataFrame(file)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

le_enc=LabelEncoder()
le_n=LabelEncoder()
data['Drained_after_socializing']=le_enc.fit_transform(data['Drained_after_socializing'])
data['Stage_fear']=le_enc.fit_transform(data['Stage_fear'])
def standardize_target(val):
    return 1 if val=="Introvert" else 0
data['Personality']=data['Personality'].apply(standardize_target)
x=data.drop(columns=['Personality'])
y=data["Personality"]
scaler=StandardScaler().fit(x)
x_scaled=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=1)

model=AdaBoostClassifier(estimator=DecisionTreeClassifier())
parameters={
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 150, 200],
    "estimator__max_depth": [1,2,3,4,5,6]
}
grid_search=GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Extrovert', 'Introvert'])
plt.yticks(tick_marks, ['Extrovert', 'Introvert'])
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