import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
# from sklearn.svm import SVC

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

# init_model=SVC(max_iter=1000)     TOO SLOW!!!!!
# params=[
#     {'kernel': ['linear'], 'C': [0.1, 10, 100]},
#     {'kernel': ['rbf'], 'C': [0.1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001]},
# ]
model=LogisticRegressionCV(
    Cs=10,              
    cv=5,               
    scoring='f1_macro',       
    max_iter=1000,
    n_jobs=-1,
    verbose=2
)
model.fit(x_train,y_train)
print("Best regularization strength (C):", model.C_[0])
print("Classes:", model.classes_)
print("Coefficients shape:", model.coef_.shape)
# Best regularization strength (C): 0.3593813663804626
# Classes: [0 1 2]
# Coefficients shape: (3, 152992)
joblib.dump(model,'optimized')
model=joblib.load('optimized')

y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))

plt.figure(figsize=(8,8))
cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm, annot=True, cbar=True, fmt='d', cmap='Blues')
ax.set_xlabel('PREDICTED')
ax.set_ylabel('ACTUAL')
ax.set_xticklabels(['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
ax.set_yticklabels(['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
plt.show()


negative_influence=sorted(model.coef_[0])[::-1][:8] 
positive_influence=sorted(model.coef_[2])[::-1][:8]
feature_names = vectorizer.get_feature_names_out()
neg_indices = np.argsort(model.coef_[0])[::-1][:8]
pos_indices = np.argsort(model.coef_[2])[::-1][:8]
print("Top negative influence words:", feature_names[neg_indices])
print("Top positive influence words:", feature_names[pos_indices])
temp=pd.DataFrame()
temp['Words']=feature_names[neg_indices]
temp['Influence on negative classification']=[round(value,2) for value in negative_influence]
temp2=pd.DataFrame()
temp2['Words']=feature_names[pos_indices]
temp2['Influence on positive classification']=[round(value,2) for value in positive_influence]
plt.figure()
ax1=sns.barplot(temp,x='Words',y='Influence on negative classification',color='red')
ax1.set_title('Most influential negative words for classification')
plt.figure()
ax2=sns.barplot(temp2,x='Words',y='Influence on positive classification',color='green')
ax2.set_title('Most influential positive words for classification')
plt.tight_layout()
plt.show()



comments=["I enjoyed this show.","I do not know what to think about this book.",
        "I am deeply impressed by the depth of this course.","I found this film anemic.",
        "That was an abysmal performance.","I do not recommend this book.","I hate this guy.",
        "I could not agree more with this video."
        ]
for comment in comments:
    comment_vec = vectorizer.transform([comment])
    prediction = model.predict(comment_vec)
    result = "POSITIVE" if prediction[0] == 2 else ("NEUTRAL" if prediction[0] == 1 else "NEGATIVE")
    print(f"Text: {comment}, Sentiment: {result}.")
