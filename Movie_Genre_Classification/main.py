import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns #type: ignore
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

file=pd.read_csv('movies.csv')
data=pd.DataFrame(file)
data=data.dropna()
print(data.isnull().sum())
data=data.drop_duplicates()

plt.figure(figsize=(10, 6))
sns.countplot(x='Genre', data=data,hue='Genre', legend=False,)
plt.title('Distribution of Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

for category in data['Content_Rating'].unique():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Genre', data=data[data['Content_Rating'] == category],hue='Genre', legend=False)
    plt.title(f'Distribution of Content Ratings for {category} content rating')
    plt.xlabel('Content Rating')
    plt.ylabel('Count')
    plt.show()

plt.figure(figsize=(10, 6))
plt.title('Average Rating by Genre')
sns.pointplot(x='Genre', y='Rating', data=data, estimator=np.mean, errorbar=None)
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.show()

def standardize_genre(genre):
    if genre=="Action":
        return 0
    elif genre=="Comedy":
        return 1
    elif genre=="Drama":
        return 2
    elif genre=="Fantasy":
        return 3
    elif genre=="Horror":
        return 4
    elif genre=="Romance":
        return 5
    elif genre=="Thriller":
        return 6
    return 7
def vectorize_text(train_df, test_df, column):
    vectorizer = CountVectorizer()
    train_vec = vectorizer.fit_transform(train_df[column])
    test_vec = vectorizer.transform(test_df[column])
    train_df_vec = pd.DataFrame(train_vec.toarray(), columns=vectorizer.get_feature_names_out([column]), index=train_df.index)
    test_df_vec = pd.DataFrame(test_vec.toarray(), columns=vectorizer.get_feature_names_out([column]), index=test_df.index)
    train_df = pd.concat([train_df.drop(columns=[column]), train_df_vec], axis=1)
    test_df = pd.concat([test_df.drop(columns=[column]), test_df_vec], axis=1)
    return train_df, test_df

def one_hot_encode(df_train,df_test,column):
    encoder=OneHotEncoder(sparse_output=False)
    df_transformed_train=encoder.fit_transform(df_train[[column]])
    df_transformed_test=encoder.transform(df_test[[column]])
    df_train_out=pd.DataFrame(df_transformed_train, columns=encoder.get_feature_names_out([column]),index=df_train.index)
    df_test_out=pd.DataFrame(df_transformed_test, columns=encoder.get_feature_names_out([column]),index=df_test.index)
    train_final=pd.concat([df_train.drop(columns=[column]), df_train_out], axis=1)
    test_final=pd.concat([df_test.drop(columns=[column]), df_test_out], axis=1)
    return train_final, test_final

def scale(df_train,df_test,column):
    scaler=MinMaxScaler()
    df_train[[column]] = scaler.fit_transform(df_train[[column]])
    df_test[[column]] = scaler.transform(df_test[[column]])
    return df_train, df_test


data=data.drop(columns=['Votes','Director','Language','Country','BoxOffice_USD','Budget_USD','Production_Company','Lead_Actor','Num_Awards','Critic_Reviews'])
x=data.drop(columns=['Genre'])
y=data['Genre'].apply(standardize_genre)

print(x.columns)
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_test = scale(x_train,x_test, 'Rating')
x_train, x_test = scale(x_train,x_test, 'Year')
x_train, x_test = scale(x_train,x_test, 'Duration')
x_train, x_test = vectorize_text(x_train,x_test, 'Description') 
x_train, x_test = vectorize_text(x_train,x_test, 'Title')
x_train, x_test = one_hot_encode(x_train,x_test, 'Content_Rating')
model=SVC()
params={
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}
f1_weighted = make_scorer(f1_score, average='weighted')
grid=GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring=f1_weighted,
    cv=5,
    verbose=2,
    n_jobs=-1
)
mod=grid.fit(x_train, y_train)
print("Best parameters found: ", mod.best_params_) # ==>>  Best parameters found:  {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
joblib.dump(mod, 'optimized_model')
model=joblib.load('optimized_model')
y_pred=model.predict(x_test)
print(classification_report(y_test,y_pred))

plt.figure(figsize=(10,6))
ax=sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cbar=True, fmt='d', cmap='Blues')
ax.set_xlabel('PREDICTED')
ax.set_xticklabels(['Action','Comedy','Drama','Fantasy','Horror','Romance','Thriller'])
ax.set_yticklabels(['Action','Comedy','Drama','Fantasy','Horror','Romance','Thriller'])
ax.set_ylabel('ACTUAL')
plt.show()

