import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type:ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree

file=pd.read_csv('social_media_vs_productivity.csv')
data=pd.DataFrame(file)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
le = LabelEncoder()
data['uses_focus_apps'] = le.fit_transform(data['uses_focus_apps'])
data['has_digital_wellbeing_enabled'] = le.fit_transform(data['has_digital_wellbeing_enabled'])
def one_hot_encode_column(df, column_name):
    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = oh.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded,
                              columns=oh.get_feature_names_out([column_name]),
                              index=df.index)
    df = pd.concat([df.drop(column_name, axis=1), encoded_df], axis=1)
    return df

data = one_hot_encode_column(data, 'gender')
data = one_hot_encode_column(data, 'job_type')
data = one_hot_encode_column(data, 'social_platform_preference')
print(data.info())
print(data.head())


x=data.drop(columns=['job_satisfaction_score'])
y=data['job_satisfaction_score']
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
print(data.shape)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=17050,random_state=1)
model=DecisionTreeRegressor(max_depth=10)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
plt.figure(figsize=(12, 8))
tree.plot_tree(model)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(x=y_pred,y=y_test)
plt.xlabel('Predicted job satisfaction')
plt.ylabel('Actual job satisfaction score')
plt.show()

plt.figure(figsize=(10,8))
plt.subplots_adjust(bottom=0.4)
cols = x.columns
frame = pd.DataFrame()
frame['name'] = cols
frame['importances'] = model.feature_importances_
sns.set_theme(font_scale=1)
sns.barplot(frame, x='name', y='importances')
plt.xticks(rotation=90)
plt.show()


print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))