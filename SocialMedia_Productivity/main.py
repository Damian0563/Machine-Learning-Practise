import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report

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
