import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

file=pd.read_csv("Gold-Silver-GeopoliticalRisk_HistoricalData.csv")
data=pd.DataFrame(file)
print(data.columns)
data.drop(columns=['SILVER_PRICE','SILVER_OPEN','SILVER_HIGH','SILVER_LOW','SILVER_CHANGE_%','EVENT'],inplace=True)
print(data.head())
data.dropna(inplace=True)
data=data.sort_values(by='DATE').reset_index(drop=True)

print(data[['DATE', 'GOLD_PRICE']].head(20))
print(data[['DATE', 'GOLD_PRICE']].tail(20))
print(data.dtypes)
print(data.shape)
data['DATE']=pd.to_datetime(data['DATE'])
data['GOLD_PRICE']=data['GOLD_PRICE'].astype(float)
preview_x=data['DATE'][::30][::-1]
preview_y=data['GOLD_PRICE'][::30][::-1]
plt.figure(figsize=(12,8))
plt.plot(preview_x,preview_y)
plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
plt.xticks(rotation=90)
plt.xlabel("Date")
plt.ylabel("Gold Price in USD")
plt.title("Gold Price over Time")
plt.show()

initial_prices=data['GOLD_PRICE'][9000:].max()


data['YEAR']=data['DATE'].dt.year
data['MONTH']=data['DATE'].dt.month
data['DAY']=data['DATE'].dt.day
data['GRPD']=data['GPRD']/data['GPRD'].max()
data['GRPD_ACT']=data['GPRD_ACT']/data['GPRD_ACT'].max()
data['GRPD_THREAT']=data['GPRD_THREAT']/data['GPRD_THREAT'].max()
data['ELAPSED_DAYS']=(data['DATE']-data['DATE'].min()).dt.days
data['ELAPSED_DAYS']=data['ELAPSED_DAYS']/data['ELAPSED_DAYS'].max()
data['GOLD_PRICE']=data['GOLD_PRICE']/data['GOLD_PRICE'].max()
data['GOLD_PRICE_PREV_DAY'] = data['GOLD_PRICE'].shift(1)
data.dropna(inplace=True)

x_train_raw=np.array(data[['ELAPSED_DAYS','GPRD','GPRD_ACT','GPRD_THREAT','GOLD_PRICE_PREV_DAY']])[:9000]
y_train_raw=np.array(data['GOLD_PRICE'])[:9000].reshape(-1,1)
x_test_raw=np.array(data[['ELAPSED_DAYS','GPRD','GPRD_ACT','GPRD_THREAT','GOLD_PRICE_PREV_DAY']])[9000:]
y_test_raw=np.array(data['GOLD_PRICE'])[9000:].reshape(-1,1)

start=np.array(data['DATE'])[:9000]
print(len(x_train_raw),len(y_train_raw),len(x_test_raw),len(y_test_raw))
print(np.array(data['DATE'])[9000])

x_scaler=MinMaxScaler()
y_scaler=MinMaxScaler()
x_train_scaled=x_scaler.fit_transform(x_train_raw)
y_train_scaled=y_scaler.fit_transform(y_train_raw)
x_test_scaled=x_scaler.transform(x_test_raw)
y_test_scaled=y_scaler.transform(y_test_raw)    

model=tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(5,)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam",loss="mse",metrics=["mae"])
history=model.fit(x_train_scaled,y_train_scaled,epochs=100,batch_size=50,validation_split=0.2)
evaluation=model.evaluate(x_test_scaled,y_test_scaled,verbose=1)
print(f"Model evaluation: \n\tLoss: {evaluation[0]}\n\tMSE: {evaluation[1]}")
joblib.dump(model,"trained")
model=joblib.load("trained")
result=model.predict(x_test_scaled)
y_pred=y_scaler.inverse_transform(result)


start=np.array(data['DATE'])[9000:]
plt.figure(figsize=(12,8))  
plt.plot(start,[i*initial_prices for i in y_test_raw],color='blue',label='Actual')
plt.plot(start,[i*initial_prices for i in y_pred],color='red',label='Predicted')
plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
plt.xticks(rotation=90)
plt.xlabel("Period of prediction")
plt.ylabel("Gold Price in USD")
plt.title(" Actual Gold Price over Time VS Predicted Gold Price over Time")
plt.legend()
plt.show()