import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


file = pd.read_csv('diabetes-dataset.csv')
data = pd.DataFrame(file)
x = data.drop(columns=["Outcome"])
y = data["Outcome"]
bmi_unscaled = x["BMI"]
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)

x_train, x_test, y_train, y_test, bmi_train, bmi_test = train_test_split(
    x_scaled, y, bmi_unscaled, test_size=50, random_state=1
)
model = LogisticRegression(class_weight="balanced")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

plt.scatter(bmi_test, y_pred, alpha=0.6)
plt.xlabel("BMI (unscaled)")
plt.ylabel("Predicted Outcome")
plt.yticks([0, 1])
plt.ylim(-0.2, 1.2)
plt.title("Unscaled BMI vs Predicted Diabetes Outcome")
plt.show()

