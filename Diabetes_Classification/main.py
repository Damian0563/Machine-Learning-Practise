import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn #type: ignore

file = pd.read_csv('diabetes-dataset.csv')
data = pd.DataFrame(file)
x = data.drop(columns=["Outcome"])
y = data["Outcome"]
bmi_unscaled = x["BMI"]
glucose_unscaled=x["Glucose"]
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)

x_train, x_test, y_train, y_test, bmi_train, bmi_test, glucose_train, glucose_test = train_test_split(
    x_scaled, y, bmi_unscaled,glucose_unscaled, test_size=50, random_state=1
)
model = LogisticRegression(class_weight="balanced")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_probs = model.predict_proba(x_test)[:, 1]
print(classification_report(y_test, y_pred))

# Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['No Diabetes', 'Diabetes'])
plt.yticks(tick_marks, ['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.tight_layout()
plt.show()


plt.scatter(bmi_test, y_probs, alpha=0.6, c=y_test, cmap='bwr')  # Optional color by true class
plt.xlabel("BMI (unscaled)")
plt.ylabel("Diabetes Probability")
plt.title("BMI vs Diabetes Probability")
plt.ylim(0, 1)          
plt.colorbar(label="True Outcome")
plt.show()

plt.scatter(glucose_test, y_probs, alpha=0.6, c=y_test, cmap='bwr')  # Optional color by true class
plt.xlabel("Glucose (unscaled)")
plt.ylabel("Diabetes Probability")
plt.title("Glucose vs Diabetes Probability")
plt.ylim(0, 1)          
plt.colorbar(label="True Outcome")
plt.show()



