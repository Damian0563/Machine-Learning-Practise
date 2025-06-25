import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
file = pd.read_csv('students.csv')
data = pd.DataFrame(file)

orig_data = data.copy()
le_gender = LabelEncoder().fit(orig_data["gender"])
le_extra = LabelEncoder().fit(orig_data["extracurricular_participation"])
le_diet = LabelEncoder().fit(orig_data["diet_quality"])
le_job = LabelEncoder().fit(orig_data["part_time_job"])
le_net = LabelEncoder().fit(orig_data["internet_quality"])

encoders = {
    "gender": le_gender,
    "extracurricular_participation": le_extra,
    "diet_quality": le_diet,
    "part_time_job": le_job,
    "internet_quality": le_net
}
data["gender"] = le_gender.transform(data["gender"])
data["extracurricular_participation"] = le_extra.transform(data["extracurricular_participation"])
data["diet_quality"] = le_diet.transform(data["diet_quality"])
data["part_time_job"] = le_job.transform(data["part_time_job"])
data["internet_quality"] = le_net.transform(data["internet_quality"])

x = data.drop(columns=["exam_score", "student_id", "parental_education_level"])
y = data["exam_score"]

plt.scatter(data["sleep_hours"],data["exam_score"])
plt.xlabel("Sleep hours")
plt.ylabel("Actual Exam Result")
plt.show()

plt.scatter(data["study_hours_per_day"],data["exam_score"])
plt.xlabel("Study hours per day")
plt.ylabel("Actual Exam Result")
plt.show()

attendance_unscaled=x["attendance_percentage"]
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)


x_train, x_test, y_train, y_test, attendance_train, attendance_test = train_test_split(x_scaled, y,attendance_unscaled, test_size=30, random_state=1)
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")
# print(classification_report(y_test, y_pred))
def make_prediction(new_df, model, encoders, scaler):
    for column, encoder in encoders.items():
        new_df[column] = encoder.transform(new_df[column])
    x_new = new_df[x.columns]
    x_new_scaled = scaler.transform(x_new)
    return model.predict(x_new_scaled)

new_data = pd.DataFrame([{
    'gender': 'Male',
    'extracurricular_participation': 'Yes',
    'diet_quality': 'Good',
    'part_time_job': 'No',
    'internet_quality': 'Good',
    'study_hours': 4.5,
    'sleep_hours': 7.0,
    'class_participation': 8,
    'age': 17,
    'study_hours_per_day': 4.5,
    'social_media_hours': 2.0,
    'netflix_hours': 1.5,
    'attendance_percentage': 92,
    'exercise_frequency': 3,      
    'mental_health_rating': 7          
}])
predicted_score = make_prediction(new_data, model, encoders, scaler)
print("Predicted Exam Score:", round(predicted_score[0], 2))

plt.scatter(attendance_test,y_pred)
plt.xlabel("Attendance Percantage")
plt.ylabel("Predicted Exam Result")
plt.show()

plt.scatter(y_pred, y_test)
plt.ylabel("Actual Result")
plt.xlabel("Predicted Result")
min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
plt.legend()
plt.show()


