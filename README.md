# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required packages and print the present data
Print the placement data and salary data.
Find the null and duplicate values.
Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sahana S
RegisterNumber: 25013621 
*/
```
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")
data.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop(['status', 'salary'], axis=1)
y = data['status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
~~~

## Output:
<img width="309" height="45" alt="image" src="https://github.com/user-attachments/assets/6c7b8530-d283-4fab-97c3-209e3170d090" />
<img width="726" height="541" alt="image" src="https://github.com/user-attachments/assets/a642904e-1d92-4b18-9e61-241a91d748d1" />
<img width="582" height="193" alt="image" src="https://github.com/user-attachments/assets/7331d0fb-bcdf-4ecd-aa60-9eab27eea55e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
