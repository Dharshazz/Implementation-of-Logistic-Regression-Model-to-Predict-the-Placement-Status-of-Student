![image](https://github.com/user-attachments/assets/09c292aa-e16b-47fa-8ef1-7835e4c0252c)# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. cse criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.
 

## Program:
```/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Tharshan.R
RegisterNumber:  212223233002
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("/content/Placement_Data.csv")

print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print(data1.isnull().sum())

print(data1.duplicated().sum())

le = LabelEncoder()
categorical_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in categorical_cols:
    data1[col] = le.fit_transform(data1[col])

X = data1.iloc[:, :-1]
y = data1["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)

sample_input = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
sample_prediction = lr.predict(sample_input)
print("Sample Prediction:", sample_prediction)

```

## Output:
![image](https://github.com/user-attachments/assets/24e06126-ffb3-4ca4-b0ce-2daaf0990d3f)
![image](https://github.com/user-attachments/assets/d68d05c8-64d7-4c35-88c6-2b1dc768671c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
