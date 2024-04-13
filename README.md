# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHARSAN KUMAR R
RegisterNumber: 212223240028 
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/30105791-1173-4dda-bb6e-7beddde95a06)

head after dropping sl_no and salary

![320196507-a1ddf9ac-987f-413e-9e1d-b1f4eb701291](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/7406a791-3dd0-45bd-aaa3-21e19e94f96d)

isnull()

![320196581-ad3d675c-b756-4625-a35c-b688cd8385e0](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/69c142b7-836d-4c13-a180-ac03fe8fe47d)

duplicated().sum()

![320196640-135e89f6-649f-4960-8096-3b1c498b5e3d](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/f5b7b66c-951d-4490-b3de-4a673bfb7e0c)

Y_pred

![320196671-f04cc7ea-233a-4a2a-9e01-8be72732b26c](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/79dc3bdf-868a-45cd-9b82-ed518683268a)

accuracy

![320196725-03b4af28-c99a-42eb-927c-376aaebf3866](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/d23e069b-5bd2-4f77-8735-d6a94a1cd7f1)

classification report

![320196705-14fc1bad-f245-4531-971d-282c049711d1](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/d5a5643b-bb60-43a0-a25f-b303d738f84e)

Predict

![320196720-4b0e911d-9017-488e-88bc-903791e5a413](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/8418fc6c-864a-45ef-a987-a6d93a7d4732)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
