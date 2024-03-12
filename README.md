# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHARSAN KUMAR R
RegisterNumber:212223240028  
*/
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
Placement data:


![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/8719fcd3-382f-4f5c-9747-943091da67df)
Salary data:


![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/d6f40bed-beed-483f-ae04-da206dac833c)


Checking The null() function():


![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/6b4f5c5a-31fb-4c26-aa70-3907079281bf)
Data duplicate:


![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/452e874f-7121-4369-b49e-eb8ef67ceebd)
Print Data:


![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/13d4e1f4-580b-4e5b-97e2-34b220a6b4bc)
Data status():




![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/6e27e324-962e-462e-9bbc-b51ca5204bc3)


Y-prediction array():

![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/8cb597a0-e2d7-4565-9cba-ae5a70fabc60)

Accuracy value:

![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/c0e4f5ff-d33b-4147-aa91-0421e15879d7)

Confusion array:
![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/3bc41d59-6e8f-4087-bac2-dc8315839a25)


Prediction of LR:
![image](https://github.com/DHARSAN23014208/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365413/39b82c6e-2181-4a81-901d-8ce8d46e2f6c)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
