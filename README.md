# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAJKUMAR G
RegisterNumber: 212223230166
*/
import pandas as pd

data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1.head()

data1=data1.drop(['sl_no','salary'],axis=1)

data1.isnull().sum()

data1.duplicated().sum()

data1

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

model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassification Report:\n",cr)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=(2,8))
cm_display.plot()

```

## Output:
![image](https://github.com/user-attachments/assets/f124ba3c-64d7-4860-bc3e-3ccc961668cc)

![image](https://github.com/user-attachments/assets/990870e4-e1f9-4576-8e40-e9954c19df22)

![image](https://github.com/user-attachments/assets/70229037-ca39-47bb-a087-141370029f6a)

![image](https://github.com/user-attachments/assets/d2331e2d-3ba1-49de-91da-5f2b3bd7d65a)

![image](https://github.com/user-attachments/assets/75b6398a-d207-4d6a-84a6-e32159c022bc)

![image](https://github.com/user-attachments/assets/fdbaf2bc-e9f2-4418-9c88-84bb91aac3b1)

![image](https://github.com/user-attachments/assets/1551eacf-13b1-4e8e-8500-6c6e8b768922)

![image](https://github.com/user-attachments/assets/8761a45a-6294-4892-83dc-caaf98afba0b)

![image](https://github.com/user-attachments/assets/4dc67959-940f-4d3c-adbf-3dd005771929)

![image](https://github.com/user-attachments/assets/dc39fdd6-8a03-40d9-95d2-ec1b062bb4bd)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
