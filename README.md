# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook

## Algorithm
1. Start the Program
2. Import the necessary packages.
3. Read the given CSV file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using the SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: John Wilfred Thomas J W
RegisterNumber:  24013517
*/
```
```
import chardet
with open('spam.csv','rb') as file:
    result = chardet.detect(file.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()
data.info()
data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc
```
## Output:

![image](https://github.com/user-attachments/assets/0192997a-4366-423e-860f-704b2719af06)
![image](https://github.com/user-attachments/assets/9d486766-3116-4d9f-a80a-924d939fe9d1)
![image](https://github.com/user-attachments/assets/76cd9698-53a0-4992-928b-f5b22421e5ef)
![image](https://github.com/user-attachments/assets/890f20f6-605e-47b1-8acd-d7eebf7bda8a)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using Python programming.
