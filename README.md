# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for Gradient Design.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.  

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:NAVEENAA V.R
RegisterNumber:212221220035  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
print("df.head():")
df.head()

print("df.tail():")
df.tail()

#Segregating data to variables
print("Array value of X:")
X=df.iloc[:,:-1].values
X

print("Array value of X:")
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print("Values of Y prediction:")
Y_pred

#displaying actual values
print("Array values of Y test:")
Y_test

#graph plot for training data
print("Training set graph:")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
print("Test set graph:")
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Values of MSE,MAE and RMSE:")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/9150d707-b43c-407a-b596-302dd0b84f43)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/ce8e648d-c93c-4ed7-9401-61a377707ab2)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/7a709fb7-83ad-4681-88b5-c34bfbac7e56)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/6985458c-344c-460d-b0be-b07c957abb74)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/48b69455-e88e-4876-8730-07f93ef2a2ce)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/eb3a004c-18dd-41ff-8cac-63d7d4b5a6b3)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/26cdb599-84f9-48d4-8178-fb2c85b94f87)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/680c0220-9eae-47c9-9bad-7e1b7601d119)
![image](https://github.com/Naveenaa28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131433133/c6093f9a-118d-4d6f-a44c-e8e36b4456cf)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
