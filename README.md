# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARSETHA J
RegisterNumber:  212223220032
*/

import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:

![433725630-241e34b7-bc1e-49e7-b56a-2fd2217b399c](https://github.com/user-attachments/assets/fb431cc2-2967-4485-a510-1994e29c4e48)

![433725765-4e44f8e0-8bfd-4270-99f2-cf0c724acf5f](https://github.com/user-attachments/assets/7c5a2fe0-5306-441d-968f-81cd1a68bc0e)

![433725832-9919b46e-22ae-45b4-ac85-4d9c33b398cc](https://github.com/user-attachments/assets/3114442a-9da4-4f03-bfe5-4532acf5eeb6)

![433725951-f5f578da-083e-483e-aa02-7ab2f7402187](https://github.com/user-attachments/assets/b95d90d8-0bea-4656-a083-e3f82f42fb0a)

![433726064-2344bda3-5390-46cb-ad02-9e79f821a3f4](https://github.com/user-attachments/assets/30789318-167f-49ca-8be9-037688fc54ad)

![433726417-873d7e7d-487e-4ea7-a845-8fbc26345d0c](https://github.com/user-attachments/assets/773ab7a0-2b9d-46ce-9b61-e227ebec9239)

![433726586-73031e50-bef6-4c9e-b722-d97fb0283ace](https://github.com/user-attachments/assets/9062b8ad-4c68-4e4b-b637-c892df0f7bf4)

![433726757-caec34ec-e2a2-4834-bc6e-70bed2487baf](https://github.com/user-attachments/assets/a51fd065-b196-4ec4-b597-e543a39619f9)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
