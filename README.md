# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
# importing library

import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading

data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

# now, we are checking start with a pairplot, and check for missing values

sns.heatmap(data.isnull(),cbar=False)

# Data Cleaning and Data Drop Process

data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())

# Change to categoric column to numeric

data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values

data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric

data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

# Drop unnecessary columns

drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

# heatmap for train dataset

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

# Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

# Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

# Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
# OUTPUT():
# Dataset:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/a309a9b1-bce1-4e7d-88b9-a9a929c2c778)
# data.tail():
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/4e824843-7f83-4f48-8dc4-20df9d620fb4)
# data.insull().sum():
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/302d8fc6-8a6f-4d62-8340-5ab171a9a797)
# data.describe():
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/1cf6f1b9-4383-4ad4-8cb5-22b943183425)
# heatmap:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/3ad0a2a8-4f38-47f7-9270-0badae96b6fc)
# Data after cleaning:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/7eb410dd-9904-4937-91ca-7524801963d5)
# Data on Heatmap:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/3e00af8a-4c93-4919-a2de-1ca8ff81bbfe)

# Cleaned Null values:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/6d2047aa-7546-466a-8f8a-dd7e595bdcf8)
# Report of (people survived & Died):
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/85abee4c-6f76-4c0c-bf68-329e07ca5b31)

# Report of Survived People's Age:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/79454901-8c04-4cf9-9b4e-bc14055f46df)


# Report of pessengers:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/0cb0e292-4b35-49c9-b8f6-4456fa56a4b9)

# Report:
![image](https://github.com/Jeevithaelumalai/Ex-07-Feature-Selection/assets/118708245/626fd32b-4da7-466a-8f48-7dcab2db840c)

# Result:
Thus, Sucessfully performed the various feature selection techniques on a given dataset
