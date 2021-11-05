import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv(r'C:\Users\yangx\Documents\Python Learning\Kaggle Project\Diabetes\diabetes.csv')

print(data)

data.head()

data.describe()

data.info()

data.shape

data.value_counts()

data.dtypes

data.columns

data.isnull().sum()

data.isnull().any()

data.isnull().all()

data.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(data.corr(), annot=True)

data.hist(figsize=(18, 12))
plt.show()

plt.figure(figsize=(14, 10))
sns.set_style(style='whitegrid')
plt.subplot(2, 3, 1)
sns.boxplot(x='Glucose', data=data)
plt.subplot(2, 3, 2)
sns.boxplot(x='BloodPressure', data=data)
plt.subplot(2, 3, 3)
sns.boxplot(x='Insulin', data=data)
plt.subplot(2, 3, 4)
sns.boxplot(x='BMI', data=data)
plt.subplot(2, 3, 5)
sns.boxplot(x='Age', data=data)
plt.subplot(2, 3, 6)
sns.boxplot(x='SkinThickness', data=data)

plt.show()

mean_col = ['Glucose', 'BloodPressure', 'Insulin', 'Age', 'Outcome', 'BMI']
sns.pairplot(data[mean_col], palette='Accent')
plt.show()

sns.boxplot(x='Outcome', y='Insulin', data=data)
plt.show()

sns.regplot(x='BMI', y='Glucose', data=data)
plt.show()

sns.scatterplot(x='Glucose', y='Insulin', data=data)
plt.show()

sns.jointplot(x='SkinThickness', y='Insulin', data=data)
plt.show()

sns.pairplot(data, hue='Outcome')
plt.show()

sns.lineplot(x='Glucose', y='Insulin', data=data)
plt.show()

sns.swarmplot(x='Glucose', y='Insulin', data=data)
plt.show()

sns.barplot(x="SkinThickness", y="Insulin", data=data[170:180])
plt.title("SkinThickness vs Insulin", fontsize=15)
plt.xlabel("SkinThickness")
plt.ylabel("Insulin")
plt.show()
plt.style.use("ggplot")
plt.show()

plt.style.use("default")
plt.figure(figsize=(5, 5))
sns.barplot(x="Glucose", y="Insulin", data=data[170:180])
plt.title("Glucose vs Insulin", fontsize=15)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.show()

# train_test_splitting of the dataset

x = data.drop(columns='Outcome')

# Getting Predicting Value
y = data['Outcome']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train, y_train)

y_pred=reg.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Training Score:\n", reg.score(x_train,y_train)*100)
print("Mean Squared Error:\n", mean_squared_error(y_test, y_pred))
print("R2 score is:\n", r2_score(y_test, y_pred))

print(accuracy_score(y_test,y_pred)*100)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",knn.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)

y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",svc.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gnb.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print("Accuracy Score:\n",gnb.score(x_train,y_train)*100)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)

y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))


print(accuracy_score(y_test,y_pred)*100)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)


y_pred=adb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",adb.score(x_train,y_train)*100)
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
print("R2 score is:\n",r2_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred)*100)

data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(data)