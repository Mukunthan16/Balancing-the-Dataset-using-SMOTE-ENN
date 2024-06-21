# -*- coding: utf-8 -*-

# This Project is done during 2020, There might be some other more accurate balancing methods right now.

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

data= pd.read_csv("/content/dataset.csv")

data.head()

data.info()

print ("Number of rows in the dataset  : " ,data.shape[0])
print ("Number of Columns in the dataset : " ,data.shape[1])
print ("Number of Features : \n" ,data.columns.tolist())
print ("Missing values :  ", data.isnull().sum().values.sum())
print ("Unique values :  \n",data.nunique())

data['TotalCharges']=data["TotalCharges"].replace(r'\s+',np.nan,regex=True)
data['TotalCharges']=pd.to_numeric(data['TotalCharges'])

fill=data.MonthlyCharges*data.tenure

data.TotalCharges.fillna(fill,inplace=True)

#data.isnull().sum()

df=data

def changeColumnsToString(df):
    columnsNames=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
    for col in columnsNames:
        df[col]=df[col].astype('str').str.replace('Yes','1').replace('No','0').replace('No internet service','0').replace('No phone service',0)

changeColumnsToString(df)

df['SeniorCitizen']=df['SeniorCitizen'].astype(bool)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

#df.head()

print("Payment methods: ",df.PaymentMethod.unique())
print("Contract types: ",df.Contract.unique())
print("Gender: ",df.gender.unique())
print("Senior Citizen: ",df.SeniorCitizen.unique())
print("Internet Service Types: ",df.InternetService.unique())

df['gender']=df['gender'].astype('category')
df['PaymentMethod']=df['PaymentMethod'].astype('category')
df['Contract']=df['Contract'].astype('category')
df['SeniorCitizen']=df['SeniorCitizen'].astype('category')
df['InternetService']=df['InternetService'].astype('category')
#df.dtypes

dfPaymentDummies = pd.get_dummies(df['PaymentMethod'], prefix = 'payment')
dfContractDummies = pd.get_dummies(df['Contract'], prefix = 'contract')
dfGenderDummies = pd.get_dummies(df['gender'], prefix = 'gender')
dfSeniorCitizenDummies = pd.get_dummies(df['SeniorCitizen'], prefix = 'SC')
dfInternetServiceDummies = pd.get_dummies(df['InternetService'], prefix = 'IS')

'''
print(dfPaymentDummies.head(3))
print(dfContractDummies.head(3))
print(dfGenderDummies.head(3))
print(dfSeniorCitizenDummies.head(3))
print(dfInternetServiceDummies.head(3))
'''

df.drop(['gender','PaymentMethod','Contract','SeniorCitizen','InternetService'], axis=1, inplace=True)

df = pd.concat([df, dfPaymentDummies], axis=1)
df = pd.concat([df, dfContractDummies], axis=1)
df = pd.concat([df, dfGenderDummies], axis=1)
df = pd.concat([df, dfSeniorCitizenDummies], axis=1)
df = pd.concat([df, dfInternetServiceDummies], axis=1)
df.head(2)

df.columns = ['customerID', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No']

numericColumns=np.array(['Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No'])

for columnName in numericColumns:
    df[columnName]=pd.to_numeric(df[columnName],errors='coerce')
#df.dtypes

df.head()

from sklearn.model_selection import train_test_split

df.Churn.value_counts()

#spliting the testing and training data 


df_test=df[5001:]
df_train=df[:5001]


X_test = df_test.drop('Churn', axis=1)
Y_test = df_test['Churn']

"""# ***UP-SAMPLING***"""

# Separate majority and minority classes
df_majority = df_train[df_train.Churn==0]
df_minority = df_train[df_train.Churn==1]

df_majority.shape,df_minority.shape

from sklearn.utils import resample

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=3688,    # to match majority class
                                 random_state=123)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled

df_up=df_upsampled.sample(frac=1)

df_up['Churn'].value_counts()

# Dividing the dataset into two part one having onlty the target value and other having all other columns
X_up = df_up.drop('Churn', axis=1)
Y_up = df_up['Churn']

X_up.shape, X_test.shape,Y_up.shape, Y_test.shape

X_up=X_up.drop('customerID', axis=1)
X_test=X_test.drop('customerID', axis=1)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(X_up,Y_up)
y_pred = logistic_model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
prec = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
print("accuracy_score : ", acc)
print("precision_score : ", prec)
print("recall_score : ", recall)
print("f1_score : ", f1)

target_names = ['class 0', 'class 1']
print(classification_report(Y_test, y_pred, target_names=target_names))
cf_matrix=confusion_matrix(y_pred,Y_test)
cf_matrix

"""# ***DOWN-SAMPLING***"""

df_majority_downsampled = resample(df_majority, 
                                 replace=True,    # sample without replacement
                                 n_samples=1313,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_down=df_downsampled.sample(frac=1)

df_down['Churn'].value_counts()

# Dividing the dataset into two part one having onlty the target value and other having all other columns
X_down = df_down.drop('Churn', axis=1)
Y_down= df_down['Churn']

X_down.shape, X_test.shape,Y_down.shape, Y_test.shape

X_down=X_down.drop('customerID', axis=1)
#X_test=X_test.drop('customerID', axis=1)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(X_down, Y_down)
y_pred = logistic_model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
prec = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
print("accuracy_score : ", acc)
print("precision_score : ", prec)
print("recall_score : ", recall)
print("f1_score : ", f1)

target_names = ['class 0', 'class 1']
print(classification_report(Y_test, y_pred, target_names=target_names))
cf_matrix=confusion_matrix(y_pred, Y_test)
cf_matrix

"""# ***SMOTE-ENN***"""

X_train_se=df_train.drop(['customerID','Churn'],axis=1)
Y_train_se=df_train['Churn']
X_train_se.shape,Y_train_se.shape

from imblearn.combine import SMOTEENN
sm = SMOTEENN(random_state = 2)
X_train_sen, y_train_sen = sm.fit_resample(X_train_se, Y_train_se.ravel())

X_train_sen.shape,y_train_sen.shape

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(X_train_sen, y_train_sen)
y_pred = logistic_model.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
prec = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
print("accuracy_score : ", acc)
print("precision_score : ", prec)
print("recall_score : ", recall)
print("f1_score : ", f1)

target_names = ['class 0', 'class 1']
print(classification_report(Y_test, y_pred, target_names=target_names))
cf_matrix=confusion_matrix(y_pred, Y_test)
cf_matrix

"""# ***VISUALIZATION***"""

import matplotlib.pyplot as plt
import numpy as np
x = ["class0","class1","accuracy","macro avg","weighted avg"]
y=[0.89,0.53,0,0.71,0.79]
z=[0.74,0.76,0,0.75,0.75]
m=[0.81,0.62,0.75,0.72,0.76]
x_axis=np.arange(len(x))
width=0.25
plt.bar(x_axis - 0.2,y,0.2,label="precision")
plt.bar(x_axis + 0.,z,0.2,label="recall")
plt.bar(x_axis + 0.2,m,0.2,label="f1-score")
plt.xticks(x_axis+width/2,x)
plt.title("Bar Graph For UP-SAMPLING")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = ["class0","class1","accuracy","macro avg","weighted avg"]
y=[0.89,0.53,0,0.71,0.79]
z=[0.74,0.76,0,0.75,0.75]
m=[0.81,0.62,0.75,0.72,0.76]
x_axis=np.arange(len(x))
width=0.25
plt.bar(x_axis - 0.2,y,0.2,label="precision")
plt.bar(x_axis + 0.,z,0.2,label="recall")
plt.bar(x_axis + 0.2,m,0.2,label="f1-score")
plt.xticks(x_axis+width/2,x)
plt.title("Bar Graph For DOWN-SAMPLING")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = ["class0","class1","accuracy","macro avg","weighted avg"]
y=[0.88,0.56,0,0.72,0.79]
z=[0.79,0.70,0,0.75,0.77]
m=[0.83,0.62,0.77,0.73,0.77]
x_axis=np.arange(len(x))
width=0.25
plt.bar(x_axis - 0.2,y,0.2,label="precision")
plt.bar(x_axis + 0.,z,0.2,label="recall")
plt.bar(x_axis + 0.2,m,0.2,label="f1-score")
plt.xticks(x_axis+width/2,x)
plt.title("Bar Graph For SMOTE-ENN")
plt.legend()
plt.show()



