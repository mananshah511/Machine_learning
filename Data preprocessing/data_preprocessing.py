# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing needful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Data.csv')

#creating independent and depenedent matrix
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#hadnling missing data using imputer from sk-learn library
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#encoding of data for X
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder_X=LabelEncoder()
X[:,0]=LabelEncoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#encoding of data for y
LabelEncoder_y=LabelEncoder()
y=LabelEncoder_y.fit_transform(y)

#splitting the data for train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)