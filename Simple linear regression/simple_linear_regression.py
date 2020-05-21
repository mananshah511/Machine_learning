#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:19:27 2020

@author: manan
"""
#simple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Salary_Data.csv')

#creating independent and depenedent matrix
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting the data for train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0) 

#fitting training data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#prediction based on model which we have trained
y_pred=regressor.predict(X_test)

#plotting graph for better understanding and visual(training data)
plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.savefig('Train_graph.png')
plt.show()


#plotting graph for better understanding and visual(testing data)
plt.scatter(X_test,y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Testing data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.savefig('Test_graph.png')
plt.show()


