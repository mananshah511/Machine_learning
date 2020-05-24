#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:23:25 2020

@author: manan
"""

#importing needful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')

#creating independent and depenedent matrix
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y.reshape(-1,1))


#fitting model to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#predict the result and applying inverse transform to get actual prediction
y_pred=regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred=sc_y.inverse_transform(y_pred.reshape(-1,1))

#plotting graph for visual
plt.scatter(X,y,color='black')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()