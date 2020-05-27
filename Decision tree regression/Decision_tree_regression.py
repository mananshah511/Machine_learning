#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:12:40 2020

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

#fitting to dataset using decisiontree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#predict new values(salary)
y_pred=regressor.predict([[6.5]])

#plotting graph to visual model
plt.scatter(X,y,color='black')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Truth or bluff(Decision Tree Regression)")
plt.savefig('DTR.png')
plt.show()

#plotting graph to visula model(higher resolution)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='black')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Truth or bluff(Decision Tree Regression)")
plt.savefig('DTR_Indetails.png')
plt.show()

