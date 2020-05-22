#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:21:05 2020

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

#Linear regression model
from sklearn.linear_model import LinearRegression
Lin_reg=LinearRegression()
Lin_reg.fit(X,y)

#Polynonial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
Lin_reg_2=LinearRegression()
Lin_reg_2.fit(X_poly,y)

#Graph plot of linear regression
plt.scatter(X,y,color='black')
plt.plot(X,Lin_reg.predict(X),color='blue')
plt.title("Truth of Bluff(Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.savefig("linear_reg.png")
plt.show()

#Graph plot of polynomial regression
plt.scatter(X,y,color='black')
plt.plot(X,Lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Truth of Bluff(Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.savefig("poly_reg_degree4.png")
plt.show()

#predcition based on linear regression
Lin_reg.predict([[6.5]])

#predcition based on polynomial regressiom
Lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))