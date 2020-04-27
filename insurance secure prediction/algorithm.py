# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 01:55:36 2019

@author: Touqeer Malik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:03:13 2019

@author: Touqeer Malik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('Predict_Ins.Score_WData_AAO.csv')
x=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values


 
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,:1])
x[:,:1]=imputer.transform(x[:,:1])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x,y)

weight_score=input("Enter the weight score :")

#predicting new result
ins_score=regressor.predict([[weight_score]])
print("The insurance score is : ",ins_score)






