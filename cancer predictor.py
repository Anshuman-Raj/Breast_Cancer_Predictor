# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:53:05 2021

@author: Anshuman Raj
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

#Loading Data
feature_names =['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
data = pd.read_csv('./mammographic_masses.data.txt', na_values=['?'], 
                          names = feature_names)
data.head()
all_features = data[feature_names].drop(['Severity'], axis=1).values
all_classes = data['Severity'].values

scaler = StandardScaler()
scaler.fit(all_features)
all_features=scaler.transform(all_features)
print(all_features)


#splitting train and test cases
X_train, X_test, y_train, y_test = train_test_split(all_features, all_classes, test_size=0.2, random_state=0)


#Setting data and parameters
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 2,
    'eta': 0.03,
    'objective': 'binary:hinge',
    'num_class': 1} 
epochs = 250

#Training and prediction
model = xgb.train(param, train, epochs)
predictions = model.predict(test)


accuracy_score(y_test, predictions)

print(accuracy_score(y_test, predictions))
#saving model
pickle.dump(model, open("Breast_Cancer_Predictor.pkl",'wb'))
