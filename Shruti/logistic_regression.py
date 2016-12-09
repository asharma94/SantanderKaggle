# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 22:47:26 2016

@author: Shruti
"""
import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.neural_network import MLPClassifier
import warnings 
warnings.filterwarnings("ignore")

training = pd.read_csv('../data/train.csv', header=0, na_values=['NA'])
test = pd.read_csv('../data/test.csv')

#drop duplicate columns
remove = []
cols = training.columns
for i in range(len(cols)-1):
    v = training[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,training[cols[j]].values):
            remove.append(cols[j])

test.drop(remove, axis=1, inplace=True)
training.drop(remove, axis=1, inplace=True)


training = training.replace(-999999,2)
test = test.replace(-999999,2)

X = training.iloc[:,:-1]
y = training.TARGET
Xt = test.iloc[:,:-1]

X['n0'] = (X==0).sum(axis=1)
Xt['n0'] = (X==0).sum(axis=1)
training['n0'] = X['n0']
test['n0'] = X['n0']

training['var38ismode'] = np.isclose(training.var38, 117310.979016)
training['logvar38'] = training.loc[~training['var38ismode'], 'var38'].map(np.log)
training.loc[training['var38ismode'], 'logvar38'] = 0

test['var38ismode'] = np.isclose(test.var38, 117310.979016)
test['logvar38'] = test.loc[~test['var38ismode'], 'var38'].map(np.log)
test.loc[test['var38ismode'], 'logvar38'] = 0

y_train = training['TARGET'].values
X_train = training.drop(['ID','TARGET'], axis=1).values

y_test = test['ID']
X_test = test.drop(['ID'], axis=1).values


parameters = {
  'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
  #'solver': ['newton-cg'], 
  'penalty':['l1','l2']
}

regr = linear_model.LogisticRegression(penalty='l2', C=0.001)
regr.fit(X_train, y_train)
predict_outcome = regr.predict_proba(X_test)
#grid = GridSearchCV(regr, parameters, cv=3)
#grid.fit(X_train, y_train)
#print grid.best_params_
# write solutions to file

with open('sub2.csv', 'wb') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["ID", "TARGET"])
    i = 0
    for x in predict_outcome:
        writer.writerow([y_test[i], x[1]])
        i= i + 1
