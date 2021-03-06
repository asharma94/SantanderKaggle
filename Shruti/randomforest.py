# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:24:45 2016

@author: Shruti
"""
import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.neural_network import MLPClassifier
import warnings 
warnings.filterwarnings("ignore")
#import seaborn as sns

#import matplotlib.pyplot as plt
#sns.set(style="white", color_codes=True)

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

#mlp2 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10,
#             solver='sgd', verbose=10, tol=1e-4, random_state=42,
#             learning_rate_init=.1)


#random forests

'''
parameters = { 
    'n_estimators': [100, 200, 300, 400, 500, 600, 700],
    #'max_features': ['auto', 'sqrt', 'log2']
}
'''
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=400, oob_score = True)  
#grid = GridSearchCV(rfc, parameters, cv=5)
#grid.fit(X_train, y_train)
#print grid.best_params_
rfc.fit(X_train, y_train)
clf_probability = rfc.predict_proba(X_test)
#clf_probability = grid.predict_proba(X_test)

# write solutions to file
with open('sub2.csv', 'wb') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["ID", "TARGET"])
    i = 0
    for x in clf_probability:
        writer.writerow([y_test[i], x[1]])
        i= i + 1

#svc = svm.SVC(gamma=0.001, C=100.)
#svc.fit(X_train, y_train)
#savetxt('sub2.csv', svc.predict(X_test), delimiter=',', fmt='%f')

#clf = svm.SVC()
#clf.fit(X_train, y_train)

#clf.predict(test)

#training[~np.isnan(training).any(axis=1)]
#print training.columns

#subset_0 = training.iloc[:,:]
#print subset_0.describe()

#X.describe()





