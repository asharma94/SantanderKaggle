# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:24:45 2016

@author: Shruti
"""
import pandas as pd
import numpy as np
import csv
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt

training = pd.read_csv('train.csv', header=0, na_values=['NA'])
test = pd.read_csv('test.csv')

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
#print training.columns
print training.shape

y_train = training['TARGET'].values
X_train = training.drop(['ID','TARGET'], axis=1).values

y_test = test['ID']
X_test = test.drop(['ID'], axis=1).values

clf = RandomForestClassifier(n_jobs=2)
clf.fit(X_train, y_train)
clf_probability = clf.predict_proba(X_test)

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





