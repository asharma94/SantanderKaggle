# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 20:33:37 2016

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

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

#note: all functions modify input frame

#this one might not actually be that useful 
def drop_duplicate_cols(training, test) :
    remove = []
    cols = training.columns
    for i in range(len(cols)-1):
        v = training[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,training[cols[j]].values):
                remove.append(cols[j])
    training = training.drop(remove, axis=1)
    test = test.drop(remove, axis=1)
    return training, test


def replace_var3_with_mean(training, test):
    training = training.replace(-999999,2)
    test = test.replace(-999999,2)
    return training, test

def one_hot_encode_countries(training, test):
    train = pd.get_dummies(training, columns = ['var3'] )
    test = pd.get_dummies(test, columns = ['var3'])

    # get the columns in train that are not in test
    col_to_add = np.setdiff1d(train.columns, test.columns)

    # add these columns to test, setting them equal to zero
    for c in col_to_add:
        test[c] = 0

    # select and reorder the test columns using the train columns
    test = test[train.columns]
    return train, test
    
def add_feature_for_sum_of_zeros(training, test):
    X = training.iloc[:,:-1]
    Xt = test.iloc[:,:-1]
    X['n0'] = (X==0).sum(axis=1)
    Xt['n0'] = (Xt==0).sum(axis=1)
    training['n0'] = X['n0']
    test['n0'] = Xt['n0']
    return training, test
    

def log_transform_var38_and_split_into_two_features(training, test):
    training['var38ismode'] = np.isclose(training.var38, 117310.979016)
    training['logvar38'] = training.loc[~training['var38ismode'], 'var38'].map(np.log)
    training.loc[training['var38ismode'], 'logvar38'] = 0

    test['var38ismode'] = np.isclose(test.var38, 117310.979016)
    test['logvar38'] = test.loc[~test['var38ismode'], 'var38'].map(np.log)
    test.loc[test['var38ismode'], 'logvar38'] = 0

    return training, test
    
def add_top_5_principal_components(training, test):
    pca = PCA(n_components=5)
    training_copy = training.drop(['TARGET'], axis=1)
    features = training_copy.columns
    pca_training = pca.fit_transform(normalize(training[features], axis=0))
    pca_test = pca.transform(normalize(test[features], axis=0))
    training['PCA_0'] = pca_training[:,0]
    training['PCA_1'] = pca_training[:,1]
    training['PCA_2'] = pca_training[:,2]
    training['PCA_3'] = pca_training[:,3]
    training['PCA_4'] = pca_training[:,4]
    test['PCA_0'] = pca_test[:,0]
    test['PCA_1'] = pca_test[:,1]
    test['PCA_2'] = pca_test[:,2]
    test['PCA_3'] = pca_test[:,3]
    test['PCA_4'] = pca_test[:,4]
    
    return training, test

def remove_low_variance_features(training, test):
    remove = []
    for col in training.columns:
        if training[col].std() == 0:
            remove.append(col)
    dontremove = ['TARGET', 'var38ismode', 'var3']
    print remove
    for elem in dontremove:
        if elem in remove: remove.remove(elem)
            
    #print remove
    training = training.drop(remove, axis=1)
    test = test.drop(remove, axis=1)
    return training, test

def drop_ID(training, test):
    training = training.drop(['ID'], axis=1)
    test = test.drop(['ID'], axis=1)
    return training, test

def standardize_data(training, test):
    features = []
    for col in training.columns:
        features.append(col)
        
    dontremove = ['TARGET', 'var38ismode', 'var3']
    for elem in dontremove:
        if elem in features: features.remove(elem)
    
    ss = StandardScaler()
    training[features] = np.round(ss.fit_transform(training[features]), 6)
    test[features] = np.round(ss.transform(test[features]), 6)
    return training, test

def add_k_means_cluster_as_feature(training, test):
    return training, test
    

def main(standardize=False):
    training = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    training, test = drop_duplicate_cols(training, test)
    training, test = drop_ID(training, test)
    training, test = replace_var3_with_mean(training, test)
    training, test = add_feature_for_sum_of_zeros(training, test)
    training, test = log_transform_var38_and_split_into_two_features(training, test)
    training, test = remove_low_variance_features(training, test)
    training, test = add_top_5_principal_components(training, test)
    if standardize:
        training, test = standardize_data(training, test)
    training, test = one_hot_encode_countries(training, test)
    return training, test
    
    
training, X_test = main(False)
y_train = training['TARGET'].values
X_train = training.drop(['TARGET'], axis=1).values
X_test = X_test.drop(['TARGET'], axis=1).values
test = pd.read_csv('../data/test.csv')
y_test = test['ID']

parameters = {
  'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
  #'solver': ['newton-cg'], 
  'penalty':['l1','l2']
}

regr = linear_model.LogisticRegression(penalty='l2', C=0.1)
grid = GridSearchCV(regr, parameters, cv=3)
grid.fit(X_train, y_train)
print grid.best_params_

regr.fit(X_train, y_train)
predict_outcome = regr.predict_proba(X_test)

with open('sub2.csv', 'wb') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["ID", "TARGET"])
    i = 0
    for x in predict_outcome:
        writer.writerow([y_test[i], x[1]])
        i= i + 1