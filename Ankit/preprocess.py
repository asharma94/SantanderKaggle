import pandas as pd
from sklearn.preprocessing import StandardScaler

def importDataSets():
	train_set = pd.read_csv('../data/train.csv')
	test_set = pd.read_csv('../data/test.csv')	
	
	X_train = train_set.drop('TARGET', axis=1)
	y_train = train_set['TARGET']
	X_test = test_set

	return X_train, y_train, X_test

def normalizeData(train, test):
	scaler = StandardScaler()

	X_train_scaled = scaler.fit_transform(train)
	X_test_scaled = scaler.transform(test)

	return X_train_scaled, X_test_scaled



