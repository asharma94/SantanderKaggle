import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


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

model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=9,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=5
)


def xgbFeatureSelection(X_train, y_train, X_test):
	xgb_param = model.get_xgb_params()
	dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=np.nan)
	cv_result = xgb.cv(
	    xgb_param,
	    dtrain,
	    num_boost_round=model.get_params()['n_estimators'],
	    nfold=5,
	    metrics=['auc'],
	    early_stopping_rounds=50)
	best_n_estimators = cv_result.shape[0]
	model.set_params(n_estimators=best_n_estimators)

	model.fit(X_train, y_train, eval_metric='auc')
	feat_imp = list(pd.Series(model.booster().get_fscore()).sort_values(ascending=False).index)
	return feat_imp




