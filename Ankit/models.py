model = xgb.XGBClassifier(
    # Tree params
    gamma=0,
    max_depth=5,
    min_child_weight=9,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    
    objective= 'binary:logistic',
    seed=5
)

def xgbParamSearch(X_train, y_train, X_test):
    cv_params = {'max_depth': [3,4], 'learning_rate': [0.1], 'min_child_weight' : [5, 7], 
                 'subsample' : [0.8, 0.9], 'colsample_bytree' : [0.8, 0.9]}
    GBM = GridSearchCV(model, cv_params, scoring = 'accuracy', cv = 2, verbose=True)
    GBM.fit(X_train, y_train)
    
    y_pred = GBM.predict_proba(X_test)[:,1]
    y_pred_train = GBM.predict_proba(X_train)[:,1]
    
    fpr, tpr, threshold = metrics.roc_curve(y_train, y_pred)
    print('xgb: %d', metrics.auc(fpr, tpr))
    
    model.set_params(max_depth=GBM.best_params_['max_depth'])
    model.set_params(learning_rate=GBM.best_params_['learning_rate'])
    model.set_params(max_depth=GBM.best_params_['min_child_weight'])
    model.set_params(learning_rate=GBM.best_params_['subsample'])
    model.set_params(learning_rate=GBM.best_params_['learning_rate'])
    
    return y_pred
    

def xgbEarlyStopping(X_train, y_train, X_test):
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
    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    index = pd.Series(model.booster().get_fscore()).sort_values(ascending=False).index
    print(index, feat_imp)
    
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred_train = model.predict_proba(X_train)[:,1]
    
    fpr, tpr, threshold = metrics.roc_curve(y_train, y_pred_train)
    print('xgb: %d', metrics.auc(fpr, tpr))
    
    return y_pred

def Keras_NN(X_train, y_train, X_test, input_dim, output_dim):
    models = Sequential()

    models.add(Dense(120, input_dim=input_dim, init='uniform', W_regularizer=l2(0.00001)))
    models.add(PReLU())
    models.add(BatchNormalization())
    models.add(Dropout(0.6))
    models.add(Dense(output_dim, init='uniform'))
    models.add(Activation('softmax'))

    opt = optimizers.Adagrad(lr=0.0125)
    models.compile(loss='binary_crossentropy', optimizer=opt)
    models.fit(X_train, y_train, nb_epoch=100, batch_size=512, verbose=verbos)
    
    y_pred = models.predict_proba(X_test)[:,1]
    y_pred_train = models.predict_proba(X_train)[:,1]
    
    fpr, tpr, threshold = metrics.roc_curve(y_train, y_pred_train)
    print('xgb: %d', metrics.auc(fpr, tpr))
    
    return y_pred

def randomforestsSearch(X_train, y_train, X_test):
    parameters = { "n_estimators":[300, 350, 400, 450],
           "criterion":["gini", "entropy"],
           "max_features":['auto', 'sqrt', 'log2', 0.4],
           "max_depth":[3, 10, 20, None],
           "min_samples_leaf":[1, 3, 10]
        }
    GBM = GridSearchCV(model, cv_params, scoring = 'accuracy', cv = 2, verbose=True)
    GBM.fit(X_train, y_train)
    print grid.best_params_
    
def randomforests(X_train, y_train, X_test):
    rfc = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=10, min_samples_leaf=1, max_features=0.4, n_jobs=-1, random_state=123) 
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict_proba(X_test)
    return y_pred


def predictCsv(y_test, y_pred):
    with open('sub.csv', 'wb') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["ID", "TARGET"])
    i = 0
    for x in y_pred:
        writer.writerow([y_test[i], x[1]])
        i= i + 1