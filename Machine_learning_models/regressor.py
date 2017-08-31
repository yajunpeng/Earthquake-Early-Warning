#!/usr/bin/env python3

"""
Several regressors:
    Linear regression: Ridge, Lasso, EN (elastic net)
    SVM (support vector regression)
    RF (Random Forest)
    Ada_RF_exp (adaboosted random forest, exponential loss)

Dimensional reduction:
    Principal compnent analysis (PCA)
    mutual information (MI)

Parameters
----------
X_train: matrix-like
    Training set features
y_train: array-like
    Training true y
X_test: matrix-like      
    Test features
    
X_name: list
    Feature names
n_components: int
    Choose n_components based on PCA or MI     
    
Returns
-------   
Regression: 
Predictions: dict
    1) "y": test predicted y
    2) "y_train": train predicted y
    3) "coef": feature coefficients or importance 
    
Dimension reduction: matrix-like
Lower-dimensional representation of the 
feature matrix    
"""

from __future__ import print_function
import time
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
from utils import print_title
from utils import dump_json, load_json


def train_ridge_linear_model(X_train, y_train, X_test):
    print_title("Ridge Regressor")

    # using the default CV
    alphas = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4]
    reg = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
    reg.fit(X_train, y_train)
    cv_mse = np.mean(reg.cv_values_, axis=0)
    print("alphas: %s" % alphas)
    print("CV MSE: %s" % cv_mse)
    print("Best alpha using built-in RidgeCV: %d" % reg.alpha_)

    # Prediction
    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)

    return {"y": y_pred, "y_train": y_train_pred, "coef": reg.coef_}


def train_lasso_model(X_train, y_train, X_test):
    print_title("Lasso Regressor")

    reg = linear_model.LassoCV(
        precompute=True, cv=5, verbose=1, n_jobs=4)
    reg.fit(X_train, y_train)
    print("alphas: %s" % reg.alphas_)
    print("mse path: %s" % np.mean(reg.mse_path_, axis=1))
    print("Best alpha using bulit-in LassoCV: %f" %
          (reg.alpha_))

    n_nonzeros = (reg.coef_ != 0).sum()
    print("Non-zeros coef: %d" % n_nonzeros)
    
    # Prediction
    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)

    return {"y": y_pred, "y_train": y_train_pred, "coef": reg.coef_}


def train_EN_model(X_train, y_train, X_test):
    print_title("Elastic Net")
    
    # Cross validation to choose the ratio of Lasso
    l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.91, 0.93, 0.95, 0.97, 0.99, 1]

    reg = linear_model.ElasticNetCV(
        l1_ratio=l1_ratio, cv=5, n_jobs=4, verbose=1, precompute=True)
    reg.fit(X_train, y_train)
    n_nonzeros = (reg.coef_ != 0).sum()

    print("best_l1_ratio(%e), n_nonzeros: %d, alpha: %f " % (
            reg.l1_ratio_, n_nonzeros, reg.alpha_))
    
    # Prediction
    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)
    
    return {"y": y_pred, "y_train": y_train_pred, "coef": reg.coef_}


def train_SVM_model(X_train, y_train, X_test):
    print_title("Support Vector Machine")
    
    t1 = time.time()   
    MLM = SVR(cache_size=500)    

    #Cross Validation to choose from RBF and linear kernel
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]},
                        {'kernel': ['linear'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}]                            
        
    reg = GridSearchCV(MLM, tuned_parameters, scoring='neg_mean_squared_error', 
                       cv=5, verbose=1, n_jobs=-1)
    reg.fit(X_train, y_train) 
    joblib.dump(reg, 'EEW_svm.pkl')

    print("Best parameters set found on training set:")
    print(reg.best_params_)
    print("Grid scores on training set:")
    means = reg.cv_results_['mean_test_score']
    stds = reg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, reg.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))    

    t2 = time.time()
    print("Time elapsed: %d s" % (t2 - t1)) 
    
    # Prediction
    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)
    
    return {"y": y_pred, "y_train": y_train_pred, "coef": []}       


def train_random_forest_model(X_train, y_train, X_test):   
    print_title("Random Forest")
    
    t1 = time.time()    
    reg = RandomForestRegressor(n_estimators = 300, oob_score = True, 
                                n_jobs = -1, random_state = 12)
    
    #Use OOB score to select hyperparameters. Trained in one sequence.
    max_features_ratio = [30, 10, 5, 3, 2, 1] 
    num_max_features = [int(X_train.shape[1]/i) for i in max_features_ratio]
    num_min_samples_leaf = [1]
        
    max_oob = 0
    y_para = []        
    for max_features in num_max_features:
        for min_samples_leaf in num_min_samples_leaf: 
            reg.set_params(max_features = max_features,
                           min_samples_leaf = min_samples_leaf)
            reg.fit(X_train, y_train)
            y_oob = reg.oob_score_
            y_para.append([y_oob, max_features, min_samples_leaf])
     
            print("RF model with max_features = %d, min_samples_leaf = %d (oob score = %f) trained." % (
                    max_features, min_samples_leaf, y_oob))
            if max_oob < y_oob:
                max_oob = y_oob
                max_features_best = max_features
                min_samples_leaf_best = min_samples_leaf
    
    print("The best hypoparameter max_features = %d, min_samples_leaf = %d (oob score = %f)." % (
        max_features_best, min_samples_leaf_best, max_oob))          
    if (len(max_features_ratio) > 1) | (len(num_min_samples_leaf) > 1):      
        reg.set_params(max_features = max_features_best,
                       min_samples_leaf = min_samples_leaf_best)
        reg.fit(X_train, y_train) 
    
    joblib.dump(reg, 'EEW_RF_tree300_model.pkl')    
    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)
        
    t2 = time.time()
    print("Time elapsed: %d s" % (t2 - t1))
    
    return {"y": y_pred, "y_train": y_train_pred, "coef": reg.feature_importances_}    


def train_adaboosted_RF_model(X_train, y_train, X_test):
    print_title("AdaBoosted RF")
    
    t1 = time.time() 
    #Hyperparameters for the base estimator are from the RF model
    base_estimator = RandomForestRegressor(n_estimators = 300,
                                           max_features = 847,
                                           min_samples_leaf = 1,
                                           n_jobs = -1)
    reg = AdaBoostRegressor(base_estimator, n_estimators = 50,
                            learning_rate = 1.0, loss = 'exponential',
                            random_state = 12)
    reg.fit(X_train, y_train)
    joblib.dump(reg, 'EEW_ADA_RF_50estimators_model.pkl')
    print("Weights for each estimator: ", reg.estimator_weights_)
    print("Errors for each estimator: ", reg.estimator_errors_)
    
    # Prediction
    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)
    t2 = time.time()
    print("Time elapsed: %d s" % (t2 - t1))    

    return {"y": y_pred, "y_train": y_train_pred, "coef": reg.feature_importances_}
    
      
def dimension_reduction_with_PCA(_X_train, _X_test, n_components):
    print_title("PCA")
    
    t1 = time.time()  
    pca = PCA(n_components = n_components, random_state = 12)
    pca.fit(_X_train)
    X_var_ratio = pca.explained_variance_ratio_ 
    print("%d principal components explain %.4f percent of the variance." %(
            n_components, np.sum(X_var_ratio)*100))
    X_train = pca.transform(_X_train)
    X_test = pca.transform(_X_test)
    t2 = time.time()
    print("Time elapsed: %d s" % (t2 - t1))
    
    return X_train, X_test

def dimension_reduction_with_MI(_X_train, _X_test, y_train, 
                                X_name, n_components):    
    print_title("Mutual Information")
     
    try:
        MI_train = load_json('MI_train.json')
        print('Mutual information file loaded.')
    except IOError:                
        t1 = time.time() 
        mi_train = mutual_info_regression(_X_train, y_train)
        t2 = time.time()
        print('Calculate mutual information: completed. Time: %d s' % (t2 - t1))
        dump_json({'mi': list(mi_train), "features": X_name}, 'MI_train.json')     
        
    feature_index = np.argsort(MI_train['mi'])[-n_components:]
    X_train = _X_train[:,feature_index]
    X_test = _X_test[:,feature_index]    
    
    return X_train, X_test
