# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:12:49 2018

@author: jubimemore
"""

import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np

with open("best_params_regression.yml", 'r') as read:
    params = yaml.load(read)
    
data = pd.read_csv("x_all.csv")

data = data.iloc[:,:-1]
clusters = ['A', 'B', 'C', 'D', 'E', 'F']


train = data[data["Year"] == 2017]
test = data[data["Year"] == 2018]

train.drop('Year', axis = 1, inplace = True)
test.drop('Year', axis = 1, inplace = True)

X_train = train.iloc[:-1,:].reset_index(drop=True)
y_train = train[clusters].iloc[1:,:].reset_index(drop=True)

X_test = test.iloc[:-1,:].reset_index(drop=True)
y_test = test[clusters].iloc[1:,:].reset_index(drop=True)


cond = [a for a in list(train) if a not in clusters+['Time index', 'Total users']]
shift_for = 1

X_train[cond] = X_train[cond].shift(-shift_for)
X_train = X_train.fillna(method='ffill')

X_test[cond] = X_test[cond].shift(-shift_for)
X_test = X_test.fillna(method='ffill')


# =============================================================================
# GO
# =============================================================================


def makeit(estim, X_train, X_test, y_train, y_test, key):
    combination = pd.read_csv("features_combination_regr.csv", index_col  = 0)

    combos =    {"+time": [e for e in list(train) if e not in ('Apop', 'Bpop', 'Temperature', 'Conditions')],
                     "+time_pops": [e for e in list(train) if e not in ('Temperature', 'Conditions')],
                     "+time_temp/cond": [e for e in list(train) if e not in ('Temperature', 'Conditions')],
                     "+time_pops_temp/cond": list(train)
                     }

    result = {}
    for cluster in clusters:
        X_train_ = X_train[combos[combination[cluster][key]]].values.astype(float)
        X_test_ = X_test[combos[combination[cluster][key]]].values.astype(float)
        
        pipe = Pipeline([('scl', StandardScaler()), ('class', estim["estimator"])])
        
        pipe.set_params(**estim['param_dist'][cluster])
        
        pipe.fit(X_train_, y_train[cluster].values)
        y_pred = pipe.predict(X_test_)
        result[cluster] = mean_absolute_error(y_test[cluster].values, y_pred)/np.mean(y_test[cluster].values)
    
    return result


mult_SVR = SVR()
mult_KR = KernelRidge(kernel='rbf')
mult_RF = RandomForestRegressor()
mult_KNN = KNeighborsRegressor()


RF_param_dist = params['mult_RF']
KNN_param_dist = params['mult_KNN']
SVR_param_dist = params['mult_SVR']
KR_param_dist = params['mult_KR']


estimators = {"mult_RF":{"estimator": mult_RF,"param_dist":RF_param_dist} ,
              "mult_KNN":{"estimator": mult_KNN,"param_dist":KNN_param_dist}, 
              "mult_SVR":{"estimator": mult_SVR,"param_dist":SVR_param_dist}, 
              "mult_KR":{"estimator": mult_KR,"param_dist":KR_param_dist}}


skata = {}
for keys in estimators:
    skata[keys] = makeit(estimators[keys], X_train, X_test, y_train, y_test, keys)
    

accuracies = pd.DataFrame(index=list(estimators.keys()), columns=clusters)

for keys in skata:
    for cluster in skata[keys]:
        accuracies[cluster][keys] = skata[keys][cluster]

accuracies.to_csv("2018_regress_results.csv", index = True)





















