# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:12:49 2018

@author: jubimemore
"""

import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

with open("best_params_class.yml", 'r') as read:
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

def classif(x, y, clusters):
    yclass=y.copy()
    dev = 0.05
    #create below array that repeats total users column in case you want the deviation 
    #to be percentage of the total users of each moment
    totalusers = pd.DataFrame(x["Total users"].values.repeat(y.shape[1]).reshape(len(x),y.shape[1]),columns=clusters)
    
    x_clusters = x[clusters]
    yclass[y < (x_clusters-totalusers*dev)]="minus" 
    yclass[y > (x_clusters+totalusers*dev)]="plus"
    for cluster in clusters:
        for k in yclass.index:
            if (not(isinstance(yclass.iloc[k][cluster],str))):
                yclass.iloc[k][cluster]='equal'
    
    for cluster in clusters:
            le = LabelEncoder()
            yclass[cluster] = le.fit_transform(yclass[cluster].values)
    
    return yclass


y_train = classif(X_train, y_train, clusters)
y_test = classif(X_test, y_test, clusters)



# =============================================================================
# GO
# =============================================================================


def makeit(estim, X_train, X_test, y_train, y_test, key):
    combination = pd.read_csv("features_combination_class.csv", index_col  = 0)

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
        result[cluster] = accuracy_score(y_test[cluster].values, y_pred)
    
    return result


mult_RF = OneVsRestClassifier(RandomForestClassifier())
mult_KNN = OneVsRestClassifier(KNeighborsClassifier())
mult_SVC = OneVsRestClassifier(SVC(random_state=0))
mult_NB = OneVsRestClassifier(GaussianNB())



mult_SVC = OneVsRestClassifier(SVC(random_state=0))
mult_NB = OneVsRestClassifier(GaussianNB())
mult_RF = OneVsRestClassifier(RandomForestClassifier())
mult_KNN = OneVsRestClassifier(KNeighborsClassifier())


RF_param_dist = params['mult_RF']
KNN_param_dist = params['mult_KNN']
SVC_param_dist = params['mult_SVC']
NB_param_dist = params['mult_NB']


estimators = {"mult_RF":{"estimator": mult_RF,"param_dist":RF_param_dist} ,
              "mult_KNN":{"estimator": mult_KNN,"param_dist":KNN_param_dist}, 
              "mult_SVC":{"estimator": mult_SVC,"param_dist":SVC_param_dist}, 
              "mult_NB":{"estimator": mult_NB,"param_dist":NB_param_dist}}


skata = {}
for keys in estimators:
    skata[keys] = makeit(estimators[keys], X_train, X_test, y_train, y_test, keys)
    

accuracies = pd.DataFrame(index=list(estimators.keys()), columns=clusters)

for keys in skata:
    for cluster in skata[keys]:
        accuracies[cluster][keys] = skata[keys][cluster]

accuracies.to_csv("2018_classif_results.csv", index = True)





















