
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:40:23 2018

@author: spele
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:04:51 2018

@author: spele
"""
import numpy as np
import pandas as pd
#==============================================================================
# ###########     choose parameters that affect the primary dataset     #######
#==============================================================================
#number of clusters and time period interval and filtering hours
clusters = 6
t11, t12, t21, t22 = '00:00:00','00:15:00','23:45:00','00:00:00'

#choose between 'all '2017' '2018'
years = 'all'


from datetime import datetime
t1 = datetime.strptime(t11, '%H:%M:%S')
t2 = datetime.strptime(t12, '%H:%M:%S')
minutes = t2.minute-t1.minute

#create datasets
from defs_after import clustersPeriods
cluster = clustersPeriods(clusters=clusters, timestep=minutes)
data, pois_to_clusters = cluster.assign_users_in_regions(df2 = "artists_list.csv",
                                                         export_csv = False,
                                                         plot = False, years = years,                                         
                                                         random_state = 42, clustering = 'kmeans') #new dataset in dataframe_teliko.csv
#==============================================================================
# #############    Parameters below affect creation of final1     #############
#==============================================================================
#False if you only want his presence
artist_metrics = True
# 'No':metric during live, 'start':metric only at start, 'end':metric only before the bands live
only_start='No'

from final_tune import combine_all
final1, cltopred, clwithlive = combine_all(data=data,
                                            pois_to_clusters=pois_to_clusters,
                                            t11=t11, t12=t12, t21=t21, t22=t22,
                                            artist_metrics=artist_metrics,
                                            cluster=cluster,minutes=minutes,
                                            only_start=only_start,years=years)
if (cltopred not in clwithlive):
   cltopred = str(input('NO live in most significant cluster. Enter cluster to predict:\n')).upper()
print("I predict: ", cltopred)


#==============================================================================
# ######################     Dataset to be scaled       #######################
#==============================================================================
#final2 will be the normalized version of final1 

clusters= ['A','B','C','D','E','F']
final2 = pd.DataFrame()
final2[clusters]=final1[clusters]
for cl in clwithlive:
    final2[cl+"pop"]=final1[cl+"pop"]
final2['Total users'] = final1['Total users']
final2['Temperature'] = final1['Temperature']
final2['Conditions'] = final1['Conditions']
if years=='all':
    final2['Year'] = final1['Year']
final2['Time index'] = final1['Time index']


###optional if you want time series dates
final2['Time series']=final1['Time series dates']

#==============================================================================
# #FROM 5 MINUTES TO 30 MINUTES
#==============================================================================
from theregulator import theregulator
multiplier=12
final_multi = theregulator(final2, minutes, multiplier,clwithlive)

#order columns
columns=np.concatenate([clusters,[clwithlive[0]+'pop',clwithlive[1]+'pop','Total users', 
                                  'Temperature', 'Conditions', 'Time index', 'Time series']])
final_multi = final_multi[columns]
final2=final2[columns[:-1]]

#==============================================================================
# time series plot
#==============================================================================
#live and popularities plot in time
import matplotlib.pyplot as plt
pt=[]
pt2=[]
timeseries = final2['Time series'].values
timeseries2=final_multi['Time series'].values
for i in final2.index.values:
    pt.append(datetime.strptime(timeseries[i],'%Y-%m-%d %H:%M:%S'))
for i in final_multi.index.values:
    pt2.append(datetime.strptime(timeseries2[i],'%Y-%m-%d %H:%M:%S'))
for cl in ['D',''] :
    plt.figure(cl)
    plt.plot(pt, final2[cl])
    plt.scatter(pt,10*final2[cl+'pop'].values)
    plt.title('Main stage cluster number of people and popularities, timestep of 15 min')
    plt.show()
for cl in ['A','B']:   
    #plot final multi live and popularities
    plt.figure(cl+'multi')
    plt.plot(pt2,final_multi[cl])   
    plt.scatter(pt2,(2*final_multi[cl+'pop'].values))
    plt.show()
# =============================================================================
# x, y WRITE TO CSV
# =============================================================================
x = final_multi.iloc[:-1,:].reset_index(drop=True)
y = final_multi[clusters].iloc[1:,:].reset_index(drop=True)

x.to_csv('x_'+years+'.csv', index=False)
y.to_csv('y_'+years+'.csv', index=False)


#==============================================================================
# IMPORT WRITTEN CSV
#==============================================================================
x_2018 = pd.read_csv('x_2018.csv')
y_2018 = pd.read_csv('y_2018.csv')

x_2017 = pd.read_csv('x_2017.csv')
y_2017 = pd.read_csv('y_2017.csv')

#==============================================================================
# PREDICTIONS
#==============================================================================
for cluster in clusters:
    X_train = x_2017
    y_train = y_2017[cluster]
    
##==============================================================================
## #############      Model Creation, Evaluation and Plots     #################
##==============================================================================
##regression model parameters to define
##SVR 
#svr_kernel_vector= ['rbf','linear']#poly rbf linear sigmoid precomputed
##random forest numbers of estimators to use for regression
#estimators = [10]
##polynomial regression maximum degree
#poly_deg = [1]
#
#from models_evaluation_and_plots import model_evaluation_and_plots
#modeler = model_evaluation_and_plots(X_train_sc_flat,X_test_sc_flat,y_train_sc, ytrue, y_train, y_test,
#            sc_cl)
##SVR
#svr=[]
#for svr_kernel in svr_kernel_vector:
#    svr.append(modeler.SVR(svr_kernel))
##Random Forest
#rf=[]
#for estimator in estimators:
#    rf.append(modeler.RF(estimator,max_depth=2, min_samples_split=4))
##polynomial
#po=[]
#for deg in poly_deg:
#    po.append(modeler.poly(deg))
##MLP
#mlp=[modeler.MLP()]
##svr, rf, po, mlp are formated as below:
##[yhat, ytrain_hat, ytest_hat, Sete, Setr,Se, R2te, R2tr, R2, mapete, mapetr, mape]
#
#
##anastasis
##anasdataset
#Xa, Xb, y, ya, yb, ya_sc, yb_sc =\
#sets2(1,final1['Time index'],output_cols,output_cols_sc,dataset,3)
#X_sc = np.concatenate([Xa, Xb])
#y_sc = np.concatenate([ya_sc, yb_sc])
#
##shift (not a model just a shift)
#shift=[modeler.shift(1,output_cols,final1['Time index'],testing_day,years=years)]
##shift is formatted as below
##yhat, ytrain_hat, ytest_hat, Sete, Setr,Se, R2te, R2tr, R2, mapete, mapetr, mape
#yhatshift=shift[0][0]
#ytrueshift=shift[0][1]
