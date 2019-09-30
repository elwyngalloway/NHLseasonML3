#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:45:55 2019

Predict performance for 20192020

@author: Galloway
"""
import LMNv0
import numpy as np

# Retrieve data and form into LSTM-ready arrays
modelfrom = LMNv0.arrayLSTM(['C','R','L'],'points', 50, ['goals'],
                            not_season=[], quiet=True, forecast=True)[0]

predictfrom = LMNv0.arrayLSTM_forecast(['C','R','L'],'points', 50, ['goals'],
                                       not_season=[], quiet=True)

# Generate predictions

numiters = 80

neurons = 25 #the number of neurons in first layer of NN
epochs = 10
batchsize = 10

for i in range(numiters):
    print("Working on prediction " + str(i+1) + "/" + str(numiters) + " = " + str(int(i/numiters*100)) + "% complete")
    if i == 0:
        result = np.expand_dims(LMNv0.modelrun(modelfrom, predictfrom, neurons, epochs, batchsize), axis=2)
    else:
        result = np.concatenate((result,np.expand_dims(LMNv0.modelrun(modelfrom, predictfrom, neurons, epochs, batchsize), axis=2)),axis=2)


# Summarize the results
result = LMNv0.result_lag_reduce(result)

final = LMNv0.id_result_names(predictfrom, result)

np.save('Probabilistic_goals_forecast_F_20192020.npy',final)

np.savetxt('Probabilistic_goals_forecast_F_20192020.csv', final, delimiter=',', fmt='%s')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    