#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:19:32 2019

The flow that leverages the fact that the best prediction is in lag3.

@author: Galloway
"""


import LMNv0
import numpy as np

# Retrieve data and form into LSTM-ready arrays
modelfrom, predictfrom = LMNv0.arrayLSTM(['L','R','C'],'points', 50,
                                         ['points'], not_season=[20182019,20172018], quiet=True)


# Generate predictions

numiters = 35

neurons = 25 #the number of neurons in first layer of NN
epochs = 10
batchsize = 5

for i in range(numiters):
    print("Working on prediction " + str(i+1) + "/" + str(numiters) + " = " + str(int(i/numiters*100)) + "% complete")
    if i == 0:
        result = np.expand_dims(LMNv0.modelrun(modelfrom, predictfrom, neurons, epochs, batchsize), axis=2)
    else:
        result = np.concatenate((result,np.expand_dims(LMNv0.modelrun(modelfrom, predictfrom, neurons, epochs, batchsize), axis=2)),axis=2)


# Summarize the results
        
# Reduce the results down to the best predictions
result_reduced = LMNv0.result_lag_reduce(result)

# Generate the QC plot or plots
LMNv0.act_pred_probabilistic_final(predictfrom, result_reduced)
