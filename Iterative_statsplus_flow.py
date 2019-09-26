#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 06:49:13 2019

Create probabilistic predictions through iterative predictions

@author: Galloway
"""

import LMNv0
import numpy as np

# Retrieve data and form into LSTM-ready arrays
modelfrom, predictfrom = LMNv0.arrayLSTM(['L','R','C'],'points', 50,
                                         ['points'], not_season=[20182019,20172018], quiet=True)

#actual = predictfrom[:,0,-1] #the actual performance is held here

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

# Generate the QC plot or plots
LMNv0.act_pred_probabilistic(predictfrom, result)






