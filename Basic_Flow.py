#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:10:47 2019

Recreate the basic ML4NHL flow using LMNv0

@author: Galloway
"""

import LMNv0

# Retrieve data and form into LSTM-ready arrays
modelfrom, predictfrom = LMNv0.arrayLSTM_basic(['L','R','C','F'], 'points', 50, 'points', not_season=[], quiet=False)

# Generate prediction
prediction = LMNv0.modelrun(modelfrom, predictfrom, 15, 15, 15)

# Plot results
LMNv0.act_pred_basic(predictfrom, prediction)