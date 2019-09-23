#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:35:43 2019

Create a basic stats+ prediction.

@author: Galloway
"""

import LMNv0

# Retrieve data and form into LSTM-ready arrays
modelfrom, predictfrom = LMNv0.arrayLSTM(['L','R','C','F'],'points', 50,
                                         ['points'], not_season=[20182019,20172018], quiet=False)

# Generate prediction
prediction = LMNv0.modelrun(modelfrom, predictfrom, 15, 15, 15)

# Plot results
LMNv0.act_pred_basic(predictfrom, prediction)