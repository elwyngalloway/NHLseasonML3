#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Galloway
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import datetime


#%%
db_name = "NHLseasonML_seasonstats_old.db"

# connect to our database
conn = sqlite3.connect(db_name)

with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    # SQLite statement to retreive the data in question (forwards who have
    # scored more than 50 points in a season):
    cur.execute("SELECT playerId FROM s_skater_summary WHERE points > 50 \
                AND playerPositionCode IN ('C', 'F', 'L', 'R') \
                AND seasonID NOT IN (20182019)")
    
    # Put selected playerIds in an array (playerId is a unique identifier)
    data = np.array(cur.fetchall())

# data contains multiple entries for some players (those who have scored
# more than 50 points in multiple seasons) - isolate unique values
players = np.unique(data)

# show number of unique players
print(players.shape[0], "players identified")


#%% Define a function

def extractlag(player, stat4lag, lag ):
    """
    For now, the stat categories extracted will be hard-coded.
    
    I've also hard-coded the database name and the table name from the database.
    
    player = playerId from the database
    
    stat4lag = name of stat to be lagged (string)
    
    lag = integer value for lagging (must be positive)


    """

    #db_name = "NHLseasonML_seasonstats.db"
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)

    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()

        # Notice that the stats extracted are hard-coded...
        cur.execute("SELECT seasonId, points, goals, ppPoints, shots, timeOnIcePerGame, assists, gamesplayed \
                    FROM s_skater_summary \
                    WHERE seasonId NOT IN (20182019) \
                    AND playerId=?", [player])

        data = cur.fetchall()
    
    if len(data) > 0: # only lag if some data is retreived

        # import data into a dataframe
        df = pd.DataFrame(data)

        # name the columns of df
        df.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games')
        # ensure the results are sorted by year, latest at the top:
        df = df.sort_values(['year'],ascending = False) # this step was not necessary for seasonML1 - results were already sorted!
        # create a dataframe of shifted values - these are lagged w.r.t. the original dataframe
        dfshift = df.shift(lag)
        # name the columns of the shifted df
        dfshift.columns = ('yearlag', 'pointslag', 'goalslag', 'ppPointslag', 'shotslag', 'timeOnIcePerGamelag', 'assistslag', 'gameslag')

        # find the index of the column desired for lagging
        columnindex = df.columns.get_loc(stat4lag)

        # append the appropriate column of the shifted df to the end of the original df
        df = df.join(dfshift.iloc[:,columnindex]).iloc[lag:,:]

        #return df # may consider changing to return an array
        return np.array(df)
    
    else: # return NaNs of appropriate shape in case no data is retreived from database
        
        # create an empty array
        temp = np.empty((1,6)) # should match number of extracted/lagged stats
        # fill it with NaNs
        temp.fill(np.nan)
        # convert to a Dataframe
        df = pd.DataFrame(temp)
        # name these columns to match typical output
        df.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games','pointslag')
        
        #return df
        return np.array(df)
#%% Use function to extract stats for players identified

if 'lagged1' in locals():
    del(lagged1, lagged2, lagged3)
        
for player in players:
    
    # Start with the first lag
    interim1 = extractlag(int(player),'points',1) # create 2D array of a player's performance
    np.array(pd.DataFrame(interim1).dropna(inplace=True)) # ignore "empty" rows
    
    if interim1.shape[0] > 0:
    
        if 'lagged1' in locals(): # if lagged1 already exists, append the player's results to it
            lagged1 = np.append(lagged1, interim1, axis=0)

        else: # else, create lagged1
            lagged1 = interim1[:]

        
        # Now the second lag
        # Ensure lagged2 will have same shape as lagged1 by making each player's
        # contribution have the same shape for each lag.
        interim = np.zeros_like(interim1) - 999 # Identify missing data as -999

        interim2 = extractlag(int(player),'points',2)
        np.array(pd.DataFrame(interim2).dropna(inplace=True))

        interim[:interim2.shape[0],:] = interim2

        if 'lagged2' in locals():
            lagged2 = np.append(lagged2, interim, axis=0)

        else:
            lagged2 = interim[:,:]

        
        # Now the third lag
        interim = np.zeros_like(interim1) - 999

        interim3 = extractlag(int(player), 'points', 3)
        np.array(pd.DataFrame(interim3).dropna(inplace=True))

        interim[:interim3.shape[0],:] = interim3

        if 'lagged3' in locals():
            lagged3 = np.append(lagged3, interim, axis=0)

        else:
            lagged3 = interim[:,:]


# Check that the shapes of the three arrays are identical:
print(lagged1.shape,lagged2.shape,lagged3.shape)

# Convert these arrays into dataframes for convenience later...
lagged1 = pd.DataFrame(lagged1)
lagged1.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')

lagged2 = pd.DataFrame(lagged2)
lagged2.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')

lagged3 = pd.DataFrame(lagged3)
lagged3.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')


#%% Separate training from target data

# predict from the 20152016 season (lag = 1)
lag1predictfrom = lagged1.loc[lagged1['year'] == 20162017]
# model from the remaining seasons
lag1model = lagged1.loc[lagged1['year'] != 20162017]

# predict from the 20142015 season (lag = 2)
lag2predictfrom = lagged2.loc[lagged1['year'] == 20162017] # the rows of interest are in the same position as those in lagged1
# model from the remaining seasons
lag2model = lagged2.loc[lagged1['year'] != 20162017]

lag3predictfrom = lagged3.loc[lagged1['year'] == 20162017]
lag3model = lagged3.loc[lagged1['year'] != 20162017]



# This array contains all data needed test and train the model
modelarrrayfrom = np.transpose(np.dstack((np.array(lag1model),
                                    np.array(lag2model),
                                    np.array(lag3model))), (0,2,1))

# This array is the one that will be predicted from:
predictarrayfrom = np.transpose(np.dstack((np.array(lag1predictfrom),
                                      np.array(lag2predictfrom),
                                      np.array(lag3predictfrom))), (0,2,1))


#%% Let's harness things from here on. Define a function that separates the
#   data into training and testing sets; trains the model; predicts; evaluates
#   prediction quality

def modelrun(modelfrom, predictfrom, nrons, epchs, bsize):
    
    """
    
    """
    
    # We need to address the missing values (-999s) before scaling.
    # Create masks of the modelfrom and predictfrom
    modelfrommask = np.ma.masked_equal(modelfrom,-999).mask
    predictfrommask = np.ma.masked_equal(predictfrom,-999).mask
    # Use them to reassign -999s as max stat value
    modelfrom[modelfrommask] = (np.ones_like(modelfrom)*np.max(modelfrom,(0,1)))[modelfrommask]
    predictfrom[predictfrommask] = (np.ones_like(predictfrom)*np.max(predictfrom,(0,1)))[predictfrommask]
    
    
    #  Apply the 3D scaler:
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Design the scaler:
    # (this flattens the 3D array into 2D, applies determines scaler, then re-stacks in 3D)
    
    scaler = scaler.fit(modelfrom.reshape(-1,modelfrom.shape[2]))
    
    # Apply the scaler:
    modelfrom_scaled = scaler.transform(modelfrom.reshape(-1, modelfrom.shape[2])).reshape(modelfrom.shape)
    predictfrom_scaled = scaler.transform(predictfrom.reshape(-1, predictfrom.shape[2])).reshape(predictfrom.shape)
    
    # Return the missing values to -999
    modelfrom[modelfrommask] = -999
    predictfrom[predictfrommask] = -999
    
    
    
    # Split into test and training sets:
    train, test = train_test_split(modelfrom_scaled,test_size=0.1)
    
    
    # Split into independant and responding variables:
    
    # Split the training data into independant and responding variables:
    train_ind, train_resp = train[:,:,:-1], train[:,:,-1]
    
    # Split test data:
    test_ind, test_resp = test[:,:,:-1], test[:,:,-1]
    
    # Split prediction data:
    predictfrom_ind, predictfrom_resp = predictfrom_scaled[:,:,:-1], predictfrom_scaled[:,:,-1]
    
    #Design and train the LSTM model:
    # Design LSTM neural network
    
    # Define the network using the Sequential Keras API
    model = Sequential()
    
    # Inform algorithm that 0 represents non-values (values of -1 were scaled to 0!)
    model.add(Masking(mask_value=-999, input_shape=(train_ind.shape[1], train_ind.shape[2])))
    
    # Define as LSTM with neurons
    model.add(LSTM(nrons, return_sequences=True))
    model.add(LSTM(nrons))
    
    # I'm not even sure why I need this part, but it doesn't work without it...
    model.add(Dense(train_ind.shape[1]))
    
    # Define a loss function and the Adam optimization algorithm
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    # train network
    history = model.fit(train_ind, train_resp, epochs=epchs, batch_size=bsize, validation_data=(test_ind, test_resp),verbose=0, shuffle=False)

    # plot history
#    plt.plot(history.history['loss'], label='train')
#    plt.plot(history.history['val_loss'], label='test')
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.show()
    
    # Make a prediction:    
    predicted_resp = model.predict(predictfrom_ind)
    
    # Invert scaling:
    
    # Make prediced_resp dimension match predictfrom_ind
    predicted_resp = np.expand_dims(predicted_resp,axis=2)
    
    # Invert scaling for forecast
    
    # Add the predicted values to the independent variables used for the prediction
    inv_predicted = np.concatenate((predictfrom_ind[:,:,:],predicted_resp), axis=2)
    inv_predicted = scaler.inverse_transform(inv_predicted.reshape(-1, inv_predicted.shape[2])).reshape(inv_predicted.shape)
    
    # Make sure the missing data is ignored
    test_predicted = np.empty_like(inv_predicted)
    test_predicted[~predictfrommask] = inv_predicted[~predictfrommask]
    
    # Isolate the predicted values
    inv_predicted_resp = test_predicted[:,:,-1]
    
    # Return results (predicted responding variables):
    return inv_predicted_resp

#%% Run iterations:
#del(result)
numiters = 3
#fig = plt.figure(figsize=(5,5))
#plt.clf()

neurons = 8
epochs = 25
batchsize = 5

for i in range(numiters):
    print("Working on prediction " + str(i+1) + "/" + str(numiters) + " = " + str(int(i/numiters*100)) + "% complete")
    if i == 0:
        result = np.expand_dims(modelrun(modelarrrayfrom, predictarrayfrom, neurons, epochs, batchsize), axis=2)
    else:
        result = np.concatenate((result,np.expand_dims(modelrun(modelarrrayfrom, predictarrayfrom, neurons, epochs, batchsize), axis=2)),axis=2)
        
    
#%% To search for hyperparameters

# define some things to evaluate results:
        # Retrieve the responding variables for predictarrayfrom
actual = predictarrayfrom[:,0,-1]
# Find the mask
#resultmask = np.ma.masked_less(result,1).mask




def hpsearch(modelfrom, predictfrom, modeliter, nrons, epch, bsize):
    
    """
    nrons, epch,bsize are expected to be lists
    """
    
    # Define the result array to be populated
    HPmap = np.empty((len(nrons), len(epch), len(bsize)))
    
    print("Tough to estimate total run time, but working on __ /__ neuron tests")
    
    for N in nrons:
        print(nrons.index(N)+1,"/",len(nrons))
        for E in epch:
            for B in bsize:
                for i in range(modeliter):
                    if i == 0:
                        iterresult = np.expand_dims(modelrun(modelarrrayfrom, predictarrayfrom, N, E, B), axis=2)
                    else:
                        iterresult = np.concatenate((iterresult,np.expand_dims(modelrun(modelarrrayfrom, predictarrayfrom, N, E, B), axis=2)),axis=2)
                        
                # Determine the error for these iterations
                RMSEmeans =np.empty((iterresult.shape[2]))
                
                meanresult = np.zeros((iterresult.shape[0],iterresult.shape[2]))
                
                for iteration in range(iterresult.shape[2]):
                    for player in range(iterresult.shape[0]):
                        meanresult[player,iteration] = np.mean(iterresult[player,:,iteration][np.ma.masked_greater(iterresult[player,:,iteration],2).mask])
                    
                    RMSEmeans[iteration] = np.sqrt(mean_squared_error(meanresult[:,iteration],actual))
                
                del iterresult
                
                HPmap[nrons.index(N), epch.index(E), bsize.index(B)] = np.mean(RMSEmeans)
                
                
    return HPmap

#%% Test the hyperparameters:

# List the HPs to test
neuronlist = [8]
epochlist = [25]
batchlist = [15] 
    
print("Start time:",datetime.datetime.time(datetime.datetime.now()))

result = hpsearch(modelarrrayfrom, predictarrayfrom, 7, neuronlist, epochlist, batchlist)
        
print("End time:",datetime.datetime.time(datetime.datetime.now()))     
        


#%% Plot HP testing results in 3D

fig, ax = plt.subplots()
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
xdata = np.ndarray.flatten(np.expand_dims(np.expand_dims(neuronlist,1),2)*np.ones((len(neuronlist), len(epochlist), len(batchlist))))
ydata = np.ndarray.flatten(np.expand_dims(np.expand_dims(epochlist,0),2)*np.ones((len(neuronlist), len(epochlist), len(batchlist))))
zdata = np.ndarray.flatten(np.expand_dims(np.expand_dims(batchlist,0),0)*np.ones((len(neuronlist), len(epochlist), len(batchlist))))
im = ax.scatter3D(xdata, ydata, zdata, c=np.ndarray.flatten(result), cmap='viridis', s=1000*(17-np.ndarray.flatten(result)))
# Add a colorbar
fig.colorbar(im, ax=ax)

ax.set_xlabel('Neurons')
ax.set_ylabel('Epochs')
ax.set_zlabel('Batch')


b = np.load('HPtestONE.npy')
im = ax.scatter3D(b[:,0], ydata[:,1], zdata[:,2], c=b[:,3], cmap='viridis', s=1000*(17-b[:,3]))


#%% Visualize prediction results in boxplot

# Find sort order - sort players by the mean of all realizations
i = np.argsort(np.mean(meanresult,axis=1))

# Plot the results in a boxplot
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(1,1,1)
ax.boxplot(meanresult[i,:].T)
plt.xlabel('Player')
plt.ylabel('Prediction')
plt.title('Predictions (10 iterations)', fontsize=16)

#%% Conventional performance evaluation:

# Retrieve the responding variables for predictarrayfrom
actual = predictarrayfrom[:,0,-1]

# Find the mask
resultmask = np.ma.masked_less(result,1).mask

# result.shape = [player, lag, iteration]

## Create an empty array for the RMSE results
## The first dimension is the lag, and the second is the run realization
#RMSEs = np.empty((result.shape[1],result.shape[2]))
#
#for iteration in range(result.shape[2]):
#    for lag in range(result.shape[1]):
#        RMSEs[lag,iteration] = np.sqrt(mean_squared_error(result[:,lag,iteration][~resultmask[:,lag,0]], actual[~resultmask[:,lag,0]]))


# Create an alternate measure of error: use mean of the lags for each player
# as the prediction. Calculate the RMSE of these means.         
RMSEmeans =np.empty((result.shape[2]))

meanresult = np.zeros((result.shape[0],result.shape[2]))

for iteration in range(result.shape[2]):
    for player in range(result.shape[0]):
        meanresult[player,iteration] = np.mean(result[player,:,iteration][np.ma.masked_greater(result[player,:,iteration],2).mask])
    
    RMSEmeans[iteration] = np.sqrt(mean_squared_error(meanresult[:,iteration],actual))

# For convenience, capture these errors in a single array. First columns are
# the RMSEs for each lag, followed by the mean of errors for all lags,
# then the error of the mean estimates for all lags.
#RMSEall = np.concatenate((RMSEs,np.expand_dims(np.mean(RMSEs,axis=0),axis=1).T,np.expand_dims(RMSEmeans,axis=1).T),axis=0)

# For now, I think the best representation of the error is the RMSE for
# the mean of the the lag estimates. Report this as error.
error = np.mean(RMSEmeans)

fig2 = plt.figure(figsize=(5,5))
az = fig2.add_subplot(1,1,1)
az.scatter(actual,np.mean(meanresult, axis=1),c="b", s=10)
#az.scatter(actual,np.mean(result[:,0,:][~resultmask[:,0,0]], axis=1),c="b", s=12)
#az.scatter(actual[~resultmask[:,1,0]],np.mean(result[:,1,:][~resultmask[:,1,0]], axis=1),c="r", s=12)
#az.scatter(actual[~resultmask[:,2,0]],np.mean(result[:,2,:][~resultmask[:,2,0]], axis=1),c="g", s=12)
az.plot([0,50,120],[0,50,120])
plt.ylim(-5,110)
plt.xlim(-5,110)
plt.xlabel('Actual Results')
plt.ylabel('Predicted Results')
plt.title('Actual vs. Predicted', fontsize=16)
plt.grid(True)
plt.text(10,85,str('RMSE = '+str(round(float(error),2))),fontsize=16)

#np.save('./results/LAG3_POINTS50/LSTM8-MSE_ADAM-epo64_batch25.npy',result)
