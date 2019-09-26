#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNM: Library for Machine learning of NHL.

Created on Wed Sep  4 06:40:16 2019

Library-ization of pieces of code used for seasonML2

Database retrieval is performed by individual player, for specific lag, and
thus cannot be separated from LSTM array prep

Wrap Array prep as function (position, point total, ?, ?)
    merge player check with retrieval, with user prompt (basic - done)
    Can I make the code more flexible re cats?
    Extract + Lag is defined in ML_V2 (done)    
    option to save array, at least for testing
    Variations of array generators for production prediction
Iterativie prediction (not done)
LSTM Model building (not done)
Prediction formatting and summarization (not done)
Plot generation (not done)


@author: Galloway
"""

import sqlite3
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.backend import clear_session

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import datetime

#Constants within library
db_name =  "NHLseasonML_seasonstats.db"

#%%
def extractlag_basic(player, stat4lag, lag, not_season=[] ):
    """
    This function extracts the basic stats, which are hard-coded.
    Some of the suggested functionality isn't actually working 
    
    I've also hard-coded the table name from the database.
    
    player = playerId from the database
    
    stat4lag = name of stat to be lagged (string)
    
    lag = integer value for lagging (must be positive)
    
    not_season = list of seasons not to be included in analysis
                example: [20162017,20172018]


    """
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)

    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()

        # Notice that the stats extracted are hard-coded...
        cur.execute("SELECT seasonId, points, goals, ppPoints, shots, timeOnIcePerGame, assists, gamesplayed \
                    FROM s_skater_summary \
                    WHERE seasonID NOT IN ({}) \
                    AND playerId={}".format(','.join('?' * len(not_season)),'?'),not_season + [player])

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
    


def extractlag(player, stats4lag, lag, not_season = []):
    """
    This one exstacts a wider variety of stats
    
    For now, the stat categories extracted will be hard-coded.
    
    I've also hard-coded the table name from the database.
    
    player = playerId from the database
    
    stats4lag = name of stat(s) to be lagged [list of strings]
    
    lag = integer value for lagging (must be positive)
    
    not_season = list of seasons not to be included in analysis
                example: [20162017,20172018]
    """
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)

    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()
        
        # I want to retrieve stats for a season, plus age, draft position,
        # position code (categorical!), name?
        cur.execute("SELECT DISTINCT s_skater_summary.seasonId, \
                s_skater_summary.playerId, s_bio_info.playerBirthDate, \
                s_bio_info.playerDraftOverallPickNo, s_skater_summary.playerPositionCode, \
                s_skater_summary.assists, s_skater_summary.goals, \
                s_skater_summary.shots, s_skater_summary.timeOnIcePerGame, \
                s_skater_summary.gamesplayed, \
                s_skater_summary.plusMinus, \
                s_time_on_ice.ppTimeOnIcePerGame, s_time_on_ice.shTimeOnIcePerGame, \
                s_time_on_ice.evTimeOnIcePerGame, \
                s_skater_points.evAssists, s_skater_points.ppAssists, \
                s_skater_goals.enGoals, s_skater_goals.ppGoals, s_skater_goals.evGoals, \
                s_realtime_events.blockedShots, s_realtime_events.hitsPerGame \
                FROM s_skater_summary \
                INNER JOIN s_bio_info \
                    ON s_bio_info.playerId = s_skater_summary.playerId \
                    AND s_bio_info.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_time_on_ice \
                    ON s_time_on_ice.playerId = s_skater_summary.playerId \
                    AND s_time_on_ice.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_skater_points \
                    ON s_skater_points.playerId = s_skater_summary.playerId \
                    AND s_skater_points.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_skater_goals \
                    ON s_skater_goals.playerId = s_skater_summary.playerId \
                    AND s_skater_goals.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_realtime_events \
                    ON s_realtime_events.playerId = s_skater_summary.playerId \
                    AND s_realtime_events.seasonId = s_skater_summary.seasonId \
                WHERE s_skater_summary.playerID = ? \
                AND s_skater_summary.seasonID NOT IN ({})".format(','.join('?' * len(not_season))),[player]+not_season)
        
        

        data = cur.fetchall()
    
    if len(data) > 0: # only lag if some data is retreived

        # import data into a dataframe
        df = pd.DataFrame(data)

        # name the columns of df
        df.columns = ('year', 'playerID', 'birthYear','draftPos', 'position',
                      'assists', 'goals',
                      'shots', 'ToipG', 'games', 'plusMinus', 'ppToipG',
                      'shToipG', 'evToipG', 'evAssists', 'ppAssists', 'enGoals',
                      'ppGoals', 'evGoals', 'blocks', 'hitsPerGame')
        # transform birth date to just birth year, then transform to age, then rename column
        df['birthYear'] = pd.to_datetime(df['birthYear']).dt.year
        df['birthYear'] = df['year'] // 10000 - df['birthYear']
        df = df.rename(index=str, columns={'birthYear': 'age'})
        # deal with the categorical data: Pandas has a function that helps...
        # define names of all categories expected:
        df['position'] = df['position'].astype('category',categories=['C', 'D', 'L', 'R'])
        # append columns for each position
        df = pd.concat([df,pd.get_dummies(df['position'], prefix='position')],axis=1)
        # drop original position column
        df.drop(['position'],axis=1, inplace=True)
        # some players were never drafted - leaves blank in draftPos. Define this as 300
        df['draftPos'].replace('', 300, inplace=True)
        # calculate a points column
        df.insert(6,'points',df['goals']+df['assists'])
        # convert hits per game to hits (per season)
        df = df.rename(index=str, columns={'hitsPerGame': 'hits'})
        df['hits'] = df['hits']*df['games']
        # ensure the results are sorted by year, latest at the top:
        df = df.sort_values(['year'],ascending = False)
        # create a dataframe of shifted values - these are lagged w.r.t. the original dataframe
        dfshift = df.shift(lag)        
        

        # Add the lagged stats:
        
        for stat in stats4lag:
            
            # name the columns of the shifted df
            dfshift = dfshift.rename(index=str, columns={stat : str(stat + 'lag')})
            
            # find the index of the column desired for lagging
            columnindex = df.columns.get_loc(stat)

            # append the appropriate column of the shifted df to the end of the original df
            df = df.join(dfshift.iloc[:,columnindex])
            
        df = df.iloc[lag:,:]
        
        #return df # may consider changing to return an array
        #return np.array(df)
        return df
    
    else: # return NaNs of appropriate shape in case no data is retreived from database
        
        # create an empty array
        temp = np.empty((1,29))
        # fill it with NaNs
        temp.fill(np.nan)
        # convert to a Dataframe
        df = pd.DataFrame(temp)
        # name these columns to match typical output 
        df.columns = ('year', 'playerID', 'birthYear','draftPos', 'position_C', 'position_D',
                      'position_L', 'position_R', 'assists', 'goals',
                      'shots', 'ToipG', 'games', 'plusMinus', 'ppToipG',
                      'shToipG', 'evToipG', 'evAssists', 'ppAssists', 'enGoals',
                      'ppGoals', 'evGoals', 'blocks', 'hitsPerGame')
        #return df
        r#eturn np.array(df)
        return df.reset_index(drop=True)
 




def arrayLSTM_basic(positions, filter_stat, min_stat, stat4lag, not_season=[], quiet=False):
    """
    This function will generate an LSTM-ready, lagged array for a set of
    players, given their positions, a stat on which to filter players,
    a minimum total for filter stat, and the stat to be predicted. Seasons can
    be ignored.
    
    positions = a list of strings, like this: ['C', 'L', 'R', 'D']
    
    filter_stat = a string defining which stat will be used to filter players
                NOTE: I haven't figured out how to do this yet, so it's
                hard-coded as 'points'
    
    min_stat = a number defining the minimum value for the stat specified.
                Players not acheiving that value (or above) in their careers
                will not be included in the retrieval. Note: a player's entire
                career is retrieved if they pass the filter criterion for
                any season.
    
    stat4lag = a string defining the stat to be predicted
    
    not_season = a list defining seasons ignored by the extraction. For example,
                [20152016, 20182019]
                
    quiet = a parameter to toggle requirement for user input. True means no
                interaction required.
    """
    #Retrieve the players to be included
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)
    
    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()
    
        # SQLite statement to retreive the data in question (forwards who have
        # scored more than min_stat points in a season):
        cur.execute("SELECT playerId FROM s_skater_summary WHERE {} > {} \
                    AND playerPositionCode IN ({}) \
                    AND seasonID NOT IN ({})".format(filter_stat,'?',','.join('?' * len(positions)),','.join('?' * len(not_season))),[min_stat]+positions+not_season)
    
        # Put selected playerIds in an array (playerId is a unique identifier)
        data = np.array(cur.fetchall())

    # data contains multiple entries for some players (those who have scored
    # more than 50 points in multiple seasons) - isolate unique values
    players = np.unique(data)

    # show number of unique players, and prompt to continue
    print(players.shape[0], "players identified")
    if quiet == False:
        if input('Continue? y/n : ') != 'y':
            print('Not continuing')
            return
        else:
            print('Conintuing')


    #Retrieve the stats from the database and apply a lag
    for player in players:
    
        # Start with the first lag
        interim1 = extractlag_basic(int(player),'points' ,1,not_season=not_season) # create 2D array of a player's performance
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
    
            interim2 = extractlag_basic(int(player),'points',2,not_season=not_season)
            np.array(pd.DataFrame(interim2).dropna(inplace=True))
    
            interim[:interim2.shape[0],:] = interim2
    
            if 'lagged2' in locals():
                lagged2 = np.append(lagged2, interim, axis=0)
    
            else:
                lagged2 = interim[:,:]
    
            
            # Now the third lag
            interim = np.zeros_like(interim1) - 999
    
            interim3 = extractlag_basic(int(player), 'points', 3,not_season=not_season)
            np.array(pd.DataFrame(interim3).dropna(inplace=True))
    
            interim[:interim3.shape[0],:] = interim3
    
            if 'lagged3' in locals():
                lagged3 = np.append(lagged3, interim, axis=0)
    
            else:
                lagged3 = interim[:,:]
    
    
    # Check that the shapes of the three arrays are identical:
    if lagged1.shape==lagged2.shape and lagged2.shape==lagged3.shape:
        print('Lagged arrays all have shape ',lagged1.shape)
    else:
        print('Lagged arrays dont have the same shape :( ')
        print(lagged1.shape,lagged2.shape,lagged3.shape)
        print('Not continuing')
        return
    
    # Convert these arrays into dataframes for convenience later...
    lagged1 = pd.DataFrame(lagged1)
    lagged1.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')
    
    lagged2 = pd.DataFrame(lagged2)
    lagged2.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')
    
    lagged3 = pd.DataFrame(lagged3)
    lagged3.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')
    
    
    #Separate training from target data - always trying to build the model by
        #predicting the latest season
        
    lastseason = int(max(lagged1['year']))
    
    # predict from the 20152016 season (lag = 1)
    lag1predictfrom = lagged1.loc[lagged1['year'] == lastseason]
    # model from the remaining seasons
    lag1model = lagged1.loc[lagged1['year'] != lastseason]
    
    # predict from the 20142015 season (lag = 2)
    lag2predictfrom = lagged2.loc[lagged1['year'] == lastseason] # the rows of interest are in the same position as those in lagged1
    # model from the remaining seasons
    lag2model = lagged2.loc[lagged1['year'] != lastseason]
    
    lag3predictfrom = lagged3.loc[lagged1['year'] == lastseason]
    lag3model = lagged3.loc[lagged1['year'] != lastseason]
    
    
    # This array contains all data needed test and train the model
    modelarrayfrom = np.transpose(np.dstack((np.array(lag1model),
                                        np.array(lag2model),
                                        np.array(lag3model))), (0,2,1))
    
    # This array is the one that will be predicted from:
    predictarrayfrom = np.transpose(np.dstack((np.array(lag1predictfrom),
                                          np.array(lag2predictfrom),
                                          np.array(lag3predictfrom))), (0,2,1))
    
    # show number of unique players, and prompt to continue
    print('Arrays for modelling and predicting have shape ',
          modelarrayfrom.shape, predictarrayfrom.shape)
    if quiet == False:
        if input('Continue? y/n : ') != 'y':
            print('Not continuing')
            return
        else:
            print('Conintuing')
            
    
    #return the arrays for model building and predicting
    return modelarrayfrom, predictarrayfrom


# Let's harness things from here on. Define a function that separates the
#   data into training and testing sets; trains the model; predicts; evaluates
#   prediction quality

def arrayLSTM(positions, filter_stat, min_stat, stats4lag, not_season=[], quiet=False, forecast=False):
    """
    This function will generate an LSTM-ready, lagged array for a set of
    players, given their positions, a stat on which to filter players,
    a minimum total for filter stat, and the stat to be predicted. Seasons can
    be ignored.
    
    positions = a list of strings, like this: ['C', 'L', 'R', 'D']
    
    filter_stat = a string defining which stat will be used to filter players
    
    min_stat = a number defining the minimum value for the stat specified.
                Players not acheiving that value (or above) in their careers
                will not be included in the retrieval. Note: a player's entire
                career is retrieved if they pass the filter criterion for
                any season.
    
    stat4lag = a string defining the stat to be predicted
    
    not_season = a list defining seasons ignored by the extraction. For example,
                [20152016, 20182019]
                
    quiet = a parameter to toggle requirement for user input. True means no
                interaction required.
                
    forecast = a parameter to toggle if function is used for forecasting. If True,
                the latest season is not removed from model-from array
    """
    #Retrieve the players to be included
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)
    
    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()
    
        # SQLite statement to retreive the data in question (forwards who have
        # scored more than min_stat points in a season):
        cur.execute("SELECT playerId FROM s_skater_summary WHERE {} > {} \
                    AND playerPositionCode IN ({}) \
                    AND seasonID NOT IN ({})".format(filter_stat,'?',','.join('?' * len(positions)),','.join('?' * len(not_season))),[min_stat]+positions+not_season)
    
        # Put selected playerIds in an array (playerId is a unique identifier)
        data = np.array(cur.fetchall())

    # data contains multiple entries for some players (those who have scored
    # more than 50 points in multiple seasons) - isolate unique values
    players = np.unique(data)

    # show number of unique players, and prompt to continue
    print(players.shape[0], "players identified")
    if quiet == False:
        if input('Continue? y/n : ') != 'y':
            print('Not continuing')
            return
        else:
            print('Conintuing')


    #Retrieve the stats from the database and apply a lag
    for player in players:
    
        # Start with the first lag
        interim1 = extractlag(int(player),stats4lag,1,not_season=not_season) # create 2D DataFrame of a player's performance
        
        if interim1.shape[0] > 0:
    
            if 'lagged1' in locals(): # if lagged1 already exists, append the player's results to it
                lagged1 = pd.DataFrame.append(lagged1, interim1)
    
            else: # else, create lagged1
                lagged1 = interim1.copy()
                
            lagged1.reset_index(inplace=True, drop=True)
                
                
            # Now the second lag
            # Ensure lagged2 will have same shape as lagged1 by making each player's
            # contribution have the same shape for each lag.
            interim = interim1.copy() * 0 - 999 # Identify missing data as -999
    
            interim2 = extractlag(int(player),stats4lag,2,not_season=not_season)
    
            interim2 = pd.DataFrame.append(interim2,interim.iloc[:(interim.shape[0]-interim2.shape[0]),:]).reset_index(drop=True)
    
            if 'lagged2' in locals():
                lagged2 = pd.DataFrame.append(lagged2, interim2)
    
            else:
                lagged2 = interim2.copy()
            
            lagged2.reset_index(inplace=True, drop=True)

 
            # Now the third lag
            interim = interim1.copy() * 0 - 999 # Identify missing data as -999
    
            interim3 = extractlag(int(player),stats4lag,3,not_season=not_season)
    
            interim3 = pd.DataFrame.append(interim3,interim.iloc[:(interim.shape[0]-interim3.shape[0]),:]).reset_index(drop=True)
    
            if 'lagged3' in locals():
                lagged3 = pd.DataFrame.append(lagged3, interim3)
    
            else:
                lagged3 = interim3.copy()  
                
            lagged3.reset_index(inplace=True, drop=True)
    
    
    # Check that the shapes of the three arrays are identical:
    if lagged1.shape==lagged2.shape and lagged2.shape==lagged3.shape:
        print('Lagged arrays all have shape',lagged1.shape)
    else:
        print('Lagged arrays dont have the same shape :( ')
        print(lagged1.shape,lagged2.shape,lagged3.shape)
        print('Not continuing')
        return

   
    # Separate training from target data - always trying to build the model by
    # predicting the latest season
        
    lastseason = int(max(lagged1['year']))
    
    # predict from the 20152016 season (lag = 1)
    lag1predictfrom = lagged1.loc[lagged1['year'] == lastseason]
    # model from the remaining seasons
    lag1model = lagged1.loc[lagged1['year'] != lastseason]
    
    # predict from the 20142015 season (lag = 2)
    lag2predictfrom = lagged2.loc[lagged1['year'] == lastseason] # the rows of interest are in the same position as those in lagged1
    # model from the remaining seasons
    lag2model = lagged2.loc[lagged1['year'] != lastseason]
    
    lag3predictfrom = lagged3.loc[lagged1['year'] == lastseason]
    lag3model = lagged3.loc[lagged1['year'] != lastseason]
    
    if forecast == True:
        # overwrite the lagXmodel arrays
        lag1model = lagged1.copy()
        lag2model = lagged2.copy()
        lag3model = lagged3.copy()
        
        #overwrite the lagXpredictfrom arrays
        lag1predictfrom = np.zeros((2,2))
        lag2predictfrom = np.zeros((2,2))
        lag3predictfrom = np.zeros((2,2))
    
    
    # This array contains all data needed test and train the model
    modelarrayfrom = np.transpose(np.dstack((np.array(lag1model),
                                        np.array(lag2model),
                                        np.array(lag3model))), (0,2,1))
    
    # This array is the one that will be predicted from:
    predictarrayfrom = np.transpose(np.dstack((np.array(lag1predictfrom),
                                          np.array(lag2predictfrom),
                                          np.array(lag3predictfrom))), (0,2,1))
    
    # check array shapes
    print('Arrays for modelling and predicting have shape ',
          modelarrayfrom.shape, predictarrayfrom.shape)
    if quiet == False:
        if input('Continue? y/n : ') != 'y':
            print('Not continuing')
            return
        else:
            print('Conintuing')
            
    
    #return the arrays for model building and predicting
    return modelarrayfrom, predictarrayfrom


def arrayLSTM_forecast(positions, filter_stat, min_stat, stats4lag, not_season=[], quiet=False):
    """
    This function will generate an LSTM-ready, lagged array for a set of
    players, given their positions, a stat on which to filter players,
    a minimum total for filter stat, and the stat to be predicted. Seasons can
    be ignored.
    
    The difference with _forecast is that it uses the latest stats to forecast
    a future season. Note: this only returns a predict-from array.
    
    positions = a list of strings, like this: ['C', 'L', 'R', 'D']
    
    filter_stat = a string defining which stat will be used to filter players
    
    min_stat = a number defining the minimum value for the stat specified.
                Players not acheiving that value (or above) in their careers
                will not be included in the retrieval. Note: a player's entire
                career is retrieved if they pass the filter criterion for
                any season.
    
    stat4lag = a string defining the stat to be predicted
    
    not_season = a list defining seasons ignored by the extraction. For example,
                [20152016, 20182019]
                
    quiet = a parameter to toggle requirement for user input. True means no
                interaction required.
    """
    #Retrieve the players to be included
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)
    
    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()
    
        # SQLite statement to retreive the data in question (forwards who have
        # scored more than min_stat points in a season):
        cur.execute("SELECT playerId FROM s_skater_summary WHERE {} > {} \
                    AND playerPositionCode IN ({}) \
                    AND seasonID NOT IN ({})".format(filter_stat,'?',','.join('?' * len(positions)),','.join('?' * len(not_season))),[min_stat]+positions+not_season)
    
        # Put selected playerIds in an array (playerId is a unique identifier)
        data = np.array(cur.fetchall())

    # data contains multiple entries for some players (those who have scored
    # more than 50 points in multiple seasons) - isolate unique values
    players = np.unique(data)

    # show number of unique players, and prompt to continue
    print(players.shape[0], "players identified")
    if quiet == False:
        if input('Continue? y/n : ') != 'y':
            print('Not continuing')
            return
        else:
            print('Conintuing')


    #Retrieve the stats from the database and apply a lag
    for player in players:
    
        # Start with the first lag
        interim1 = extractlag(int(player),stats4lag,0,not_season=not_season) # create 2D DataFrame of a player's performance
        
        if interim1.shape[0] > 0:
    
            if 'lagged1' in locals(): # if lagged1 already exists, append the player's results to it
                lagged1 = pd.DataFrame.append(lagged1, interim1)
    
            else: # else, create lagged1
                lagged1 = interim1.copy()
                
            lagged1.reset_index(inplace=True, drop=True)
                
                
            # Now the second lag
            # Ensure lagged2 will have same shape as lagged1 by making each player's
            # contribution have the same shape for each lag.
            interim = interim1.copy() * 0 - 999 # Identify missing data as -999
    
            interim2 = extractlag(int(player),stats4lag,1,not_season=not_season)
    
            interim2 = pd.DataFrame.append(interim2,interim.iloc[:(interim.shape[0]-interim2.shape[0]),:]).reset_index(drop=True)
    
            if 'lagged2' in locals():
                lagged2 = pd.DataFrame.append(lagged2, interim2)
    
            else:
                lagged2 = interim2.copy()
            
            lagged2.reset_index(inplace=True, drop=True)

 
            # Now the third lag
            interim = interim1.copy() * 0 - 999 # Identify missing data as -999
    
            interim3 = extractlag(int(player),stats4lag,2,not_season=not_season)
    
            interim3 = pd.DataFrame.append(interim3,interim.iloc[:(interim.shape[0]-interim3.shape[0]),:]).reset_index(drop=True)
    
            if 'lagged3' in locals():
                lagged3 = pd.DataFrame.append(lagged3, interim3)
    
            else:
                lagged3 = interim3.copy()  
                
            lagged3.reset_index(inplace=True, drop=True)
    
    
    # Check that the shapes of the three arrays are identical:
    if lagged1.shape==lagged2.shape and lagged2.shape==lagged3.shape:
        print('Lagged arrays all have shape',lagged1.shape)
    else:
        print('Lagged arrays dont have the same shape :( ')
        print(lagged1.shape,lagged2.shape,lagged3.shape)
        print('Not continuing')
        return

   
    # Separate training from target data - always trying to build the model by
    # predicting the latest season
        
    lastseason = int(max(lagged1['year']))
    
    # predict from the 20152016 season (lag = 1)
    lag1predictfrom = lagged1.loc[lagged1['year'] == lastseason]
    
    # predict from the 20142015 season (lag = 2)
    lag2predictfrom = lagged2.loc[lagged1['year'] == lastseason] # the rows of interest are in the same position as those in lagged1    
    lag3predictfrom = lagged3.loc[lagged1['year'] == lastseason]

    
    # This array is the one that will be predicted from:
    predictarrayfrom = np.transpose(np.dstack((np.array(lag1predictfrom),
                                          np.array(lag2predictfrom),
                                          np.array(lag3predictfrom))), (0,2,1))
    
    # check array shapes
    print('Array for prediction has shape ', predictarrayfrom.shape)
            
    
    #return the arrays for model building and predicting
    return predictarrayfrom





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
    
    # Clear Keras backend, then define the network using the Sequential Keras API
    clear_session()
    model = Sequential()
    
    # Inform algorithm that 0 represents non-values (values of -1 were scaled to 0!)
    model.add(Masking(mask_value=-999, input_shape=(train_ind.shape[1], train_ind.shape[2])))
    
   # Define as LSTM with neurons
    # Neural network will be a triangle. Calculate the layers and neurons per layer:
    while nrons > 2:
        print("adding ", nrons, "neurons")
        model.add(LSTM(nrons,return_sequences=True))
        nrons = math.ceil(nrons/2)
    model.add(LSTM(2))
    
    # I'm not even sure why I need this part, but it doesn't work without it...
    model.add(Dense(train_ind.shape[1]))
    
    # Define a loss function and the Adam optimization algorithm
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    # train network
    history = model.fit(train_ind, train_resp, epochs=epchs, batch_size=bsize, validation_data=(test_ind, test_resp),verbose=0, shuffle=False)

    # plot history
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()
    
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


def act_pred_basic(predict_from, result):
    """
    A function that creates a simple actual vs predicted plot to visually
    inspect the results.
    
    predict_from = array to be predicted from, as generated by LSTM_basic. This
                    should include the actual results.
                    
    predicted = array of predicted results, as returned by modelrun. Currently
                    expects the result of a single iteration. Could be extended
                    to deal with sets of results
                    
    
    
    
    
    """
    
    # Extend the dimension of result to mimic structure if there were multiple
    # realizations.
    if len(result.shape) < 3:
        result = np.expand_dims(result,2)
    
    # Retrieve the responding variables for predictarrayfrom
    actual = predict_from[:,0,-1]
    
    # Find the mask
    resultmask = np.ma.masked_less(result,1).mask
    
    # result.shape = [player, lag, iteration]
    
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

def act_pred_mean(predict_from, result):
    """
    A function that creates a simple actual vs mean predicted plot to visually
    inspect the results.
    
    predict_from = array to be predicted from, as generated by LSTM_basic. This
                    should include the actual results.
                    
    predicted = array of predicted results, as returned by iterations of
                    modelrun.    
    
    
    """
    
    # Retrieve the responding variables for predictarrayfrom
    actual = predict_from[:,0,-1]
    
    # Find the mask
    resultmask = np.ma.masked_less(result,1).mask
    
    # result.shape = [player, lag, iteration]
    
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
        

def act_pred_probabilistic(predict_from, result):
    """
    act_pred_probabilistic compares actual to predicted results, demonstrating
    the a number of predictions by showing the P10 and P90 results as error bars.
    
    predict_from = array to be predicted from, as generated by LSTM_basic. This
                    should include the actual results.
                    
    predicted = array of probabilistic results, as returned by result_prob.
    
    """
    
    probabilistic_result = result_probs(result)
    
    # create error array that contains the y error that we're representing.
    # First row is error below, second is error above
    
    err_arr = np.zeros((probabilistic_result.shape[0],2))
    err_arr[:,0] = probabilistic_result[:,1]-probabilistic_result[:,0]
    err_arr[:,1] = probabilistic_result[:,2]-probabilistic_result[:,1]
    err_arr = np.transpose(err_arr)
    
    
    # Color the points by their success in predicting production within error

    # Create a 1D array indicating whether a player's performance matched their
    # the prediction within error
    test_acc = np.full((predict_from.shape[0]), False)
    for i in range(predict_from.shape[0]):
        if probabilistic_result[i,0] <= predict_from[i,0,-1] and probabilistic_result[i,2] >= predict_from[i,0,-1]:
            test_acc[i] = True

    # plot a error bars
    plt.figure(figsize=(5,5))
    
    ax = plt.subplot(1,1,1)
    ax.plot([0,50,120],[0,50,120])
    
    # plot all error bars in red, then all P50 points as red
    ax.errorbar(predict_from[:,0,-1],probabilistic_result[:,1], yerr=err_arr, fmt='|', capsize=1, linewidth=0.5,c='r')
    ax.scatter(predict_from[:,0,-1],probabilistic_result[:,1],c="r",s=10)
    
    # now overwrite correct error bars in green, and all correct P50 points as green
    ax.errorbar(predict_from[:,0,-1][test_acc],probabilistic_result[:,1][test_acc], yerr=err_arr.T[test_acc].T, fmt='|', capsize=1, linewidth=0.5,c='g')
    ax.scatter(predict_from[:,0,-1][test_acc],probabilistic_result[:,1][test_acc],c="g",s=10)

    ax.set_xlim(-5,110)
    ax.set_ylim(-5,110)
    
    ax.set_aspect('equal')
    ax.grid(True,linestyle='--')
    ax.annotate(np.round(len(np.where(test_acc==True)[0])/len(test_acc)*100,1), xy=(15, 5))
    ax.annotate( '% of predictions within error', xy=(25, 5))
    plt.yticks(range(0,120,20), fontsize=8)
    plt.xticks(range(0,120,20), fontsize=8)
    plt.xlabel('Actual Results')
    plt.ylabel('Predicted Results')
    plt.title('Probabilistic Prediction\nP10 P50 P90', fontsize=10)
    
    # Add the RMSE to plot
    # Find the mask
    resultmask = np.ma.masked_less(result,1).mask
    
    # result.shape = [player, lag, iteration]
    
    # Create an alternate measure of error: use mean of the lags for each player
    # as the prediction. Calculate the RMSE of these means.         
    RMSEmeans =np.empty((result.shape[2]))
    
    meanresult = np.zeros((result.shape[0],result.shape[2]))
    
    for iteration in range(result.shape[2]):
        for player in range(result.shape[0]):
            meanresult[player,iteration] = np.mean(result[player,:,iteration][np.ma.masked_greater(result[player,:,iteration],2).mask])
        
        RMSEmeans[iteration] = np.sqrt(mean_squared_error(meanresult[:,iteration],predict_from[:,0,-1]))
    
    error = np.mean(RMSEmeans)
    
    plt.text(10,85,str('RMSE = '+str(round(float(error),2))),fontsize=16)    
    plt.show()


def result_probs(result):
    """
    result_probs takes an array of prediction interation results and determines
    the P10 P50 and P90 preditions for each player.
    
    result: dimension 1 is the player; dimension 2 is the lag; dimensiion
                    3 is the iteration
                    
    """
    
    # I need to mask the predicitons for nan rows...
    
    out = np.zeros((result.shape[0],3))

    for player in range(result.shape[0]):
        out[player] = np.percentile(result[player,:,:][np.ma.masked_greater(result[player,:,:],2).mask],[10,50,90])
    
    return out







def unbias_model_set_points(model_from):
    """
    unbias_model_set selects a relatively unbiased set of results to model from
    
    model_from is generated by arrayLSTM.
    
    
    * I didn't find this made a difference *
    
    
    """
    
    # Select a subset of data from the first slice, then use indicies to select
    # the rest of the data
    
    #bin_size = 10
    target_count = 50
    
    # Let's target 50 samples per bin of 10 points
    # I'm being "clever" here by mixing a few steps together
    out_indx = np.random.choice(np.where(np.logical_and(0 < model_from[:,0,6], model_from[:,0,6] <= 10))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(10 < model_from[:,0,6], model_from[:,0,6] <= 20))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(20 < model_from[:,0,6], model_from[:,0,6] <= 30))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(30 < model_from[:,0,6], model_from[:,0,6] <= 40))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(40 < model_from[:,0,6], model_from[:,0,6] <= 50))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(50 < model_from[:,0,6], model_from[:,0,6] <= 60))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(60 < model_from[:,0,6], model_from[:,0,6] <= 70))[0], target_count, replace=False).tolist()
    out_indx += np.random.choice(np.where(np.logical_and(70 < model_from[:,0,6], model_from[:,0,6] <= 80))[0], target_count, replace=False).tolist()
    out_indx += np.where(80 < model_from[:,0,6])[0].tolist()
    
    out = model_from[out_indx,:,:]
    
    return out


#%%
    
def act_pred_mean_by_lag(predict_from, result):
    """
    A function that creates a simple actual vs mean predicted plot to visually
    inspect the results. Presents a separate plot for each lag.
    
    predict_from = array to be predicted from, as generated by LSTM_basic. This
                    should include the actual results.
                    
    predicted = array of predicted results, as returned by iterations of
                    modelrun.    
    
    
    """
    
    # Retrieve the responding variables for predictarrayfrom
    actual = predict_from[:,0,-1]
    
    # Find the mask
    resultmask = np.ma.masked_less(result,2).mask
    
    # result.shape = [player, lag, iteration]
    
    # Create an alternate measure of error: use mean of the lags for each player
    # as the prediction. Calculate the RMSE of these means.         
    RMSEmeans =np.empty((result.shape[1]))
    
    meanresult = np.zeros((result.shape[0],result.shape[1]))
    
    for lag in range(result.shape[1]):
        for player in range(result.shape[0]):
            meanresult[player,lag] = np.mean(result[player,lag,:][np.ma.masked_greater(result[player,lag,:],2).mask])
        
        RMSEmeans[lag] = np.sqrt(mean_squared_error(actual[~resultmask[:,lag,-1]],meanresult[:,lag][~resultmask[:,lag,-1]]))
    
    # For convenience, capture these errors in a single array. First columns are
    # the RMSEs for each lag, followed by the mean of errors for all lags,
    # then the error of the mean estimates for all lags.
    #RMSEall = np.concatenate((RMSEs,np.expand_dims(np.mean(RMSEs,axis=0),axis=1).T,np.expand_dims(RMSEmeans,axis=1).T),axis=0)
    
    # For now, I think the best representation of the error is the RMSE for
    # the mean of the the lag estimates. Report this as error.
    #error = np.mean(RMSEmeans)
    
    fig2 = plt.figure(figsize=(15,5))
    ax = fig2.add_subplot(1,3,1)
    #ax.scatter(actual,meanresult[0],c="b", s=10)
    ax.scatter(actual,np.mean(result[:,0,:][~resultmask[:,0,0]], axis=1),c="b", s=12)
    ax.plot([0,50,120],[0,50,120])
    plt.ylim(-5,110)
    plt.xlim(-5,110)
    plt.xlabel('Actual Results')
    plt.ylabel('Predicted Results')
    plt.title('Actual vs. Predicted, lag0', fontsize=16)
    plt.grid(True)
    plt.text(10,85,str('RMSE = '+str(round(RMSEmeans[0],2))),fontsize=12)
    
    ay = fig2.add_subplot(1,3,2)
    ay.scatter(actual[~resultmask[:,1,0]],np.mean(result[:,1,:][~resultmask[:,1,0]], axis=1),c="r", s=12)
    ay.plot([0,50,120],[0,50,120])
    plt.ylim(-5,110)
    plt.xlim(-5,110)
    plt.xlabel('Actual Results')
    plt.ylabel('Predicted Results')
    plt.title('Actual vs. Predicted, lag1', fontsize=16)
    plt.grid(True)
    plt.text(10,85,str('RMSE = '+str(round(RMSEmeans[1],2))),fontsize=12)    
    az = fig2.add_subplot(1,3,3)
    az.scatter(actual[~resultmask[:,2,0]],np.mean(result[:,2,:][~resultmask[:,2,0]], axis=1),c="g", s=12)
    az.plot([0,50,120],[0,50,120])
    plt.ylim(-5,110)
    plt.xlim(-5,110)
    plt.xlabel('Actual Results')
    plt.ylabel('Predicted Results')
    plt.title('Actual vs. Predicted, lag2', fontsize=16)
    plt.grid(True)
    plt.text(10,85,str('RMSE = '+str(round(RMSEmeans[2],2))),fontsize=12)    
    #np.save('./results/LAG3_POINTS50/LSTM8-MSE_ADAM-epo64_batch25.npy',result)












