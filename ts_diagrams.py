
##Disclaimer: The code came from http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2018/02/TasksP_03.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats as sm_stat

#Define our tsdisplay function (equivalent to R):
def tsdisplay(y, figsize = (14,8), title = "", max_lag = 20):
    tmp_data = pd.Series(y)
    fig = plt.figure(figsize = figsize)
    #Plot the time series
    tmp_data.plot(ax = fig.add_subplot(211), title = "$Time\ Series\ " + title + "$", legend = False)
    #Plot the ACF:
    sm.graphics.tsa.plot_acf(tmp_data, lags = max_lag, zero = False, ax = fig.add_subplot(223))
    #Plot the PACF:
    sm.graphics.tsa.plot_pacf(tmp_data, lags = max_lag, zero = False, ax = fig.add_subplot(224))
    #Fix the layout of the plots:
    plt.tight_layout()
    
#Define our tsdiag function (equivalent to R):
def tsdiag(y, figsize = (14,8), title = ""):
    #The data:
    tmp_data = pd.Series(y)
    #The standardized data:
    tmp_stand_data = pd.Series(y / np.sqrt(y.var()))
    #The Ljung-Box test results for the first 10 lags:
    p_vals = pd.Series(sm_stat.diagnostic.acorr_ljungbox(tmp_data, lags = 10)[1])
    #Start the index from 1 instead of 0 (because Ljung-Box test is for lag values from 1 to k)
    p_vals.index += 1
    
    fig = plt.figure(figsize = figsize)
    #Plot the time series
    tmp_stand_data.plot(ax = fig.add_subplot(311), title = "$Standardized\ Time\ Series\ " + title + "$", legend = False)
    #Plot the ACF:
    sm.graphics.tsa.plot_acf(tmp_data, lags = 20, zero = False, ax = fig.add_subplot(312))
    #Plot the PACF:
    p_vals.plot(ax = fig.add_subplot(313), linestyle='', marker='o', title = "p-values for Ljung-Box statistic", legend = False)
    #Add the horizontal 0.05 critical value line
    plt.axhline(y = 0.05, color = 'blue', linestyle='--')
    
    # Annotate the p-value points above and to the left of the vertex
    x = np.arange(p_vals.size) + 1
    for X, Y, Z in zip(x, p_vals, p_vals):
        plt.annotate(round(Z, 4), xy=(X,Y), xytext=(-5, 5), ha = 'left', textcoords='offset points')
    
    #Change the range of the axis:
    plt.xlim(xmin = 0)
    plt.xlim(xmax = p_vals.size + 1)
    #Change the X axis tick marks - make them from 1 to the max. number of lags in the L-jung-Box test
    plt.xticks(np.arange(1, p_vals.size + 1, 1.0))
    
    #Fix the layout of the plots:
    plt.tight_layout()