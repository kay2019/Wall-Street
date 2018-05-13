#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:25:20 2018

@author: kiyoumars
"""
import datetime
import quandl
import pandas_datareader.data as web
import pandas_datareader as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

def obtain_data(start_date,end_date):
    """
    Bank of America Corp	86.81	$175.81
    Apple Inc.	54.36	$760.72
    General Electric Co	42.57	$277.29
    Microsoft Corp	38.35	$386.26
    Intel Corp	30.43	$159.92
    Cisco Systems, Inc.	28.59	$149.07
    AT&T Inc.	28.07	$181.50
    Pfizer Inc.	27.74	$211.21
    Ford Motor Co	26.58	$60.90
    Facebook, Inc.
    """
#    stocks=['BAC','GE','T','PFE','F','TWTR','GNC','UAA','VIPS','SNAP',
#            'UA','WFT','APRN','FCX','CHK','VALE','BABA','SWN','TEVA','WFC',
#            'AKS','ESV','RGC','NOK','X','ABC','CSRA','CLF','KO','CX']

    #stocks=['BAC','GE']
    #mydata = quandl.get(["NSE/OIL.1", "WIKI/AAPL.4"])
    #mydata = quandl.get("FRED/GDP", start_date="2001-12-31", end_date="2005-12-31")
   
    
    i=0
    for s in stocks:
        stk=pdr.get_data_yahoo(s,start_date,end_date)
        stk[i]=stk["Adj Close"]
        i=i+1
    



    plt.show
    return stk





if __name__=="__main__":
    start=datetime.datetime(2016,1,1)
    end=datetime.datetime(2018,1,1)
    #BAC=obtain_data(start_date,end_date)
    
    stocks=['BAC','GE','T','PFE','F','HPE','GNC','UAA','VZ','CHGG',
            'RAD','FCX','CHK','SWN','WFC',
            'AKS','ESV','RGC','X','ABC','CSRA','CLF','KO','TWTR',
            'GM','PG','NE','S','C','JCP']
    stocks=['GOOGL','AAPL','MSFT']
    #stk=web.DataReader('SNAP', 'quandl', start_date, end_date)

    converted_obj3 = pd.DataFrame()
    #mydata = quandl.get("BAC",start_date="2001-12-31", end_date="2005-12-31")
    
    sleeptime=2

#    
    for s in stocks:
        sleep(sleeptime)
        #stk=web.DataReader(s, 'quandl', start_date, end_date)
        stk = quandl.get("WIKI/" + s, start_date=start, end_date=end)
        print(s,'done!') 
        converted_obj3.loc[:,s]=stk["Close"]
             
#        d = {'NPI': NPI, 'Credential':Credential}
#    df2 = pd.DataFrame(data=d)
#        df=pd.DataFrame(stk["Adj Close"])
        