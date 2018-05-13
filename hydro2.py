#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:21:32 2018

@author: kiyoumars
"""



import csv
import pandas as pd
from sklearn import linear_model
from scipy import stats
import numpy as np
import math

def file_reader():

    stocks=['BAC.csv','GE.csv','T.csv','PFE.csv','F.csv','HPE.csv','GNC.csv','UAA.csv','VZ.csv','CHGG.csv',
            'RAD.csv','FCX.csv','CHK.csv','SWN.csv','WFC.csv',
            'AKS.csv','ESV.csv','RGC.csv','X.csv','ABC.csv','CSRA.csv','CLF.csv','KO.csv','TWTR.csv',
            'GM.csv','PG.csv','NE.csv','S.csv','C.csv','JCP.csv']
    stocks=['BAC.csv','GE.csv']
    AllPrices = pd.DataFrame()
    for file_name in stocks:    
        with open(file_name) as csvfile:
            reader=csv.DictReader(csvfile)
          
            close=[]
      
            for row in reader:        
                temp_close=row['close']
                close.append(temp_close)
        d = {file_name: close}
        df1 = pd.DataFrame(data=d)
        AllPrices.loc[:,file_name[:-4]]=df1

    return df1,AllPrices
if __name__=="__main__":
    #df1,AllPrices=file_reader()
    stocks=['BAC.csv','GE.csv','T.csv','PFE.csv','F.csv','HPE.csv','GNC.csv','UAA.csv','VZ.csv','CHGG.csv',
            'RAD.csv','FCX.csv','CHK.csv','WFC.csv',
            'AKS.csv','RGC.csv','X.csv','ABC.csv','CSRA.csv','CLF.csv','KO.csv','TWTR.csv',
            'GM.csv','PG.csv','S.csv','C.csv','JCP.csv','GOOGL.csv','AAPL.csv','MSFT.csv']
    
    stocks=['BAC.csv','GE.csv','T.csv','PFE.csv','F.csv','HPE.csv','GNC.csv','UAA.csv','VZ.csv','CHGG.csv',
            'RAD.csv','FCX.csv','CHK.csv','WFC.csv',
            'AKS.csv','RGC.csv','X.csv','ABC.csv','CSRA.csv','CLF.csv','KO.csv','TWTR.csv',
            'GM.csv','PG.csv','S.csv','C.csv','JCP.csv','GOOGL.csv','MSFT.csv','AAPL.csv']
    
    AllPrices = pd.DataFrame(index=range(502))

                     
    for file_name in stocks:    
        with open(file_name) as csvfile:
            reader=csv.DictReader(csvfile)
          
            close=[]
      
            for row in reader:        
                temp_close=row['close']
                close.append(temp_close)

        AllPrices.loc[:,file_name[:-4]]=close