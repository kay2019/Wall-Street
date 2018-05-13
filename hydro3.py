#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:43:16 2018

@author: kiyoumars
"""

import csv
import pandas as pd
from sklearn import linear_model
from scipy import stats
import numpy as np 
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns


def NormalD(r1,r2,r3,v1,v2,v3,X):
    
    

if __name__=="__main__":
    df = pd.read_csv('AllPrices.csv')
    stocks=['BAC','GE','T','PFE','F','HPE','GNC','UAA','VZ','CHGG',
            'RAD','FCX','CHK','WFC',
            'AKS','RGC','X','ABC','CSRA','CLF','KO','TWTR',
            'GM','PG','S','C','JCP','GOOGL','MSFT','AAPL']
 
    num_row,num_col=df.shape
    timelist=list(df.index)
    p=np.empty((num_row,num_col))

    start_time = time.time()  
    
    for i1, t in enumerate(timelist):
        for i2,i in enumerate(stocks):
                p[i1,i2]=df.loc[t,i]
                
    print("--- %s seconds ---" % (time.time() - start_time))

    s=np.empty((num_row,num_col,num_col))
    s.fill(np.nan)


    for t in range(num_row):
        for i in range(num_col):
            for j in range(num_col):
                s[t,i,j]=p[t,i]/p[t,j]
    


    #print(np.count_nonzero(np.isnan(s)) )   
    """ let's choose days=1, week=5, month=20"""
    tau=[1,5,20]
    H=3
    d=np.empty((H,num_row,num_col,num_col))
    d.fill(np.nan)
    
    for k in range(3):
        for t in range(tau[k],num_row):
            for i in range(num_col):
                for j in range(num_col):
                    d[k,t,i,j]=(1/tau[k]) * math.log(s[t,i,j]/s[t-tau[k],i,j])
    
    print(np.count_nonzero(np.isnan(d)) )
    x=np.empty((H,num_row,num_col))
    x.fill(np.nan)
    
    for k in range(3):
        for t in range(tau[k],num_row):
            for i in range(num_col):
                    x[k,t,i]=np.nanmean(d[k,t,i,:])
    
    sigma=np.empty((H,num_row))
    mid=np.empty((H,num_row))
    sigma.fill(np.nan)
    mid.fill(np.nan)
    
    for k in range(3):
        for t in range(num_row):
                    sigma[k,t]=np.nanstd(x[k,t,:])
                    mid[k,t]=np.nanmean(x[k,t,:])
                    
    """ starting time should be t=20 """
    
    r=np.empty((H,num_row,num_col))
    r.fill(np.nan)
    
    for k in range(3):
        for t in range(tau[k],num_row):
            for i in range(num_col):
                    r[k,t,i]=x[k,t,i]/abs(sigma[k,t])
                    
                    
   
    v=np.empty((H,num_row,num_col))
    v.fill(np.nan)
    
    for k in range(3):
        for t in range(tau[k]+1,num_row):
            for i in range(num_col):
                    v[k,t,i]=(r[k,t,i]-r[k,t-tau[1],i])/tau[1]
                    
    v2=np.empty((num_row,num_col))
    v2.fill(np.nan)
    for t in range(tau[k],num_row):
        for i in range(num_col):
            v2[t,i] =np.dot(v[:,t,i],v[:,t,i])
    
    Tt=np.empty((num_row))
    Tt.fill(np.nan)        
    for t in range(tau[k],num_row):
        Tt[t] =np.nanmean(v2[t,:])/H
    
 
    plt.plot(Tt,marker='o', linestyle='--', color='b', label='Temperature')
    plt.xlabel('Days')
    plt.ylabel('Temperature')
    plt.show()      
    
    T=np.nanmean(Tt[:])
    
    m=np.empty((num_col))
    m.fill(np.nan)        
    for i in range(num_col):
        m[i] =(H*T)/np.nanmean(v2[:,i])
     


    xn = range(len(stocks))
    plt.plot(xn,m.tolist(),marker='o', linestyle='', color='r', label='Masses')
    plt.xticks(xn, stocks, rotation=90)
    plt.xlabel('Stocks')
    plt.ylabel('Masses')
    plt.show() 
    
    for i in range(num_col):
        for k in range(3):
            print("Direction ",k,"average speed",np.nanmean(v[k,:,i]),
            "for particle",i)
    """ Maximum for average speed is around <v>~0.006 """
    
    

    DetMcov=np.empty((num_row))    
    DetMcov.fill(np.nan)

    Mcov=np.empty((num_row,6,6))    
    Mcov.fill(np.nan)  
    
    X=np.empty((num_row,6,num_col))    
    X.fill(np.nan)  
    
    
    for t in range(tau[k],num_row):
        
        
        R1=np.array(r[0,t,:])
        R2=np.array(r[1,t,:])
        R3=np.array(r[2,t,:])        
        V1=np.array(v[0,t,:])
        V2=np.array(v[1,t,:])
        V3=np.array(v[2,t,:])        
        #R=R.ravel()
        #V=V.ravel()
        X[t,:,:] = np.stack((R1,R2,R3,V1,V2,V3), axis=0)
        Mcov[t,:,:]=np.cov(X[t,:,:])
        DetMcov[t]=np.linalg.det(Mcov[t,:,:])
        
       
    