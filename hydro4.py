#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:43:16 2018

@author: kiyoumars
"""


import pandas as pd
from sklearn import linear_model
from scipy import stats
import numpy as np 
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf





        
        
def deep_velocity(x_pos,c_temp,eta_temp,u_T,t_ini):
 

    tf.reset_default_graph()
# setting of the problem
    d = 20
    T = 0.3
    Xi = np.ones([1,d])*x_pos

# setup of algorithm and implementation
    N = 20
    h = T/N
    sqrth = np.sqrt(h)
    n_maxstep = 10000
    batch_size = 1
    gamma = 0.001

# neural net architectures
    n_neuronForGamma = [d,d,d,d**2]
    n_neuronForA = [d,d,d,d]
    
    # ( adapted ) rhs of the pde
    def f0(t,X,Y,Z,Gamma):
        return -c_temp*Z-eta_temp*Gamma

# terminal condition
    def g(X):
        return u_T(X)

# helper functions for constructing the neural net ( s )
    def _one_time_net (x , name , isgamma = False ):
        with tf.variable_scope(name):
            layer1 = _one_layer(x,(1-isgamma)*n_neuronForA[1]+isgamma*n_neuronForGamma[1],name = 'layer1')
            layer2 = _one_layer(layer1,(1-isgamma)*n_neuronForA[2]+isgamma*n_neuronForGamma[2],name = 'layer2')
            z = _one_layer(layer2,(1-isgamma)*n_neuronForA[3]+isgamma*n_neuronForGamma[3],
                       activation_fn = None,name ='final')
            return z

    def _one_layer(input_ ,output_size,activation_fn=tf.nn.relu,stddev =5.0,name='linear'):
        with tf.variable_scope(name):
            shape =input_.get_shape().as_list()
            w = tf.get_variable('Matrix',[shape[1],output_size],tf.float64,
                               tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+output_size)))
            b =tf.get_variable('Bias',[1,output_size],tf.float64,tf.constant_initializer(0.0))
            hidden = tf.matmul(input_,w)+b
            if activation_fn :
                return activation_fn ( hidden )
            else :
                return hidden
    
    
    
    with tf.Session() as sess :
# background dynamics
        dW=tf.random_normal(shape=[batch_size,d],stddev=sqrth,dtype=tf.float64)    
# initial values of the stochastic processes
        X=tf.Variable(np.ones([batch_size,d])*Xi, dtype=tf.float64,name ='X',trainable =False)
        Y0=tf.Variable(tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float64),name='Y0')
        Z0=tf.Variable(tf.random_uniform([1,d],minval=-.1,maxval=.1,dtype=tf.float64),name='Z0')
        Gamma0=tf.Variable(tf.random_uniform([d,d],minval=-.1,maxval=.1,dtype=tf.float64),name='Gamma0')
        A0 = tf.Variable(tf.random_uniform ([1,d],minval=-.1,maxval=.1,dtype =tf.float64),name ='A0')
        allones =tf.ones(shape=[batch_size,1],dtype = tf.float64,name ='MatrixOfOnes')
        Y = allones*Y0
        Z = tf.matmul(allones,Z0)
        Gamma =tf.multiply(tf.ones([batch_size,d,d],dtype=tf.float64),Gamma0)
        A =tf.matmul(allones,A0)

# forward discretization
        with tf.variable_scope('forward'):
            for i in range(N-1):
                Y =Y+(f0(i*h,X,Y,Z,Gamma)+eta_temp*tf.trace(Gamma))*h+tf.reduce_sum(dW*Z,1,keep_dims=True)
                Z =Z+A*h+tf.squeeze(tf.matmul(Gamma,tf.expand_dims(dW,-1)))
                Gamma = tf.reshape(_one_time_net(X,name = str(i)+'Gamma',isgamma = True)/d **2,[batch_size,d,d])
                if i != N-1:
                    A =_one_time_net(X,name=str(i)+'A')/d
                X = X + dW
                dW = tf.random_normal(shape=[batch_size,d],stddev=sqrth,dtype=tf.float64)

                Y = Y+(f0((N-1)*h,X,Y,Z,Gamma)+eta_temp*tf.trace(Gamma))*h+tf.reduce_sum(dW*Z,1,keep_dims = True )
                X = X+dW
                loss_function = tf.reduce_mean(tf.square(Y-g(X)))
# specifying the optimizer
        global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),
                                    trainable = False,dtype = tf.int32)

        learning_rate = tf.train.exponential_decay(gamma,global_step,
                                decay_steps = 10000,decay_rate = 0.0,staircase = True )

        trainable_variables= tf.trainable_variables()
        grads = tf.gradients(loss_function,trainable_variables)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate )
        apply_op=optimizer.apply_gradients(zip(grads,trainable_variables),global_step=global_step,name ='train_step')

        with tf.control_dependencies([apply_op]):
            train_op_2=tf.identity(loss_function , name = 'train_op2')

# the actual training loop
    for _ in range(n_maxstep+1):
        y0_value , step = sess.run([Y0,global_step])
        currentLoss, currentLearningRate=sess.run([train_op_2,learning_rate])
        if step % 100 == 0:
            print ("step: ",step,"loss:",currentLoss,"Y0:",y0_value,"learning rate:",currentLearningRate)
        
    return y0_value


def NormalD(t,r_temp,v_temp,X,Mcov,DetMcov):
#X[t,:,:] = np.stack((R1,V1), axis=0)
#Mcov[t,:,:]=np.cov(X[t,:,:])
#DetMcov[t]=np.linalg.det(Mcov[t,:,:]) 
    mu1=np.nanmean(X[t,0,:])
    mu2=np.nanmean(X[t,1,:])
    sigma11=Mcov[t,0,0]
    sigma12=Mcov[t,0,1]
    sigma22=Mcov[t,1,1]
    f= (1/(2*math.pi))*(1/math.sqrt(DetMcov[t])) \
            * math.exp(-0.5*((v_temp-mu2)*(v_temp*sigma11-mu2*sigma11 \
                             +2*(-r_temp+mu1)*sigma12)+ sigma22*(r_temp-mu1)**2))    
    return f 

def rho(X,Mcov,DetMcov):#rho(t,x) sum over all v
    
    integral2D=np.empty((num_row,num_col))
    integral2D.fill(np.nan)  
    k=0
    temp=np.empty((num_col))
    
    for t in range(tau[k],num_row):
        for i in range(num_col):#for x
            for j in range(num_col):#for v
            #jacobi=
                temp[j]=m[j]*NormalD(t,X[t,0,i],X[t,1,j],X,Mcov,DetMcov)
            integral2D[t,i]=np.trapz(temp, x=X[t,1,:])
    return integral2D

def velocity(X,Mcov,DetMcov):#u(t,x) sum over all v
    
    integral2D=np.empty((num_row,num_col))
    integral2D.fill(np.nan) 
    k=0
    temp=np.empty((num_col))
    
    for t in range(tau[k],num_row):
        for i in range(num_col):#for x
            for j in range(num_col):#for v
            #jacobi=
                temp[j]=X[t,1,j]*m[j]*NormalD(t,X[t,0,i],X[t,1,j],X,Mcov,DetMcov)
            integral2D[t,i]=np.trapz(temp, x=X[t,1,:])
    return np.divide(integral2D,rho(X,Mcov,DetMcov))

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
    


    for t in range(num_row):
        for i in range(num_col):
            for j in range(num_col):
                s[t,i,j]=p[t,i]/p[t,j]
    


    #print(np.count_nonzero(np.isnan(s)) )   
    """ let's choose days=1, week=5, month=20"""
    tau=[1,5,20]
    H=1
    d=np.empty((num_row-1,num_col,num_col))
     
   
    for t in range(0,num_row-1):
        for i in range(num_col):
            for j in range(num_col):
                d[t,i,j]= math.log(s[t+1,i,j]/s[t,i,j])
    
    print(np.count_nonzero(np.isnan(d)) )
    x=np.empty((num_row-1,num_col))

    

    for t in range(0,num_row-1):
        for i in range(num_col):
            x[t,i]=np.mean(d[t,i,:])
    
    sigma=np.empty((H,num_row))
    mid=np.empty((H,num_row))
    sigma.fill(np.nan)
    mid.fill(np.nan)
    
    for k in range(1):
        for t in range(num_row):
                    sigma[k,t]=np.nanstd(x[k,t,:])
                    mid[k,t]=np.nanmean(x[k,t,:])
                    
    """ starting time should be t=20 """
    
    r=np.empty((H,num_row,num_col))
    r.fill(np.nan)
    
    for k in range(1):
        for t in range(tau[k],num_row):
            for i in range(num_col):
                    r[k,t,i]=x[k,t,i]/abs(sigma[k,t])
                    
                    
   
    v=np.empty((H,num_row,num_col))
    v.fill(np.nan)
    
    for k in range(1):
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
    
#    for i in range(num_col):
#        for k in range(1):
            #print("Direction ",k,"average speed",np.nanmean(v[k,:,i]),"for particle",i)
    """ Maximum for average speed is around <v>~0.006 """
    
    

    DetMcov=np.empty((num_row))    
    DetMcov.fill(np.nan)

    Mcov=np.empty((num_row,2,2))    
    Mcov.fill(np.nan)  
    
    X=np.empty((num_row,2,num_col))    
    X.fill(np.nan)  
    
    
    for t in range(tau[k],num_row):
        
        
        R1=np.array(r[0,t,:])
        #R2=np.array(r[1,t,:])
        #R3=np.array(r[2,t,:])        
        V1=np.array(v[0,t,:])
        #V2=np.array(v[1,t,:])
        #V3=np.array(v[2,t,:])        
        #R=R.ravel()
        #V=V.ravel()
        X[t,:,:] = np.stack((R1,V1), axis=0)
        Mcov[t,:,:]=np.cov(X[t,:,:])
        DetMcov[t]=np.linalg.det(Mcov[t,:,:])
        
    #u=np.empty((num_row,num_col))   
    #u=velocity(X,Mcov,DetMcov) 
#    
#    c = np.linspace(0.0001, 1, 100)
#    eta = np.linspace(0.0001, 1, 100)
#    T=num_row
#    t_ini=num_row-100
#    x_pos=X[t_ini,0,30]
#    x_data=X[T,0,:]
#    u_data=u[T,:]
#    u_fit = np.polyfit(x_data,u_data, 10)
#    u_T = np.poly1d(u_fit)
#    
#    for c_temp in range(c):
#        for eta_temp in range(eta):
#            if abs(u(t_ini,x_pos)-deep_velocity(x_pos,c_temp,eta_temp,u_T,t_ini))<10:
#                print(c_temp,eta_temp)