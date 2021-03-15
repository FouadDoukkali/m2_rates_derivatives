#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:33:58 2019

@author: FouadDoukkali, P.H.Guittard, A.M.Raohilisonn, T.Chazan
"""

import numpy as np
import pandas as pd
from scipy.optimize import broyden1 as o1
import time 
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
def dbrownian(n):  
    elements = [ -1, 1]
    probabilities = [0.5, 0.5]   
    stock= np.random.choice(elements, n, probabilities)
    rw = (1/math.sqrt(n))*np.cumsum(stock)    
    time = np.linspace(0,1,n)
    return time,rw


t = 0
for _ in range(1000):
    
    start = time.time()
    dbrownian(100000)
    end = time.time()
    t = t + (end - start)
    
print('The Donsker brownian Function simulates a sample of size n=100000 on an average of {:.6f} seconds'.format(t/1000))
# In[3]:

def box_muller(n=1):

    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
   
    r = np.sqrt(-2*np.log(u1))
    x = np.cos(2*np.pi*u2)
    y = np.sin(2*np.pi*u2)
    z1 = r*x
    z2 = r*y
   
    return z1, z2

import time
t = 0
for _ in range(1000):
    
    start = time.time()
    box_muller(100000)
    end = time.time()
    t = t + (end - start)
    
print('The Box-Muller Function simulates a sample of size n=100000 on an average of {:.6f} seconds'.format(t/1000))

def marsaglia_array(n=1):
    
    def marsaglia():
    
        while True:

            w1 = (np.random.rand() * 2) - 1
            w2 = (np.random.rand() * 2) - 1

            if w1 * w1 + w2 * w2 < 1:

                return w1, w2
    
    w1 = np.empty(n)
    w2 = np.empty(n)
    
    for i in range(n):
        
        w1[i], w2[i] = marsaglia()
        
    s = w1*w1 + w2*w2
    t = np.sqrt(-2*np.divide(np.log(s), s))
    z1 = w1*t
    z2 = w2*t
   
    return z1, z2
t = 0
# for i in range(1000):
#     start = time.time()
#     marsaglia_array(100000)
#     end = time.time()
#     t = t + (end - start)
    
# print('The Marsaglia-Bray Function simulates a sample of size n=100000 on an average of {:.6f} seconds'.format(t/1000))

#We get 2 normal arrays of size n      
def brownian_marsaglia(n, t0 = 0, t1 = 1):
    s = marsaglia_array(n)
    t = np.linspace(t0, t1,n)
    b1 = np.cumsum(np.sqrt(np.diff(t))*s[0][1:])
    b2 = np.cumsum(np.sqrt(np.diff(t))*s[1][1:])
    return b1, b2
def brownian_boxm(n, t0 = 0, t1 =1):
    s = box_muller(n)
    t= np.linspace(t0, t1,n)
    b1 = np.append(np.array(0),np.cumsum(np.sqrt(np.diff(t))*s[0][1:]))
    b2 = np.append(np.array(0),np.cumsum(np.sqrt(np.diff(t))*s[1][1:]))
    return b1, b2

t = 0
for i in range(1000):
    start = time.time()
    brownian_boxm(100000)
    end = time.time()
    t = t + (end - start)

print('The Box-Muller brownian Function simulates a sample of size n=100000 on an average of {:.6f} seconds'.format(t/1000))
# In[3]:


def drifted_bm(n, x, mu, sigma):
    trw = dbrownian(n)
    drifted =  mu*trw[0] + sigma*trw[1] + x  
    return drifted

def visbm(d, n, x, mu=0, sigma=1, t0 = 0, t1 = 1):
    if d==1:
        plt.plot(np.linspace(t0, t1, n-1), sigma*brownian_boxm(n)[0])
        plt.title('Box-Muller Brownian motion representation, %i steps' %n)
    if d==2:
        b = brownian_boxm(n)
        plt.plot(sigma*b[0]+mu*np.linspace(t0, t1, n-1), sigma*b[1]+mu*np.linspace(t0, t1, n-1))
        
        plt.plot(b[0][0],b[1][0], 'go') #begin
        plt.plot(b[0][-1], b[1][-1], 'ro') #end
        plt.title('Box-Muller 2-brownian motion representation, %i steps \n Dots are to perceive the shift and drift' %n)
        plt.xlabel('B1')
        plt.ylabel('B2')
        plt.show()
    if d==3:
        from mpl_toolkits.mplot3d import Axes3D
        b = brownian_boxm(n)
        b1=  sigma*b[0]+mu*np.linspace(t0, t1, n-1)
        b2 = sigma*b[1]+mu*np.linspace(t0, t1, n-1)
        b3 = sigma*brownian_boxm(n)[0]+mu*np.linspace(t0, t1, n-1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')    
        ax.scatter3D(b1,b2,b3, c = b3, cmap='Blues', depthshade=True)
        ax.set_xlabel('B1')
        ax.set_ylabel('B2')
        ax.set_zlabel('B3')
        ax.set_title('3D Brownian Motion Box-Muller %d steps'%n)
    plt.tight_layout()
    plt.show()
    return  
        
a=visbm(d = 3,n = 10000, x =  0, mu = 0, sigma = 1)

#Here in order to simplify the following formula a=alpha, and b=mu



#========================ZC==================================
def M(t,T,b,a,rt): #with rt the rate at t 
    return b*(T-t)+(rt-b)*(1-np.exp(-a*(T-t)))/a
def V(t,T,sig0car,sigcar,a):
    return ((sig0car/a**2)*(1-np.exp(-a*(T-t)))**2)+(sigcar/a**2)*((T-t)+((1-np.exp(-2*a*(T-t)))/2*a)-(2*(1-np.exp(-a*(T-t)))/a))

def B(t,T,a,b,sig0car,sigcar,rt):
    return np.exp(-M(t,T,b,a,rt)+(1/2)*V(t,T,sig0car,sigcar,a))

#x=[a,b,sig0car,sigcar,r0]
#===================================Parameters Computations=================    
def calc(S1,S2,S3,S4,S5):
    B1=1/(S1+1)
    B2=(1/(S2+1))-S2*B1/(S2+1)
    B3=(1/(S3+1))-(S3*B2/(S3+1))-(S3*B1/(S3+1))
    B4=(1/(S4+1))-(S4*B3/(S4+1))-(S4*B2/(S4+1))-(S4*B1/(S4+1))
    B5=(1/(S5+1))-(S5*B4/(S5+1))-(S5*B3/(S5+1))-(S5*B2/(S5+1))-(S5*B1/(S5+1))
    return ([round(B1,2),round(B2,2),round(B3,2),round(B4,2),round(B5,2)])
##EUSWAP rate Paramters in calc
def F(x):
    return np.array([B(0,1,x[0],x[1],x[2],x[3],x[4]),B(2,x[0],x[1],x[2],x[3],x[4]),B(0,3,x[0],x[1],x[2],x[3],x[4]),B(0,4,x[0],x[1],x[2],x[3],x[4]),B(0,5,x[0],x[1],x[2],x[3],x[4])])-calc(-0.0032,-0.0030,-0.0026,-0.0021,-0.0015)

def testconv(tol):
   
    a0=np.random.uniform(0.02,1.1)
    b0=np.random.uniform(0.02,1.5)
    r0=np.random.uniform(-0.03,0.2)
    sig0car=np.random.uniform(0.1,3)
    sigcar=np.random.uniform(0.1,3)
    return [o1(F,[a0,b0,sig0car,sigcar,r0],f_tol=tol),[a0,b0,sig0car,sigcar,r0]]

def find(tol,nb):
    start_time=time.time()
    dic=dict()
    i=0
    a=[]
    b=[]
    sg0=[]
    sg=[]
    r0=[]
    while i<nb:
        try:
            k=testconv(tol)
            if k[0][2]>0 and k[0][3]>0 and k[0][4]<0.10 and k[0][0] >0:
                a.append(k[0][0])
                b.append(k[0][1])
                sg0.append(k[0][2])
                sg.append(k[0][3])
                r0.append(k[0][4])
            i+=1
                
                

        except:
            print(i)
            i+=1
    dic['a']=a
    dic['b']=b
    dic['sig0car']=sg0
    dic['sigcar']=sg
    dic['r0']=r0
    print("Temps d'exécution de : %s"%(time.time()-start_time))
    return pd.DataFrame(dic)
#df_new=find(0.003,10000)
#df_new.to_excel('financeSwapEU10000.xlsx',sheet_name='Feuille1')
    

#========Reading parameters from pandas===========
#Lecture file à mettre au même endroit que le fichier
#df_EU=pd.read('financeSwapEU.xlsx')
#df_US=pd.read('financeSwapUS.xlsx')
#for i in df_EU.index:
#    alpha=df_EU['alpha'][i]
#    sig=df_EU['sig'][i]
#    sig0car=df_EU['sig0car'][i]
#    sigcar=df_EU['sigcar'][i]
#    r0=df_EU['r0'][i]
#    mu=df_EU['Mu'][i]
#
#for i in df_US.index:
#    alpha=df_US['alpha'][i]
#    sig=df_US['sig'][i]
#    sig0car=df_US['sig0car'][i]
#    sigcar=df_US['sigcar'][i]
#    r0=df_US['r0'][i]
#    mu=df_US['mu'][i]    


#==============Linear Forward Rate==========   
def FowR(t,T1,T2,a,b,sig0car,sigcar,rt):
    return (B(t,T1,a,b,sig0car,sigcar,rt)/(T2-T1)*B(t,T2,a,b,sig0car,sigcar,rt))

#=========Florlet/Caplet========================
def caplet(t,T1,T2,a,b,sig0car,sig,rt,K):          #ici on a SIG pas SIGMA CARRE
    sigcar=sig**2
    Ft=FowR(t,T1,T2,a,b,sig0car,sigcar,rt)
    d1=(np.log(Ft/K)+0.5*(T1-t)*sigcar)/(np.sqrt(T1-t)*sig)
    d2=d1-(np.sqrt(T1-t)*sig)
    Clt=(T2-T1)*B(t,T2,a,b,sig0car,sigcar,rt)*((Ft*norm.cdf(d1)) - (K*norm.cdf(d2)))
    return Clt

def floorlet(t,T1,T2,a,b,sig0car,sig,rt,K):
    sigcar=sig**2
    Ft=FowR(t,T1,T2,a,b,sig0car,sigcar,rt)
    d1=(np.log(Ft/K)+0.5*(T1-t)*sigcar)/(np.sqrt(T1-t)*sig)
    d2=d1-(np.sqrt(T1-t)*sig)
    Flt=(T2-T1)*B(t,T2,a,b,sig0car,sigcar,rt)*((K*norm.cdf(-d2)) - (Ft*norm.cdf(-d1)))
    return Flt
#==============ReadingFinanceSwapUS and convert it to array============
f=pd.read_excel('financeSwapUS.xlsx')
#============EulerScheme=================
def VasicekEul(a,b,sig,r0,T,N):
    r=np.ones(shape = (N,1))*r0
    B = np.diff(brownian_boxm(N+2)[0])
    for i in range(1,N):      
        r[i]=r[i-1]+a*(b-r[i-1])*(T/N)+sig*(B[i-1])
    return r
def plotV(indexpanda,T,N,df):
    k=indexpanda
    Y=VasicekEul(df['alpha'][k],df['mu'][k],df['sig'][k],df['r0'][k],T,N)
    X=np.arange(0,T,T/N)
    return plt.plot(X,Y)
def VasicekMM(a,b,sig,r0,T,N):
    r=VasicekEul(a,b,sig,r0,T,N)
    l=[r0]*(N)
    for i in range(1,N):
        l[i]=sum(r[:i+1])/(i+1)
    return l
def plotVMM(indexpanda,T,N,df):#MM
    k=indexpanda
    Y=VasicekMM(df['alpha'][k],df['mu'][k],df['sig'][k],df['r0'][k],T,N)
    X=np.arange(0,T,T/N)
    return plt.plot(X,Y)
def VasicekLn(a,b,sig,r0,T,N):#Ln(1+abs(MM))
    Y=VasicekEul(a,b,sig,r0,T,N)
    ln=np.log(1+np.absolute(Y))-0.5
    return ln
def plotLn(indexpanda,T,N,df): #Ln(1+abs(MM))
    k=indexpanda
    Y=VasicekLn(df['alpha'][k],df['mu'][k],df['sig'][k],df['r0'][k],T,N)
    X=np.arange(0,T,T/N)
    return plt.plot(X,Y)



#for i in range (0,48): For VMM ploting 
    #plotVMM(i,2,3000)   
#for i in range (0,48):
#plotV(i,2,3000)
#One by one:
#for i in range(48):
    #plt.figure(i)
    
    #plotVMM(i,2,3000)
#Best24-25 for US SWAP
    
#=================ForwardRatePlot =============
def plotFoWSUS(indexpanda,T1,T2,T,N,df):
    k=indexpanda
    Y=[]
    r=VasicekEul(df['alpha'][k],df['mu'][k],df['sig'][k],df['r0'][k],T,N)
    i=0
    while i*(T/N)<T1:
        
        Y.append(FowR(i*(T/N),T1,T2,df['alpha'][k],df['mu'][k],df['sig0car'][k],df['sigcar'][k],r[i]))
        i+=1
        X=np.arange(0,T1,T/N)

    return (plt.plot(X,Y))
#=========Floorlet/Caplet Plots============================
    
def plotCaplet(indexpanda,T1,T2,T,N,K,df):
    l=indexpanda
    Y=[]
    r=VasicekEul(df['alpha'][l],df['mu'][l],df['sig'][l],df['r0'][l],T,N)
    i=0
    while i*(T/N)<T1:
        Y.append(caplet(i*(T/N),T1,T2,df['alpha'][l],df['mu'][l],df['sig0car'][l],df['sig'][l],r[i],K))
        X=np.arange(0,T1,T/N)
        i+=1
    return (plt.plot(X,Y))
def plotFloorlet(indexpanda,T1,T2,T,N,K,df):
    l=indexpanda
    Y=[]
    r=VasicekEul(df['alpha'][l],df['mu'][l],df['sig'][l],df['r0'][l],T,N)
    i=0
    while i*(T/N)<T1:
        Y.append(floorlet(i*(T/N),T1,T2,df['alpha'][l],df['mu'][l],df['sig0car'][l],df['sig'][l],r[i],K))
        X=np.arange(0,T1,T/N)
        i+=1
    return (plt.plot(X,Y))
#==============ZC plot============ 
def plotB(indexpanda,T,N,df):
    l=indexpanda
    Y=[]
    r=VasicekEul(df['alpha'][l],df['mu'][l],df['sig'][l],df['r0'][l],T,N)
    i=0
    while i*(T/N)<T:
        Y.append(B(i*(T/N),T,df['alpha'][l],df['mu'][l],df['sig0car'][l],df['sigcar'][l],r[i]))
        X=np.arange(0,T,T/N)
        i+=1
    return(plt.plot(X,Y))
        
    
    
    
        

        


    


 
 