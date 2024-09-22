# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:54:21 2018

@author: Administrator
"""
import sqlite3 as sqlt
import scipy.stats as scs
import scipy as sc
import scipy.optimize as opt
import statsmodels.api as sm
import pandas_datareader as pdrd
import numpy as np
import numexpr as ne
import pandas as pd
import seaborn as sns
from numba import jit
#from numba.decorators import jit,autojit,generated_jit
import matplotlib.pyplot as plt
import random as rnd
import math
#import cmath
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from QuantLib import *
#import autograd.numpy as npa
#from autograd import grad
#import cython

@jit(nopython=True,parallel=True)
def lchfuni(theta,nu,sgm,u):
    denom=np.log((1-1j*u*theta*nu+0.5*((sgm*u)**2)*nu).real)*(-1.0/nu)
    return denom

@jit(nopython=True,parallel=True)
def VGfs(x,theta,sgm,nu,mu,tau,nus,cz):
    #x=cz*x
    theta=theta*cz;sgm=sgm*cz
    #phii=lchfuni(theta,nu,sgm,-1j*cz)*tau
   # fnom=2.0*math.exp(((theta+sgm**2)*x-sgm**2)/(sgm**2))*(((x-mu)**2)/(2.0*sgm**2/nus+theta**2))**(0.5*tau/nu-0.25)
    fnom=2.0*math.exp(((theta)*x)/(sgm**2))*(((x-mu)**2)/(2.0*sgm**2/nus+theta**2))**(0.5*tau/nu-0.25)
    fdenom=nus**(tau/nu)*math.sqrt(math.pi*2.0*sgm**2)*math.gamma(tau/nu)
    xv=math.sqrt((((x-mu)/sgm**2)**2)*(2.0*sgm**2/nus+theta**2))
    return [fnom,fdenom,xv]


def VGdensity(x,theta,sgm,nu,mu,tau,nus,cz):
    #fnom=2.0*math.exp(theta*x/(sgm**2))*(((x-mu)**2)/(2.0*sgm**2/nu+theta**2))**(0.5*tau/nu-0.25)
    #fdenom=nu**(tau/nu)*math.sqrt(math.pi*2.0*sgm**2)*math.gamma(tau/nu)
    #xv=math.sqrt((((x-mu)/sgm**2)**2)*(2.0*sgm**2/nu+theta**2))
    fs=VGfs(x,theta,sgm,nu,mu,tau,nus,cz)
    fnom,fdenom,xv=fs[0],fs[1],fs[2]    
    
    #Inxv=sc.special.kv(tau/nu-0.5,-xv)
    #Ipxv=sc.special.iv(tau/nu-0.5,xv)
    
    if (x==0.0):
        xv=1e-6
        bess=sc.special.kv(tau/nu-0.5,xv)
        #bess=0.0
        #bess=0.5*math.pi*(Inxv-Ipxv)/math.sin(math.pi*(tau/nu-0.5))
    else:
        bess=sc.special.kv(tau/nu-0.5,xv)
    return (fnom/fdenom)*bess


def CDFVG(d,theta,sgm,nu,mu,tau,s,cz):
    
    if s:
        k=chfunk(theta,nu,sgm,-1j)
        nus=nu/k
        thetas=sgm**2+theta
        fvg=lambda x:VGdensity(x,thetas,sgm,nu,mu,tau,nus,cz)
    else:
        fvg=lambda x:VGdensity(x,theta,sgm,nu,mu,tau,nu,cz)
    p=sc.integrate.quad(fvg,-d,np.inf,epsabs=1.49e-06, epsrel=1.49e-06,full_output=1)
    return p


def VGcallBarrierVGD(At,r,L,nu,theta,sgm,tau,q,smp):
    phii=-math.log(1.0-theta*nu-0.5*sgm**2*nu)*(1.0/nu);
    #lchfuni(theta,nu,sgm,-1j)
    Bt=(r-q-phii)*tau
#    if (lm):
#        dL=(math.log(L/At))
#        dR=(math.log(L/At))
#        mu=0.0
#        thetas=theta+lamda*Bt
#        FVG_BL=CDFVG(dL,-thetas,sgm,nu,mu,tau,False)[0]
#        FVG_BR=CDFVG(dR,thetas,sgm,nu,mu,tau,False)[0]
#        PD=(FVG_BL+np.exp(2.0*(thetas)*(math.log(L/At))/sgm**2)*FVG_BR)
    #else:
        
    dL=(math.log(L/At)-Bt)
    dR=(math.log(L/At)-Bt)
        #mu=(math.log(At/L)+Bt)
    FVG_BL=FVGfull(dL,-theta,sgm,nu,tau)[0]
    #
    #CDFVG(dL,-theta,sgm,nu,0.0,tau,False,1.0)[0]
    FVG_BR=FVGfull(dL,theta,sgm,nu,tau)[0]
    #
    #CDFVG(dR,theta,sgm,nu,0.0,tau,False,1.0)[0]
    PD=(FVG_BL+np.exp(2.0*(theta)*(math.log(L/At)-Bt)/sgm**2)*FVG_BR)
        
    if (smp):
        Bt=0.0
        #mu=0.0
        dL=(math.log(L/At)-Bt)
        dR=(math.log(L/At)-Bt)
        FVG_BL=CDFVG(dL,-theta,sgm,nu,0.0,tau,False,1.0)[0]
        #FVGfull(dL,theta,sgm,nu,tau)[0]
        #CDFVG(dL,-theta,sgm,nu,0.0,tau,False,1.0)[0]
        FVG_BR=CDFVG(dR,theta,sgm,nu,0.0,tau,False,1.0)[0]
        #CDFVG(dR,theta,sgm,nu,0.0,tau,False,1.0)[0]
        #FVGfull(dR,theta,sgm,nu,tau)[0]
        
        PD=FVG_BL+np.exp(2.0*(theta)*(np.log(L/At)-Bt)/sgm**2)*FVG_BR
        #PD=FVG_BL+(St/K)*FVG_BR
    #np.exp(2.0*theta*(np.log(K/St)+B)/sgm**2)*FVG_BR
    #min(PD,1.0)  
    return PD
def VGcallSBarrierVGD(At,Dt,r,L,nu,thetaA,thetaD,sgmA,sgmD,rhoAD,tau,q,smp):
    phiiA=-math.log(1.0-thetaA*nu-0.5*sgmA**2*nu)*(1.0/nu);
    phiiD=-math.log(1.0-thetaD*nu-0.5*sgmD**2*nu)*(1.0/nu);
    #phiiA=lchfuni(thetaA,nu,sgmA,-1j)
    #phiiD=lchfuni(thetaD,nu,sgmD,-1j)
    thetaAD=thetaA-thetaD
    sgmAD=math.sqrt(sgmA*sgmA+sgmD*sgmD-2.0*rhoAD*sgmA*sgmD)
    phiiAD=-math.log(1.0-thetaAD*nu-0.5*sgmAD**2*nu)*(1.0/nu);
    B=(phiiD-phiiA-q-phiiAD)*tau
    Lt=At/Dt
    #lchfuni(theta,nu,sgm,-1j)
    dL=(math.log(L/Lt)-B)
    dR=(math.log(L/Lt)-B)
    mu=0.
    FVG_BL=CDFVG(dL,-thetaAD,sgmAD,nu,0.0,tau,False,1.0)[0]
    FVG_BR=CDFVG(dR,thetaAD,sgmAD,nu,0.0,tau,False,1.0)[0]
    PD=FVG_BL+np.exp(2.0*(thetaAD)*(np.log(L/Lt)-B)/sgmAD**2)*FVG_BR
    if (smp):
        B=0.0
        mu=0.0
        dL=(math.log(L/Lt)-B)
        dR=(math.log(L/Lt)-B)
        FVG_BL=CDFVG(dL,-thetaAD,sgmAD,nu,mu,tau,False,1.0)[0]
        FVG_BR=CDFVG(dR,thetaAD,sgmAD,nu,mu,tau,False,1.0)[0]
        PD=FVG_BL+math.exp(2.0*(thetaAD)*(math.log(L/Lt)-B)/sgmAD**2)*FVG_BR
        #PD=FVG_BL+(St/K)*FVG_BR
    #np.exp(2.0*theta*(np.log(K/St)+B)/sgm**2)*FVG_BR
    return min(PD,1.0)    
  
def VGcallfactorBarrierVGD(At,r,L,nuY,nuZ,thetaY,thetaZ,sgmY,sgmZ,a,tau,q,smp):
    #FVG_S=FVG1(St,r,K,nu,theta,sgm,tau,q)[0]
    
    theta=thetaY+a*thetaZ
    sgm=np.sqrt(sgmY**2+(a*sgmZ)**2)
    nu=(nuY*nuZ)/(nuY+nuZ)
    phiiY=lchfuni(thetaY,nuY,sgmY,-1j)
    phiiZ=lchfuni(thetaZ,nuZ,sgmZ,-1j*a)
    phii=phiiY+phiiZ
    B=(r-q-phii)*tau
    dL=(math.log(L/At)-B)
    dR=(math.log(L/At)-B)
    FVG_BL=CDFVG(dL,-theta,sgm,nu,0.0,tau,False,1.0)[0]
    FVG_BR=CDFVG(dR,theta,sgm,nu,0.0,tau,False,1.0)[0]
    #callp=St*FVG_S-np.exp(-r*tau)*K*FVG
    #putp=np.exp(-r*tau)*K*(1.0-FVG)-np.exp(-q*tau)*St*(1.0-FVG_S)
    #IS=(-1.0/tau)*np.log(1.0-putp/(K*np.exp(-r*tau)))
    
    PD=FVG_BL+np.exp(2.0*(theta)*(np.log(L/At)-B)/sgm**2)*FVG_BR
    if (smp):
        B=0
        PD=FVG_BL+np.exp(2.0*(theta)*(np.log(L/At)-B)/sgm**2)*FVG_BR
        #PD=FVG_BL+(St/K)*FVG_BR
    #np.exp(2.0*theta*(np.log(K/St)+B)/sgm**2)*FVG_BR
    return min(PD,1.0)   


import random
@jit(nopython=True,parallel=True)
def simaxBM(nsim,M,tau,mx,r,sgm,At,L):
    
    B=(r-0.5*sgm**2)*tau
    mt=np.zeros(nsim)
    I=np.empty_like(mt)
    
    for i in range(nsim):
        wt=sgm*rnd.normalvariate(0.0,1.0)*math.sqrt(tau)+B             
        U=random.random()
        C=(math.log(U))
        sqdelta=math.sqrt(0.25*wt**2-0.5*C*sgm**2*tau)
        if mx:
            mt[i]=math.exp((0.5*wt+sqdelta))*At
            
        
        else:
            
            mt[i]=math.exp((0.5*wt-sqdelta))*At
            I[i]=(mt[i]<L)
            
    PD=np.sum(I)/nsim
    
    #s=np.std(I)/math.sqrt(nsim)
   
    return PD
#PD=simaxBM(nsim,M,tau,mx,r,sgm,At,L)
#mn=simaxBM(nsim,M,tau,mx,r,sgm,At,L,)
def PDminBM(r,sgm,At,L,tau):
    theta=r-0.5*sgm**2
    m=np.log(L/At)
    P1=scs.norm.cdf((m-theta*tau)/(sgm*np.sqrt(tau)))
    P2=scs.norm.cdf((m+theta*tau)/(sgm*np.sqrt(tau)))
    BC=np.exp(2.0*theta*m/(sgm**2))
    PD=P1+BC*P2
    return PD
def genincgam(b,tau,nu,mx):
    if (mx):
        g=nu**(1.5*tau/nu-1.)*sc.special.kv(tau/nu,2.0*math.sqrt(2.0*m*(m-w)/nu))*(m*(m-w))**(0.5*tau/nu)*2.0\
        **(1.0+0.5*tau/nu)/math.gamma(tau/nu)
    else:
        a=tau/nu
        #nu**(2.0*tau/nu)
        g=(1.0/math.gamma(a))*sc.special.kv(a,2.0*math.sqrt(b))*2.0*(b**(a/2))
        #g=nu**(1.5*tau/nu-1.)*sc.special.kv(tau/nu,2.0*math.sqrt(abs(2.0*m*(w-m)/nu))*(m*(w-m))**(0.5*tau/nu)*2.0\
        #**(1.0+0.5*tau/nu)/math.gamma(tau/nu)
    return g
def gengaminv(b,tau,nu,u):
    gen=genincgam(b,tau,nu,False)-u
    return gen
@jit(nopython=True,parallel=True)
def gampdf(x,tau,nu,nus):
    p=((x**(tau/nu)/x)*math.exp(-x/nus)/math.gamma(tau/nu))*nus**(-tau/nu)
    #p=scs.gamma.pdf(x,tau/nu,0.0,nus)
    return p
@jit(nopython=True,parallel=True)
def gampdfvg(x,tau,nu,nus):
    p=((x**(tau/nu)-0.5)*math.exp(-x/nus)/math.gamma(tau/nu))*nus**(-tau/nu)
    #p=scs.gamma.pdf(x,tau/nu,0.0,nus)
    return p

def vgamma(vx):
    arr=np.array([math.gamma(v) for v in vx])
    return arr
#@jit(nopython=True,parallel=True)
def genwVG(At,L,r,q,theta,sgm,nu,tau,nsim,b,mx):
    phii=-math.log(1.0-theta*nu-0.5*sgm**2*nu)*(1.0/nu)
    #B=(r-phii)*tau
    
    #wt=sgm*rnd.normalvariate(0.0,1.0)*math.sqrt(G)+theta*G
    B=(r-q-phii)*tau
    mt=np.empty(nsim)
    wt=np.empty_like(mt)
    I=np.empty_like(mt)
    
    for i in range(nsim):
        G=rnd.gammavariate(tau/nu,nu)
        #b=VGmx(tau,nu)[0]
        wt[i]=sgm*rnd.normalvariate(0.0,1.0)*math.sqrt(G)+(theta)*G             
        #U=random.random()
        #C=(math.log(U))
        #sqdelta=math.sqrt(0.25*wt**2-0.5*C*sgm**2*G)
        bb=b[i]
        w=wt[i]
        sqdelta=np.sqrt(0.25*w**2+0.5*bb*nu*sgm**2)
        if mx:
            mt[i]=np.exp((0.5*wt[i]+sqdelta))*At
            
        
        else:
            mb=(0.5*wt[i]-sqdelta)
            mt[i]=np.exp(mb)
            m=(At/L)*np.exp(B)*mt[i]
            I[i]=(m<1.0)
            #*gampdf(G,tau,nu,nu)
            #scs.gamma.pdf(G,tau/nu,0.0,nu)
    PD=np.mean(I) 
    #wtm=np.exp(wt)       
    return [PD,mt]      

def VGmx(tau,nu):
    
    #if (w<0.0):
        #m0=0.01
    #else:
     #   m0=abs(w)+1e-3
    m0=0.001
    run=True
    while run:
        
        try:
            u=random.random()
            root=sc.optimize.fsolve(gengaminv,m0,args=(tau,nu,u),xtol=1e-4)
            run=False
            break
        except:    
            pass    
    #sc.optimize.root(vgmxCDF,m0,args=(w,tau,nu,u))
    m=root
    #sc.optimize.root(gengaminv,m0,args=(w,tau,nu,u))
    #chk=gengaminv(root,tau,nu,u)
    #vgmxCDF(m,w,tau,nu,u)
    #gengaminv(root.x,w,tau,nu,u)

    return root
#######################################################################################
@jit(nopython=True,parallel=True)
def genwVGSB(At,Dt,L,r,thetaA,thetaD,sgmA,sgmD,rhoAD,nu,tau,nsim,b):
    phiiA=-math.log(1.0-thetaA*nu-0.5*sgmA**2*nu)*(1.0/nu);
    phiiD=-math.log(1.0-thetaD*nu-0.5*sgmD**2*nu)*(1.0/nu);
    thetaAD=thetaA-thetaD
    sgmAD=math.sqrt(sgmA*sgmA+sgmD*sgmD-2.0*rhoAD*sgmA*sgmD)
    phiiAD=-math.log(1.0-thetaAD*nu-0.5*sgmAD**2*nu)*(1.0/nu);
    #B=(phiiD-phiiA-q-phiiAD)*tau
    Lt=At/Dt
    #B=(r-phii)*tau
    
    #wt=sgm*rnd.normalvariate(0.0,1.0)*math.sqrt(G)+theta*G
    B=(phiiD-phiiA-phiiAD-q)*tau
    mt=np.empty(nsim)
    wt=np.empty_like(mt)
    I=np.empty_like(mt)
    Lt=At/Dt
    for i in range(nsim):
        G=rnd.gammavariate(tau/nu,nu)
        #b=VGmx(tau,nu)[0]
        wt[i]=sgmAD*rnd.normalvariate(0.0,1.0)*math.sqrt(G)+(thetaAD)*G             
        #U=random.random()
        #C=(math.log(U))
        #sqdelta=math.sqrt(0.25*wt**2-0.5*C*sgm**2*G)
        bb=b[i]
        w=wt[i]
        sqdelta=np.sqrt(0.25*w**2+0.5*bb*nu*sgmAD**2)
        if mx:
            mt[i]=np.exp((0.5*wt[i]+sqdelta))*At
            
        
        else:
            mb=(0.5*wt[i]-sqdelta)
            mt[i]=np.exp(mb)
            m=Lt*np.exp(B)*mt[i]
            I[i]=(m<L)
            #*gampdf(G,tau,nu,nu)
            #scs.gamma.pdf(G,tau/nu,0.0,nu)
    PD=np.mean(I) 
    #wtm=np.exp(wt)       
    return PD      
    

def makesurf(surfdata,voldata,K,tau,B,S,f,Xaxsnom,ttl):
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('Maturity')
    if (S and f):
        ax.set_xlabel(Xaxsnom)
        ax.set_ylabel('Leverage Barrier')
        ax.set_zlabel('PD')
        ax.set_title('Affine Linear Factor Variance Gamma Stochastic Barrier Structural Model PD$(A_{t}=100)$')
    elif (B and f):
        ax.set_xlabel(Xaxsnom)
        ax.set_ylabel('Debt Barrier')
        ax.set_zlabel('PD')
        ax.set_title('Affine Linear Factor Variance Gamma Barrier Structural Model PD$(A_{t}=100)$')
    else:
        ax.set_xlabel(Xaxsnom)
        ax.set_ylabel('Debt Barrier')
        ax.set_zlabel(ttl)
        ax.set_title('Variance Gamma Barrier Structural Model : '+ ttl+' $(A_{t}=100)$')
    # Make data.
    X = np.arange(tau.min(), tau.max(), (tau.max()-tau.min())/tau.size)
    Y = np.arange(K.min(), K.max(), (K.max()-K.min())/K.size)
    X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X**2+Y**2)
    #Z = np.sin(R)
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, surfdata, cmap=cm.RdBu,
                           linewidth=0, antialiased=True)
    
    # Customize the z axis.
    ax.set_zlim(0.00, np.max(surfdata))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=10)
    
    if (voldata):
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_xlabel(Xaxsnom)
        ax.set_ylabel('Distress(Debt) Barrier')
        ax.set_zlabel('Credit Spread')
        ax.set_title('Credit Spread Surface')
        ax.plot_surface(X[0:voldata.shape[0]], Y[0:voldata.shape[1]], voldata, cmap=cm.RdBu,
                           linewidth=0, antialiased=True)
#    else:
#        ax = fig.add_subplot(1, 2, 2, projection='3d')
#        ax.set_xlabel('Maturity')
#        ax.set_ylabel('Distress(Debt) Barrier')
#        ax.set_zlabel('Credit Spread')
#        ax.set_title('Variance Gamma Credit Spread Surface')
    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    #ax.plot_wireframe(X, Y, vsrf, rstride=10, cstride=10)
    
        ax.plot_surface(X, Y, voldata, cmap=cm.RdBu,
                           linewidth=0, antialiased=True)
    

#fig.colorbar(surf, shrink=0.5, aspect=10)
#plt.show()
def makescattersurf(calibdat,prds,X,Y):
    try:
        X,Y=np.meshgrid(X,Y)
    except:
        pass    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(X,Y,calibdat.values,cmap=cm.winter,linewidth=0, antialiased=True)
    if (not prds==None):
        ax.plot_wireframe(X,Y,prds,color='Red',linewidth=1.)
    ax.set_xlabel('Term Structure')
    ax.set_ylabel('Dates')
     
def VGSBo(x,At,Dt,r,Li,tau,q,smp,cdsP,S,mrt):
    n=cdsP.size
    if (S):
        L=1.
        PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        if (mrt):
            At=Dt*(1.0+math.exp(-x[6]))
            PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        #PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],x[3],x[4],2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
            PD=np.array(PD)
            sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
            eps=sp-cdsP
            
            pars=[x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1.0/(1.0+math.exp(-x[5])))-1.0,(1.0+math.exp(-x[6])),At,PD]
        else:
             PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        #PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],x[3],x[4],2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
             PD=np.array(PD)
             sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
             eps=sp-cdsP
             pars=[x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1.0/(1.0+math.exp(-x[5])))-1.0,PD]
    else:
        L=Li
        if mrt:
            At=L*(1.0+math.exp(-x[3]))
        #1.0/(1.0+math.exp(-x[2]))
        #1.0/(1.0+math.exp(-x[3]))
        PD=[VGcallBarrierVGD(At,r,L,x[0],x[1],x[2],tt,q,smp) for tt in tau]
        PD=np.array(PD)
        sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
        eps=sp-cdsP
        if mrt:
            #At=L*(1.0+math.exp(-x[3]))
            pars=[x[0],x[1],x[2],x[3],At,PD]
        else:
            pars=[x[0],x[1],x[2],PD]
    
    funv=math.sqrt(np.dot(eps,eps.T)/n)
    return [funv,sp,pars]
def VGSFBo(x,At,Dt,r,Li,tau,q,smp,cdsP,S):
    n=cdsP.size
    if (S):
        L=1.
        #PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],x[3],x[4],2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        PD=np.array(PD)
        sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
        eps=sp-cdsP
        pars=[x[0],x[1],x[2],x[3],x[4],2.0*(1.0/(1.0+math.exp(-x[5])))-1.0]
    else:
        L=Li
        PD=[VGcallfactorBarrierVGD(At,r,L,x[0],x[1],x[2],x[3],(1./(1+math.exp(-x[4]))),(1./(1+math.exp(-x[5]))),x[6],tt,q,smp) for tt in tau]
        PD=np.array(PD)
        sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
        eps=sp-cdsP
        pars=[x[0],x[1],x[2],x[3],1.0/(1.+math.exp(-x[4])),1.0/(1.+math.exp(-x[5])),x[6]]
    
    funv=math.sqrt(np.dot(eps,eps.T)/n)
    return [funv,sp,pars]



def VGSB(x,At,Dt,r,Li,tau,q,smp,cdsP,S,mrt):
    n=cdsP.size
    if (S):
        L=1.
        PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        if (mrt):
            At=Dt*(1.0+math.exp(-x[6]))
            PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        
            
        #PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],x[3],x[4],2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
    else:
        L=Li
        if (mrt):
            At=L*(1.0+math.exp(-x[3]))
        #1.0/(1.0+math.exp(-x[2]))#1.0/(1.0+math.exp(-x[3]))
        PD=[VGcallBarrierVGD(At,r,L,x[0],x[1],x[2],tt,q,smp) for tt in tau]
    PD=np.array(PD)
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    eps=sp-cdsP
    return math.sqrt(np.dot(eps,eps.T)/n)
def VGSBlm(x,At,Dt,r,Li,tau,q,smp,cdsP,S,mrt):
    #n=cdsP.size
    if (S):
        L=1.
        PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        if (mrt):
            At=Dt*(1.0+math.exp(-x[6]))
            PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        
            
        #PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],x[3],x[4],2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
    else:
        L=Li
        if mrt:
            At=L*(1.0+math.exp(-x[3]))
        #1.0/(1.0+math.exp(-x[2])#1.0/(1.0+math.exp(-x[3]))
        PD=[VGcallBarrierVGD(At,r,L,x[0],x[1],x[2],tt,q,smp) for tt in tau]
    PD=np.array(PD)
    #R=0.5
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
#    sp=np.append(sp,lchfuni(x[1],x[0],x[2],-1j)*100000)
#    cdsP=np.append(cdsP,r*100000)
    eps=sp-cdsP
    return eps

def VGSFB(x,At,Dt,r,Li,tau,q,smp,cdsP,S):
    n=cdsP.size
    if (S):
        L=1.
        #PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4])),2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
        PD=[VGcallSBarrierVGD(At,Dt,r,L,x[0],x[1],x[2],x[3],x[4],2.0*(1./(1+math.exp(-x[5])))-1.,tt,q,smp) for tt in tau]
    else:
        L=Li
        PD=[VGcallfactorBarrierVGD(At,r,L,x[0],x[1],x[2],x[3],(1./(1+math.exp(-x[4]))),(1./(1+math.exp(-x[5]))),x[6],tt,q,smp) for tt in tau]
    PD=np.array(PD)
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    eps=sp-cdsP
    return math.sqrt(np.dot(eps,eps.T)/n)
def VGcallF(x,At,Dt,r,tau,q,cdsP):
    n=cdsP.size    
    survP=[VGcalld(At,r,Dt,x[0],x[1],x[2],tt,q) for tt in tau]
    PD=np.array(1.0-np.array(survP))
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    #sp=np.array(sp)
    eps=sp-cdsP
    return math.sqrt(np.dot(eps,eps.T)/n)
def VGcallFlm(x,At,Dt,r,tau,q,cdsP,mrt):
    #n=cdsP.size    
    if mrt:
        At=Dt*(1.0+np.exp(-x[3]))
    survP=[VGcalld(At,r,Dt,x[0],x[1],x[2],tt,q) for tt in tau]
    PD=np.array(1.0-np.array(survP))
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    #sp=np.array(sp)
    eps=sp-cdsP
    return eps

def VGcallFopt(x,At,Dt,r,tau,q,optP):
    n=optP.size    
    modP=[VGcall(At,r,K,x[0],x[1],x[2],tau,q)[0] for K in Dt]
    #PD=np.array(1.0-np.array(survP))
    #sp=(-np.log((1.0-PD)+PD*0.5)/tau)*1cali0000
    #sp=np.array(sp)
    eps=modP-optP
    return math.sqrt(np.dot(eps,eps.T)/n)
def VGcallFoptlm(x,At,Dt,r,tau,q,optP):
    #n=optP.size    
    modP=[VGcall(At,r,K,x[0],x[1],x[2],tau,q)[0] for K in Dt]
    #PD=np.array(1.0-np.array(survP))
    #sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    #sp=np.array(sp)
    eps=modP-optP
    return eps
def VGcallFoptout(x,At,Dt,r,tau,q):
    #n=optP.size    
    modP=[VGcall(At,r,K,x[0],x[1],x[2],tau,q)[0] for K in Dt]
    return modP

def VGcallFo(x,At,Dt,r,tau,q,cdsP,mrt):
    n=cdsP.size    
    if mrt:
        At=Dt*(1.0+np.exp(-x[3]))
    survP=[VGcalld(At,r,Dt,x[0],x[1],x[2],tt,q) for tt in tau]
    PD=np.array(1.0-np.array(survP))
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    eps=sp-cdsP
    funv=math.sqrt(np.dot(eps,eps.T)/n)
    pars=x
    return [funv,PD,sp,pars,At]

def VGcallfF(x,At,Dt,r,tau,q,cdsP,c):
    n=cdsP.size    
    survP=[VGfacall(At,r,Dt,x[0],x[1],x[2],x[3],x[4],x[5],tt,q,c)[1] for tt in tau]
    PD=np.array(1.0-np.array(survP))
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    #sp=np.array(sp)
    eps=sp-cdsP
    return math.sqrt(np.dot(eps,eps.T)/n)
def VGcallfFo(x,At,Dt,r,tau,q,cdsP,c):
    #n=cdsP.size    
    survP=[VGcalld(At,r,Dt,x[0],x[1],x[2],x[3],x[4],x[5],tt,q)[1] for tt in tau]
    PD=np.array(1.0-np.array(survP))
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    #eps=sp-cdsP
    return [PD,sp]



@jit(nopython=True,parallel=True)
def chfunVG(theta,nu,sgm,u,t):
    denom=(1-1j*u*theta*nu+0.5*((sgm*u)**2)*nu)
    return (denom)**(-t/nu)
@jit(nopython=True,parallel=True)
def chfunVGs(theta,nu,sgm,u,t):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    thetas=sgm**2+theta
    denom=(1-1j*u*thetas*nus+0.5*((sgm*u)**2)*nus)
    return (denom)**(-t/nu)
@jit(nopython=True,parallel=True)
def dphidsgmS(theta,nu,sgm):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    thetas=sgm**2+theta
    deriv=sgm/(1-thetas*nus-0.5*sgm**2*nus)
    return deriv
@jit(nopython=True,parallel=True)
def dphidsgm(theta,nu,sgm):
    deriv=sgm/(1-theta*nu-0.5*sgm**2*nu)
    return deriv

@jit(nopython=True,parallel=True)
def lchfuni(theta,nu,sgm,u):
    denom=np.log((1-1j*u*theta*nu+0.5*((sgm*u)**2)*nu).real)*(-1.0/nu)
    return denom
@jit(nopython=True,parallel=True)
def lchfunis(theta,nu,sgm,u):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    thetas=sgm**2+theta
    denom=np.log((1-1j*u*thetas*nus+0.5*((sgm*u)**2)*nus).real)*(-1.0/nu)
    return denom
@jit(nopython=True,parallel=True)
def chfunk(theta,nu,sgm,u):
    denom=((1-1j*u*theta*nu+0.5*((sgm*u)**2)*nu).real)
    return denom


@jit(nopython=True,parallel=True)
def NdVG(St,r,K,theta,nu,sgm,g,tau,q):
    phii=lchfuni(theta,nu,sgm,-1j)
    d1=(math.log(St/K)+(r-q-phii)*tau+theta*g)/(sgm*math.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    Nd1=appNormCDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return Nd1

@jit(nopython=True,parallel=True)
def fdVG(St,r,K,theta,nu,sgm,g,tau,q):
    phii=lchfuni(theta,nu,sgm,-1j)
    d1=(math.log(St/K)+(r-q-phii)*tau+theta*g)/(sgm*math.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    Nd1=appNormPDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return Nd1

@jit(nopython=True,parallel=True)
def NdVGm(St,r,K,theta,nu,sgm,g,tau,q):
    phii=lchfuni(theta,nu,sgm,-1j)
    thetas=sgm**2+theta
    d1=(math.log(St/K)+(r-q-phii)*tau+thetas*g)/(sgm*math.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    Nd1=appNormCDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return Nd1

@jit(nopython=True,parallel=True)
def NdVGmSFZ(St,r,K,theta,nu,sgm,g,tau,q,c,z):
    phii=lchfuni(theta,nu,sgm,-1j)
    thetas=sgm**2+theta
    d1=(math.log(St/K)+(r-q-phii)*tau+c*z+thetas*g)/(sgm*math.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    Nd1=appNormCDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return Nd1
@jit(nopython=True,parallel=True)
def NdVGmFZ(St,r,K,theta,nu,sgm,g,tau,q,c,z):
    phii=lchfuni(theta,nu,sgm,-1j)
    d1=(math.log(St/K)+(r-q-phii)*tau+c*z+theta*g)/(sgm*math.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    Nd1=appNormCDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return Nd1


@jit(nopython=True,parallel=True)
def fdVGm(St,r,K,theta,nu,sgm,g,tau,q):
    phii=lchfuni(theta,nu,sgm,-1j)
    thetas=sgm**2+theta
    d1=(math.log(St/K)+(r-q-phii)*tau+thetas*g)/(sgm*math.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    fd1=appNormPDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return fd1

def FVG2(St,r,K,nu,theta,sgm,tau,q):
    #phii=chfuni(theta,nu,sgm,-1j)
    f=lambda g: (NdVG(St,r,K,theta,nu,sgm,g,tau,q))*gampdf(g,tau,nu,nu)
    GK=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    vgcdf=GK(f,0.0,50.)
    #sc.integrate.quad(f,0.0,10.0)
    return vgcdf

def FVG1(St,r,K,nu,theta,sgm,tau,q):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    f=lambda g: (NdVGm(St,r,K,theta,nu,sgm,g,tau,q))*gampdf(g,tau,nu,nus)
    GK=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    vgcdf=GK(f,0.0,50.)
    #sc.integrate.quad(f,0.0,50.0,full_output=1)
    #GK(f,0.0,50.)
    return vgcdf

def FVG1Z(St,r,K,nu,theta,sgm,tau,q,c,z):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    f=lambda g: (NdVGmSFZ(St,r,K,theta,nu,sgm,g,tau,q,c,z))*gampdf(g,tau,nu,nus)
    #scs.gamma.pdf(g,tau/nu,0.0,nus)
    #gampdf(g,tau,nu,nus)
    # GK=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    # vgcdf=GK(f,0.0,50.)
    vgcdf=sc.integrate.quad(f,0.0,50.0,full_output=1)[0]
    return vgcdf

def FVG2Z(St,r,K,nu,theta,sgm,tau,q,c,z):
    #k=chfunk(theta,nu,sgm,-1j)
    #nus=nu/k
    f=lambda g: (NdVGmFZ(St,r,K,theta,nu,sgm,g,tau,q,c,z))*gampdf(g,tau,nu,nu)
    #scs.gamma.pdf(g,tau/nu,0.0,nu)
    #gampdf(g,tau,nu,nu)
    # GK=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    # vgcdf=GK(f,0.0,50.)
    vgcdf=sc.integrate.quad(f,0.0,50.0,full_output=1)[0]
    return vgcdf


def fVG1vg(St,r,K,nu,theta,sgm,tau,q):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    f=lambda g: (fdVGm(St,r,K,theta,nu,sgm,g,tau,q))*gampdf(g,tau,nu,nus)
    vgcdf=sc.integrate.quad(f,0.0,10.0)
    return vgcdf

def VGdtheta(St,r,K,nu,theta,sgm,tau,q):
    dtheta=((St*FVG1(St,r,K,nu,theta+0.0001,sgm,tau,q)[0]-np.exp(-r*tau)*K*FVG2(St,r,K,nu,theta+0.0001,sgm,tau,q)[0])\
    -((St*FVG1(St,r,K,nu,theta-0.0001,sgm,tau,q)[0]-np.exp(-r*tau)*K*FVG2(St,r,K,nu,theta-0.0001,sgm,tau,q)[0])))/0.0002
    return dtheta

def VGvega(St,r,K,nu,theta,sgm,tau,q):
    vega=((St*FVG1(St,r,K,nu,theta,sgm+0.0001,tau,q)[0]-np.exp(-r*tau)*K*FVG2(St,r,K,nu,theta,sgm+0.0001,tau,q)[0])\
    -((St*FVG1(St,r,K,nu,theta,sgm-0.0001,tau,q)[0]-np.exp(-r*tau)*K*FVG2(St,r,K,nu,theta,sgm-0.0001,tau,q)[0])))/0.0002
    
#    phii=lchfuni(theta,nu,sgm,-1j)
#    thetas=sgm**2+theta
#    k=chfunk(theta,nu,sgm,-1j)
#    nus=nu/k
#    ds=lambda g: (math.log(St/K)+(r-q-phii)*tau+thetas*g)
#    d=lambda g:(math.log(St/K)+(r-q-phii)*tau+theta*g)
#    dfs=lambda g:ds(g)-d(g)
#    D=dphidsgm(theta,nu,sgm)
#    #Ds=dphidsgmS(theta,nu,sgm)
#    
#    fd1=lambda g: fdVGm(St,r,K,theta,nu,sgm,g,tau,q)
#    fd2=lambda g: fdVG(St,r,K,theta,nu,sgm,g,tau,q)
#    
#    f2=lambda g:((D*tau*sgm+d(g))/(sgm**2*np.sqrt(g))*fd2(g)*K*math.exp(-r*tau)*gampdf(g,tau,nu,nu))
#    f1=lambda g:((sgm*(D*tau-2.0*g*sgm)+ds(g))/(sgm**2*np.sqrt(g))*fd1(g)*St*math.exp(-q*tau)*gampdf(g,tau,nu,nus))
#    #f=lambda g:fd1(g)*St*math.exp(-q*tau)*gampdf(g,tau,nu,nus)-fd2(g)*K*math.exp(-r*tau)*gampdf(g,tau,nu,nu)
#    #fd=lambda g:(-dfs(g)*fd1(g)*St)/(sgm**2*np.sqrt(g))*gampdf(g,tau,nu,nus)
#        #(fdVG(St,r,K,theta,nu,sgm,g,tau,q))*gampdf(g,tau,nu,nu)
#        f=lambda g:f1(g)-f2(g)
#    vgcdf=sc.integrate.quad(f2,0.0,10.0)
    return vega

def FVGc(S0,r,K,nu,theta,sgm,tau,q):
    k=chfunk(theta,nu,sgm,-1j)
    nus=nu/k
    f=lambda g: S0*math.exp(-q*tau)*(NdVGm(S0,r,K,theta,nu,sgm,g,tau,q))*gampdf(g,tau,nu,nus)\
    -K*math.exp(-r*tau)*(NdVG(S0,r,K,theta,nu,sgm,g,tau,q))*gampdf(g,tau,nu,nu)
    vgcdf=sc.integrate.quad(f,0.0,10.0)[0]
    return vgcdf


@jit(nopython=True,parallel=True)
def IFFVG1(St,r,q,K,theta,nu,sgm,tau):
    phii=lchfuni(theta,nu,sgm,-1j)
    #phiis=lchfunis(theta,nu,sgm,-1j)
    P1=sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*math.log(K)+1j*u*math.log(St)+(1j*u)*(r-q-phii)*tau)*chfunVGs(theta,nu,sgm,u,tau)/(1j*u))*(1.0/math.pi),1e-6,1500.0)[0]+0.5
    P2=sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*math.log(K)+1j*u*math.log(St)+1j*u*(r-q-phii)*tau)*chfunVG(theta,nu,sgm,u,tau)/(1j*u))*(1.0/math.pi),1e-6,1500.0)[0]+0.5
    call=St*P1-math.exp(-r*tau)*K*P2
    return [call,P1,P2]
@jit(nopython=True,parallel=True)
def gampdf(x,tau,nu,nus):
    p=((x**(tau/nu)/x)*np.exp(-x/nus)/math.gamma(tau/nu))*nus**(-tau/nu)
    #p=scs.gamma.pdf(x,tau/nu,0.0,nus)
    return p
@jit(nopython=True,parallel=True)
def appNormCDF(x):
    p=1.0/(np.exp(-358.0*x/23.0+111*np.arctan(37.0*x/294))+1.0)
    return p
@jit(nopython=True,parallel=True)
def appNormPDF(x):
    dd=(-358.0/23.0+111*(1.0/(1.0+((37.0/294)**2)*x*x))*37.0/294)
    nn=(appNormCDF(x)*appNormCDF(x))
    p=-dd*math.exp(-358.0*x/23.0+111*math.atan(37.0*x/294))*nn
    #p=1.0/(math.exp(-358.0*x/23.0+111*math.atan(37.0*x/294))+1.0)
    return p

@jit(nopython=True,parallel=True)
def nrmpdf(x,mu, sgm):
    #import numpy as np

    pis=math.pi
    y=math.exp(-0.5*(((x-mu)*(x-mu))/(sgm*sgm)))/math.sqrt(sgm*sgm*2*pis)
    return y
@jit(nopython=True,parallel=True)
def normcdff(x,mu,sgm):
    return sc.integrate.quad(nrmpdf,-10.0,x,args=(mu,sgm))[0]

def CDFVG1Z(S0,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c):
    
    #f=lambda gZ: FFVG1Z(S0,r,K,thetaY,thetaZ,nu,sgmY,sgmZ,gZ,tau,q,c)*gampdf(gZ,tau,nu,nu)
    phii=lchfuni(theta,nu,sgm,-1j)
    #phiiZ=lchfuni(thetaZ,nuZ,sgmZ,-1j*c)
    d=lambda Z: math.log(S0/K)+(r-q-phii)*tau+c*Z
    #mu=-phiiZ
    #thetasZ=thetaZ+c*sgmZ**2
    f=lambda Z: CDFVG(d(Z),thetaY,sgmY,nuY,0.0,tau,True,c)[0]\
    *VGdensity(Z,thetaZ,sgmZ,nuZ,0.0,tau,nuZ,c)
    #*math.exp(-phiiZ+c*Z)
    vgcdfZ=sc.integrate.quad(f,-5.,5.01,epsabs=1e-5)[0]
    return vgcdfZ

def CDFVG2Z(S0,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c):
    #phii=chfuni(theta,nu,sgm,-1j)
    #f=lambda gZ: FFVG2Z(S0,r,K,thetaY,thetaZ,nu,sgmY,sgmZ,gZ,tau,q,c)*gampdf(gZ,tau,nu,nu)
    phii=lchfuni(theta,nu,sgm,-1j)
    d=lambda Z: math.log(S0/K)+(r-q-phii)*tau+c*Z
    
    f=lambda Z: CDFVG(d(Z),thetaY,sgmY,nuY,0.0,tau,False,c)[0]\
    *VGdensity(Z,thetaZ,sgmZ,nuZ,0.0,tau,nuZ,0.0)
    
    vgcdfZ=sc.integrate.quad(f,-5.0,5.01,epsabs=1e-5)[0]
    return vgcdfZ


@jit(nopython=False,parallel=True)
def VGfcall(St,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c):
    P1=CDFVG1Z(St,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c)
    P2=CDFVG2Z(St,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c)    
    Cfz=St*P1-K*math.exp(-r*tau)*P2
    return [Cfz,P2]
#Cfz=St*CDFVG1Z(St,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c)-K*math.exp(-r*tau)*CDFVG2Z(St,r,K,theta,nu,sgm,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c)
def VGfacall(St,r,K,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tau,q,c):
    theta=thetaY+c*thetaZ
    sgm=np.sqrt(sgmY**2+(c*sgmZ)**2)
    nu=(nuY*nuZ)/(nuY+nuZ)
    #FVG_S=FVG1(St,r,K,nu,theta,sgm,tau,q)[0]
    FVG=FVG2(St,r,K,nu,thCDFeta,sgm,tau,q)[0]
    callp=St*FVG1(St,r,K,nu,theta,sgm,tau,q)[0]-np.exp(-r*tau)*K*FVG
    #putp=np.exp(-r*tau)*K*(1.0-FVG)-np.exp(-q*tau)*St*(1.0-FVG_S)
    #IS=(-1.0/tau)*np.log(1.0-putp/(K*np.exp(-r*tau)))
    return [callp,1.0-FVG]  

def VGcall(St,r,K,nu,theta,sgm,tau,q):
    FVG_S=FVG1(St,r,K,nu,theta,sgm,tau,q)
    FVG=FVG2(St,r,K,nu,theta,sgm,tau,q)
    callp=np.exp(-q*tau)*St*FVG_S-np.exp(-r*tau)*K*FVG
    putp=np.exp(-r*tau)*K*(1.0-FVG)-np.exp(-q*tau)*St*(1.0-FVG_S)
    IS=(-1.0/tau)*np.log(1.0-putp/(K*np.exp(-r*tau)))
    return [callp,putp,FVG_S,FVG,FVG,IS]  

def VGcallFZ(St,r,K,nu,theta,sgm,tau,q,c,z):
    FVG_S=FVG1Z(St,r,K,nu,theta,sgm,tau,q,c,z)
    FVG=FVG2Z(St,r,K,nu,theta,sgm,tau,q,c,z)
    callp=np.exp(-q*tau)*St*FVG_S-np.exp(-r*tau)*K*FVG
    putp=np.exp(-r*tau)*K*(1.0-FVG)-np.exp(-q*tau)*St*(1.0-FVG_S)
    #IS=(-1.0/tau)*np.log(1.0-putp/(K*np.exp(-r*tau)))
    return [callp,putp,FVG_S,FVG]  

def VGFZPD(St,r,K,nu,theta,sgm,tau,q,c,z):
    FVG=FVG2Z(St,r,K,nu,theta,sgm,tau,q,c,z)
    #IS=(-1.0/tau)*np.log(1.0-putp/(K*np.exp(-r*tau)))
    return FVG



def VGcalld(St,r,K,nu,theta,sgm,tau,q):    
    phii=lchfuni(theta,nu,sgm,-1j)
    d=math.log(St/K)+(r-q-phii)*tau
    #mu=0.0
    P2=CDFVG(d,theta,sgm,nu,0.0,tau,False,1.0)[0]
    return P2
def VGcalldel(St,r,K,nu,theta,sgm,tau,q):    
    phii=lchfuni(theta,nu,sgm,-1j)
    d=math.log(St/K)+(r-q-phii)*tau
    #mu=0.0
    P1=CDFVG(d,theta,sgm,nu,0.0,tau,True,1.0)[0]*math.exp(-q*tau)
    return P1

def cosFf(mdl,s,sgmY,sgmZ,sgm,theta,thetaY,thetaZ,nuY,nuZ,nu,d,npt,tau,c,af):
    
    N=2**npt;
    k=np.linspace(0,N,N,dtype=int)
    #k=range(N)
    if (mdl):
        a=-5.;b=5.
        #w=[1.0 for k in range(N-1)]
        w=np.ones(N);
        #w.append(0.5)
#        w.sort()
#        w=np.array(w)
        w[0]=0.5
        Fk=w*(2.0/(b-a))*(normCF(k*math.pi/(b-a))*np.exp(-1j*(a*k*math.pi)/(b-a))).real;
        dt=0.001;
        n=int((b-d)/dt)
        dd=d+np.linspace(0,n,num=n,dtype=int)*dt
        #dd=np.array(range(n))*dt+d
        #np.linspace(d, b,n)
        cc=k*math.pi
        dda=((dd-a)/(b-a))
        ddm=np.matlib.repmat(dda,N,1).T
        cddm=ddm*cc
        mcos=np.cos(cddm)
        dm=np.matmul(mcos,Fk)
        fpdf=dm
        #fpdf=[np.sum(Fk*np.cos(cc*((ddd-a)/(b-a)))) for ddd in dd]
        F=np.sum(fpdf)*dt
        #F=np.sum(f)*dt;
    else:
    
        c1 = theta*tau;
        c2 = (sgm**2+nu*theta**2)*tau;
        c4 = 3*(sgm**4*nu+2*theta**4*nu**3+4*sgm**2*theta**2*nu**2)*tau;
#%Truncation range


        a = c1-npt*math.sqrt(c2+math.sqrt(c4))
        b = c1+npt*math.sqrt(c2+math.sqrt(c4))
        
#cf=@(u)(real(1-1i.*theta.*nu.*u+sgm^2*nu.*u.^2/ 2).^(-tau./nu));
#cfi=(1.0-theta*nu-0.5*sgm^2*nu).^(-tau/nu);
        if(af):            
            w=np.ones(N);
            w[0]=0.5;             
            u=k*math.pi/(b-a)
            Fls=[(chfunVG(thetaY,nuY,sgmY,uu,tau)*chfunVG(thetaZ,nuZ,sgmZ,c*uu,tau)).real for uu in u]
            Fk=w*(2.0/(b-a))*(np.array(Fls)*np.exp(-1j*(a*k*math.pi)/(b-a))).real
            dt=0.001;
            n=int((b-d)/dt)
            #dd=np.array(range(n))*dt+d
            dd=d+np.linspace(0,n,num=n,dtype=int)*dt
            #np.linspace(d, b, n)
            cc=k*math.pi
            dda=((dd-a)/(b-a))
            ddm=np.matlib.repmat(dda,N,1).T
            cddm=ddm*cc
            mcos=np.cos(cddm)
            dm=np.matmul(mcos,Fk)
            fpdf=dm
            
            #fpdf=[np.sum(Fk*np.cos(cc*((ddd-a)/(b-a)))) for ddd in dd]
            
            #fpdf=[np.sum(Fk*np.cos(cc*((ddd-a)/(b-a)))) for ddd in dd]
            F=np.cumsum(fpdf)*dt
        else:
            
            w=np.ones(N);
            w[0]=0.5;             
            u=k*math.pi/(b-a)
            Fl=[chfunVG(theta,nu,sgm,uu,tau).real for uu in u]
            Fk=w*(2.0/(b-a))*(np.array(Fl)*np.exp(-1j*(a*k*math.pi)/(b-a))).real
            dt=0.001;
            n=int((b-d)/dt)
            #dd=np.array(range(n))*dt+d
            dd=d+np.linspace(0,n,num=n,dtype=int)*dt
            #np.linspace(d, b,n)
            cc=k*math.pi
            dda=((dd-a)/(b-a))
            ddm=np.matlib.repmat(dda,N,1).T
            cddm=ddm*cc
            mcos=np.cos(cddm)
            dm=np.matmul(mcos,Fk)
            fpdf=dm
            
            #fpdf=[np.sum(Fk*np.cos(cc*((ddd-a)/(b-a)))) for ddd in dd]
            F=np.cumsum(fpdf)*dt
            
    
    return [F,fpdf,dd]
def corls(pars,cv,obspars,mul,high,retsm,vgm):
    theta1=obspars[0];theta2=obspars[1];theta3=obspars[2];nu1=obspars[3];
    nu2=obspars[4];nu3=obspars[5];sgm1=obspars[6];sgm2=obspars[7];sgm3=obspars[8];
    if high:
        eps=np.zeros(9)
    else:
        eps=np.zeros(3)
    rho23=cv[1,2];rho13=cv[0,2];rho12=cv[0,1]
    if vgm:
        V11=np.sqrt(theta1**2*nu1+sgm1**2)
        V22=np.sqrt(theta2**2*nu2+sgm2**2)
        V33=np.sqrt(theta3**2*nu3+sgm3**2)
    else:
        V11,V22,V33=np.sqrt(retsm.var().values)
    
    a1=pars[3];a2=pars[4];a3=pars[5];
    Csgm=np.sqrt(min(sgm1**2/a1**2,sgm2**2/a2**2,sgm3**2/a3**2));
    Cnu=min(1./nu1,1./nu2,1./nu3)
    sgmZ=Csgm/(1.+np.exp(-pars[2]));thetaZ=pars[1];inuZ=Cnu/(1.+np.exp(-pars[0]));nuZ=1./inuZ;
    
#    eps[3]=a1-(sgm1**2/theta1)*(thetaZ/sgmZ**2)
#    eps[4]=a2-(sgm2**2/theta2)*(thetaZ/sgmZ**2)
#    eps[5]=a3-(sgm3**2/theta3)*(thetaZ/sgmZ**2)
#
    if high:
        eps[0]=rho12-(a1*a2)*(sgmZ+thetaZ**2*nuZ)/(V11*V22)
        eps[1]=rho13-(a1*a3)*(sgmZ+thetaZ**2*nuZ)/(V11*V33)
        eps[2]=rho23-(a2*a3)*(sgmZ+thetaZ**2*nuZ)/(V22*V33)
        eps[3]=nuZ*(thetaZ*a1)-nu1*theta1
        eps[4]=nuZ*(thetaZ*a2)-nu2*theta2
        eps[5]=nuZ*(thetaZ*a3)-nu3*theta3
        eps[6]=nuZ*(a1**2*sgmZ**2)-(nu1*sgm1**2)
        eps[7]=nuZ*(a2**2*sgmZ**2)-(nu2*sgm2**2)
        eps[8]=nuZ*(a3**2**sgmZ**2)-(nu3*sgm3**2)
    else:
        eps[0]=rho12-(a1*a2)*(sgmZ+thetaZ**2*nuZ)/(V11*V22)
        eps[1]=rho13-(a1*a3)*(sgmZ+thetaZ**2*nuZ)/(V11*V33)
        eps[2]=rho23-(a2*a3)*(sgmZ+thetaZ**2*nuZ)/(V22*V33)
        
    #eps[4]=a1*(thetaZ*sgmZ)-nu1*theta1
    #pars[3]=nu1*theta1/(thetaZ*sgmZ)
    #pars[4]=nu2*theta2/(thetaZ*sgmZ)
    #pars[5]=nu3*theta3/(thetaZ*sgmZ)
    
#    eps[0]=rho12-(sgm1**2/theta1)*(sgm2**2/theta2)*(pars[2]**2/pars[3]+pars[2]**4*pars[1]/pars[3]**2)/(V11*V22)
#    eps[1]=rho13-(sgm1**2/theta1)*(sgm3**2/theta3)*(pars[2]**2/pars[3]+pars[2]**4*pars[1]/pars[3]**2)/(V11*V33)
#    eps[2]=rho23-(sgm2**2/theta2)*(sgm3**2/theta3)*(pars[2]**2/pars[3]+pars[2]**4*pars[1]/pars[3]**2)/(V22*V33)
    consts={'rhos':[rho12,rho13,rho23],'nuth':[nu1*theta1,nu2*theta2,nu3*theta3],'nusgm':[nu1*sgm1**2,nu2*sgm2**2,nu3*sgm3**2]}
    ests={'parsthZnuZsgmZa...':[thetaZ,nuZ,sgmZ,a1,a2,a3]}
    #R23*pars[2]**2*((1.0/(1+np.exp(-pars[0])))+((2.0/(1+np.exp(-pars[1]))-1.0)**2)*nuz)
    #-pars[1]*pars[2]*pars[3]
    
    #R23*pars[2]**2*((1.0/(1+np.exp(-pars[0])))+((2.0/(1+np.exp(-pars[1]))-1.0)**2)*nuz)
    #eps[1]=s13-R13*pars[2]**2*((1.0/(1+np.exp(-pars[0])))+((2.0/(1+np.exp(-pars[1]))-1.0)**2)*nuz)
    #-pars[0]*pars[2]*pars[3]
    #R13*pars[2]**2*((1.0/(1+np.exp(-pars[0])))+((2.0/(1+np.exp(-pars[1]))-1.0)**2)*nuz)
    #eps[2]=s12-R12*(R23**2)*pars[2]**2*((1.0/(1+np.exp(-pars[0])))+((2.0/(1+np.exp(-pars[1]))-1.0)**2)*nuz)
    #pars[0]*pars[1]*pars[3]
    #R12*(R23**2)*pars[2]**2*((1.0/(1+np.exp(-pars[0])))+((2.0/(1+np.exp(-pars[1]))-1.0)**2)*nuz)
    n=eps.shape[0]
    if mul:
        return [np.sqrt(np.dot(eps,eps)/n),eps,consts,ests]
    else:
        return np.sqrt(np.dot(eps,eps)/n)
    
def calibcorr(obspars,highlow,prnlogchf,method,retsm,vgm):
    cv=np.array(retsm.corr())
    theta1=obspars[0];theta2=obspars[1];theta3=obspars[2];nu1=obspars[3];
    nu2=obspars[4];nu3=obspars[5];sgm1=obspars[6];sgm2=obspars[7];sgm3=obspars[8];
    x0=np.ones(6)/6
    pars=x0;mul=False
    #for it in range(10000):   
    mul=False;high=highlow
    optsCG = {'maxiter' : None, 'disp' : True,'gtol' : 1e-5,'norm' : np.inf, 'eps' : 1.4901161193847656e-07}  # default value.

    if (method=='trf'):
        opt=sc.optimize.least_squares(corls,pars,method='trf',args=(cv,obspars,mul,high,retsm,vgm),xtol=1e-5, ftol=1e-10,verbose=0,max_nfev=10000)
    elif (method=='BFGS'):
        opt=sc.optimize.minimize(corls,pars,args=(cv,obspars,mul,high,retsm,vgm),method='BFGS',options=optsCG)
    else:
        opt=sc.optimize.minimize(corls,pars,args=(cv,obspars,mul,high,retsm,vgm),method='CG',options=optsCG)
    mul=True;pars=opt.x;funv=corls(pars,cv,obspars,mul,high,retsm,vgm);opt.fun
        #eps=opt.fun-pfun,pfun=opt.fun
        
    #funv[1];funv[3]
    a1,a2,a3=pd.DataFrame(funv[3]).iloc[3].values[0],pd.DataFrame(funv[3]).iloc[4].values[0],pd.DataFrame(funv[3]).iloc[5].values[0]
    thetaZ,nuZ,sgmZ=pd.DataFrame(funv[3]).iloc[0].values[0],pd.DataFrame(funv[3]).iloc[1].values[0],pd.DataFrame(funv[3]).iloc[2].values[0]
    thetaS1=theta1-a1*thetaZ;thetaS2=theta2-a2*thetaZ;thetaS3=theta3-a3*thetaZ;
    
    sgmS1=np.sqrt(sgm1**2-a1**2*sgmZ**2);sgmS2=np.sqrt(sgm2**2-a2**2*sgmZ**2);
    sgmS3=np.sqrt(sgm3**2-a3**2*sgmZ**2);
    
    nuS1=(1.0/nu1-1.0/nuZ)**-1;nuS2=(1.0/nu2-1.0/nuZ)**-1;nuS3=(1.0/nu3-1.0/nuZ)**-1
    if (vgm):
        V22=np.sqrt(theta2**2*nu2+sgm2**2);V11=np.sqrt(theta1**2*nu1+sgm1**2);V33=np.sqrt(theta3**2*nu3+sgm3**2)
    else:
        V11,V22,V33=np.sqrt(retsm.var().values)
    erho12=(a1*a2)*(sgmZ+thetaZ**2*nuZ)/(V11*V22); 
    erho13=(a1*a3)*(sgmZ+thetaZ**2*nuZ)/(V11*V33);    
    erho23=(a2*a3)*(sgmZ+thetaZ**2*nuZ)/(V22*V33);    
    paramD={'EstRho':[erho12,erho13,erho23],'Rho':[cv[0,1],cv[0,2],cv[1,2]],'nu':[nu1,nu2,nu3],'nuS':[nuS1,nuS2,nuS3],'nuZ,thetaZ,sigmaZ':[nuZ,thetaZ,sgmZ],'a':[a1,a2,a3],'thetaS':[thetaS1,thetaS2,thetaS3],'sigmaS':[sgmS1,sgmS2,sgmS3]}
    nuY1,thetaY1,sgmY1,nuZ,thetaZ,sgmZ=paramD['nuS'][0],paramD['thetaS'][0],paramD['sigmaS'][0],paramD['nuZ,thetaZ,sigmaZ'][0],paramD['nuZ,thetaZ,sigmaZ'][1],paramD['nuZ,thetaZ,sigmaZ'][2]
    nuY2,thetaY2,sgmY2,nuZ,thetaZ,sgmZ=paramD['nuS'][1],paramD['thetaS'][1],paramD['sigmaS'][1],paramD['nuZ,thetaZ,sigmaZ'][0],paramD['nuZ,thetaZ,sigmaZ'][1],paramD['nuZ,thetaZ,sigmaZ'][2]
    nuY3,thetaY3,sgmY3,nuZ,thetaZ,sgmZ=paramD['nuS'][2],paramD['thetaS'][2],paramD['sigmaS'][2],paramD['nuZ,thetaZ,sigmaZ'][0],paramD['nuZ,thetaZ,sigmaZ'][1],paramD['nuZ,thetaZ,sigmaZ'][2]
    
    if prnlogchf:
    
        print(lchfuni(theta3,nu3,sgm3,-1j),lchfuni(thetaZ,nuZ,sgmZ,-1j*a3)+lchfuni(thetaY3,nuY3,sgmY3,-1j))
        print(lchfuni(theta2,nu2,sgm2,-1j),lchfuni(thetaZ,nuZ,sgmZ,-1j*a2)+lchfuni(thetaY2,nuY2,sgmY2,-1j))
        print(lchfuni(theta1,nu1,sgm1,-1j),lchfuni(thetaZ,nuZ,sgmZ,-1j*a1)+lchfuni(thetaY1,nuY1,sgmY1,-1j))
        print(opt.fun)
    return [(nuY1,thetaY1,sgmY1,nuY2,thetaY2,sgmY2,nuY3,thetaY3,sgmY3,nuZ,thetaZ,sgmZ),paramD,opt]

class YahooDailyReader():
    
    def __init__(self, symbol=None, start=None, end=None):
        import datetime, time
        self.symbol = symbol
        
        # initialize start/end dates if not provided
        if end is None:
            end = datetime.datetime.today()
        else:
            end= datetime.datetime(int(end.split("-")[0]),int(end.split("-")[1]),int(end.split("-")[2]))
        if start is None:
            start=datetime.datetime(2007,1,5)
        else:
            start = datetime.datetime(int(start.split("-")[0]),int(start.split("-")[1]),int(start.split("-")[2]))
        
        self.start = start
        self.end = end
        
        # convert dates to unix time strings
        unix_start = int(time.mktime(self.start.timetuple()))
        day_end = self.end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))
        
        url = 'https://finance.yahoo.com/quote/{}/history?'
        url += 'period1={}&period2={}'
        url += '&filter=history'
        url += '&interval=1d'
        url += '&frequency=1d'
        self.url = url.format(self.symbol, unix_start, unix_end)
        
    def read(self):
        import requests, re, json
       
        r = requests.get(self.url)
        
        ptrn = r'root\.App\.main = (.*?);\n}\(this\)\);'
        txt = re.search(ptrn, r.text, re.DOTALL).group(1)
        jsn = json.loads(txt)
        df = pd.DataFrame(
                jsn['context']['dispatcher']['stores']
                ['HistoricalPriceStore']['prices']
                )
        df.insert(0, 'symbol', self.symbol)
        df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
        
        # drop rows that aren't prices
        df = df.dropna(subset=['close'])
        
        #df = df[['symbol', 'date', 'high', 'low', 'open', 'close', 
          #       'volume', 'adjclose']]
        df=df[['symbol','date','adjclose']]
        df = df.set_index(df['date'])
        return df
def getinitpars(S,F):
    if (S):
        x0=np.array([1.,0.001,0.002,0.3,0.2,0.5,0.5])
    else:
        if (F):
            x0=np.array([2.,0.9,-0.001,-0.001,0.2,0.2,0.5])
        else:
            x0=np.array([2.5,-0.05,0.3,0.1])
    return x0
def calibrateEM(At,Dt,r,Li,x0,tau,q,smp,cdsP,S,F,method,plotg,B,disp,mrt,bname):
    eps=1.
    x0=getinitpars(S,F)
#    x0=x0[:-1]
    epsv=np.zeros(10)
    for j in range(10):    
        optLM=sc.optimize.least_squares(VGSBlm,x0,method='lm',args=(At,Dt,r,Li,tau1,q,smp,cdsP2,S,mrt),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
        pars=optLM.x
        tmp=pars
        flchfuni=lambda sgm:r+lchfuni(theta,nu,sgm,-1j)
        x0=pars
        nu,theta,sgm,R=pars
        x00=sgm
        sgm=sc.optimize.fsolve(flchfuni,x00)[0]
        x0[2]=sgm
        pars[2]=sgm
        phii=-math.log(1.0-pars[1]*pars[0]-0.5*pars[2]**2*pars[0])*(1.0/pars[0]);
        eps=np.dot((tmp-pars),(tmp-pars).T)
        epsv[j]=eps
    if (plotg):
        plotC(pars,At,Dt,r,Li,tau1,q,smp,cdsP2,S,F,mrt,bname)
    return pars
        
def calibrate(At,Dt,r,Li,tau,q,smp,cdsP,S,F,method,plotg,B,disp,mrt,bname):
    optsCG = {'maxiter' : None, 'gtol' : 1e-4,'norm' : np.inf, 'eps' : 1.4901161193847656e-06,'disp':disp}
    optionsBFGS={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': disp, 'return_all': True}
    optNM={'xtol': 1e-5, 'ftol':1e-5,'disp': disp,'maxfev':10000}
    if (B=='B'):
        x0=getinitpars(S,F)
        if (method=='BS'):
            optBS=sc.optimize.minimize(VGSB,x0,args=(At,Dt,r,Li,tau,q,smp,cdsP,S,mrt),method='BFGS',options=optionsBFGS)
            pars=optBS.x
        elif (method=='NM'):    
            optNM=sc.optimize.minimize(VGSB,x0,args=(At,Dt,r,Li,tau,q,smp,cdsP,S,mrt),method='Nelder-Mead',options=optNM)
            pars=optNM.x
        elif (method=='CG'):    
            optCG=sc.optimize.minimize(VGSB,x0,args=(At,Dt,r,Li,tau,q,smp,cdsP,S,mrt),method='CG',options=optsCG)
            pars=optCG.x
        elif(method=='LM'):
            optLM=sc.optimize.least_squares(VGSBlm,x0,method='lm',args=(At,Dt,r,Li,tau,q,smp,cdsP,S,mrt),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
            pars=optLM.x
        if (plotg):
            plotC(pars,At,Dt,r,Li,tau,q,smp,cdsP,S,F,mrt,bname)
                
    elif(B=='nB'):
        if mrt:
            x0=[2.,0.25,0.35,0.5]
        else:
            x0=[2.,0.25,0.35]
        if (method=='BS'):
            optVGm=sc.optimize.minimize(VGcallF,x0,args=(At,Dt,r,tau,q,cdsP,mrt),method='BFGS',options=optionsBFGS)
        elif (method=='NM'):
            optVGm=sc.optimize.minimize(VGcallF,x0,args=(At,Dt,r,tau,q,cdsP,mrt),method='Nelder-Mead',options=optNM)
        elif(method=='CG'):
            optVGm=sc.optimize.minimize(VGcallF,x0,args=(At,Dt,r,tau,q,cdsP,mrt),method='CG',options=optsCG)
        elif(method=='LM'):
            optVGm=sc.optimize.least_squares(VGcallFlm,x0,method='lm',args=(At,Dt,r,tau,q,cdsP,mrt),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
        
        pars=optVGm.x
        
        if (plotg):
            funv,PD,sp,pars,At=VGcallFo(optVGm.x,At,Dt,r,tau,q,cdsP,mrt);
            plotCnb(pars,At,Dt,r,tau,q,smp,cdsP,mrt)
            #plt.plot(tau,sp,'o');plt.plot(tau,cdsP)
    else:
        x0=[2.,0.25,0.35]
        if (method=='BS'):
            optVG=sc.optimize.minimize(VGcallFopt,x0,args=(At,cdsP['Strikes'],r,tau,q,cdsP['Prems']),method='BFGS',options=optionsBFGS)
            pars=optVG.x
        elif (method=='NM'):
            optVG=sc.optimize.minimize(VGcallFopt,x0,args=(At,cdsP['Strikes'],r,tau,q,cdsP['Prems']),method='Nelder-Mead',options=optNM)
            pars=optVG.x
        elif(method=='CG'):
            optVG=sc.optimize.minimize(VGcallFopt,x0,args=(At,cdsP['Strikes'],r,tau,q,cdsP['Prems']),method='CG',options=optsCG)
            pars=optVG.x
        elif(method=='LM'):
            optVG=sc.optimize.least_squares(VGcallFoptlm,x0,args=(At,cdsP['Strikes'],r,tau,q,cdsP['Prems']),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
            pars=optVG.x
        
        if (plotg):
            Ks=cdsP['Strikes']
            Ps=cdsP['Prems']
            moptP=VGcallFoptout(optVG.x,At,cdsP['Strikes'],r,tau,q);plt.plot(Ks,Ps,'o');plt.plot(Ks,moptP);plt.legend(['Market Price','Model Price']);plt.xlabel('Strike');plt.ylabel('Option Price')
    return pars

def writeParsLtx(parsd,fln):
    import os
    ltxcode=pd.DataFrame(parsd).to_latex()
    text_file = open(fln +'.txt', "w")
    text_file.write(ltxcode)
    text_file.close()
    print("Latex Code is Written to: "+os.getcwd()+" and its name is "+fln+'.txt',end='\n')


def plotCnb(pars,At,Dt,r,tau,q,smp,cdsP,mrt):
    v=VGcallFo(pars,At,Dt,r,tau,q,cdsP,mrt)
    fcds,=plt.plot(tau,cdsP,'ro')
    ffit,=plt.plot(tau,v[2])
    plt.xlabel('Maturity')
    plt.ylabel('CDS Spread')
    plt.legend([fcds,ffit],['Market Values ','Curve Fit'])

    
def plotC(pars,At,Dt,r,Li,tau,q,smp,cdsP,S,F,mrt,bname):
    if (S):
        v=VGSBo(pars,At,Dt,r,Li,tau,q,smp,cdsP,S,mrt)
        fcds,=plt.plot(tau,cdsP,'ro')
        ffit,=plt.plot(tau,v[1])
        plt.xlabel('Maturity')
        plt.ylabel('CDS Spread')
        plt.legend([fcds,ffit],['Market Values ','Curve Fit'])
    else:
        if(F):
            v=VGSFBo(pars,At,Dt,r,Li,tau,q,smp,cdsP,S,mrt)
            fcds,=plt.plot(tau,cdsP,'ro')
            ffit,=plt.plot(tau,v[1])
            plt.xlabel('Maturity')
            plt.ylabel('CDS Spread')
            plt.legend([fcds,ffit],['Market Values ','Curve Fit'])
        else:
            v=VGSBo(pars,At,Dt,r,Li,tau,q,smp,cdsP,S,mrt)
            fcds,=plt.plot(tau,cdsP,'ro')
            ffit,=plt.plot(tau,v[1])
            plt.xlabel('Maturity')
            plt.ylabel('CDS Spread')
            plt.legend([fcds,ffit],['Market Values ','Curve Fit'])
    
    if (F):
        plt.title(bname+' VG-Cox Factor Barrier Model CDS Curve Fit')
    else:
        if (S):
            plt.title(bname+' VG-Cox Stoch. Barrier Model CDS Curve Fit')
        else:
            plt.title(bname+' VG-Cox Barrier Model CDS Curve Fit')
#plt.title('Ford VG-Cox Barrier Model CDS Curve Fit'+' ('+GScds.index[115]+')')

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlt.connect(db_file)
        print(sqlt.version)
    except Error as e:
        print(e)
    finally:
        conn.close()
def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]

def mid(s, offset, amount):
    return s[offset:offset+amount]

def VGMertonNS(initAt,r,Dt,initheta,initsgm,thetaE,sgmE,nuE,tau,q,targetp,prn):
    Att=initAt;
    sgmt=initsgm
    thetat=initheta
    eps=1.;
    iter=0;
    nu=nuE
    try:
        while (eps>1e-5):
            
            iter=iter+1;
            Fvgd1=(VGcall(Att,r,Dt,nu,thetat,sgmt,tau,q)[2]*math.exp(-q*tau))
            At=Att-(VGcall(Att,r,Dt,nu,thetat,sgmt,tau,q)[0]-targetp)/Fvgd1
            Att=At
            sgm=targetp*sgmE/(At*Fvgd1)
            theta=targetp*thetaE/(At*Fvgd1)
            thetat=theta;sgmt=sgm
            eps=abs(VGcall(At,r,Dt,nu,theta,sgm,tau,q)[0]-targetp)
            #sgm=sgmt-((VGcall(At,r,Dt,nu,thetat,sgmt,tau,q)[0]-targetp)/(VGvega(At,r,Dt,nu,thetat,sgmt,tau,q)))
            #sgmt=sgm
            #theta=thetat-((VGcall(At,r,Dt,nu,thetat,sgm,tau,q)[0]-targetp)/(VGdtheta(At,r,Dt,nu,thetat,sgm,tau,q)))
            #thetat=theta
        call=VGcall(At,r,Dt,nu,theta,sgm,tau,q)[0]
        if prn:
            print("Check Error:%0.7f" %(eps))
    except:
        pass;
    return [At,sgm,theta,call]
def VGMertonIter(initAt,r,Dt,initheta,initsgm,thetaE,sgmE,nuE,tau,q,targetp,prn):
    Att=initAt;
    sgmt=initsgm
    thetat=initheta
    eps=1.;
    iter=0;
    nu=nuE
    try:
        while (eps>1e-5):
            
            iter=iter+1;
            Fvgd1=(VGcall(Att,r,Dt,nu,thetat,sgmt,tau,q)[2]*math.exp(-q*tau))
            At=Att-(VGcall(Att,r,Dt,nu,thetat,sgmt,tau,q)[0]-targetp)/Fvgd1
            Att=At            
            eps=abs(VGcall(At,r,Dt,nu,thetat,sgmt,tau,q)[0]-targetp)
        call=VGcall(At,r,Dt,nu,thetat,sgmt,tau,q)[0]
        if prn:
            print("Check Error:%0.7f" %(eps))
    except:
        pass;
    return [At,call]

def VGMertonIterWrap(Et,initAt,r,Dt,initheta,initsgm,thetaE,sgmE,nuE,tau,q,targetp,prn):
    AtV=np.zeros_like(Et)
    N=Et.shape[0]
    eps=1.
    it=0
    #fatV=lambda At:VGcall(At,r,Dt,nuE,initheta,initsgm,tau,q)[0]-targetp;
    while(eps>1e-5):
        lastheta=initheta
        lastsgm=initsgm
        it+=1
        for j in range(N):
            AtV[j]=VGMertonIter(initAt,r,Dt,initheta,initsgm,thetaE,sgmE,nuE,tau,q,Et[j],prn)[0]
            targetp=Et[j]
        initsgm=np.std(np.diff(np.log(AtV)))*np.sqrt(N)
        initheta=(np.mean(np.diff(np.log(AtV)))*N-r)/nuE
        parv=np.array([initsgm,initheta])
        lparv=np.array([lastsgm,lastheta])
        eps=np.linalg.norm(parv-lparv)
        print("No.of Iterations:" +str(it))
    print("Check Error:%0.7f" %(eps))
    return AtV,lparv,eps
    
def VGparlsq(pars,rets,tau,colnum):
    N=np.shape(rets)[0]
    m1=np.mean(rets[:,colnum])
    m2=np.var(rets[:,colnum])
    m3=np.sum((rets[:,colnum]-m1)**3)/(N-1)
    theta=pars[0];sgm=1.0/(1.0+math.exp(-pars[1]));
    #pars[1]#
    nu=pars[2]
    eps=np.zeros(3)
    eps[0]=m1-theta*tau
    eps[1]=m2-(sgm**2+theta**2*nu)*tau
    eps[2]=m3-(3.0*sgm**2*theta*nu+3.0*theta**3*nu**2)*tau
    return np.sqrt(np.dot(eps.T,eps)/3.0)

def getVGretpars(rets,tau,colnum,x0):
    optionsBFGS=optionsBFGS={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': True, 'return_all': True}
    optBS=sc.optimize.minimize(VGparlsq,x0,args=(rets,tau,colnum),method='BFGS',options=optionsBFGS)
    pars=optBS.x[0],1.0/(1.+math.exp(-optBS.x[1])),optBS.x[2]
    return [optBS.fun,pars] 

def getVGvasicek(pars,theta,sgm,nu,cv):
    eps=np.zeros(3)
    rho1=pars[0];rho2=pars[1];rho3=pars[2];thetaZ=pars[3];sgmZ=pars[4];nuZ=pars[5];
    rho12=cv[0,1];rho13=cv[0,2];rho23=cv[1,2]
    V11=theta[0]**2*nu[0]**2+sgm[0]**2;V22=theta[1]**2*nu[1]**2+sgm[1]**2;V33=V11=theta[0]**2*nu[2]**2+sgm[2]**2
    eps[0]=rho12-rho1*rho2*(thetaZ**2*nuZ**2+sgmZ**2)/math.sqrt(V11*V22)
    eps[1]=rho13-rho1*rho3*(thetaZ**2*nuZ**2+sgmZ**2)/math.sqrt(V11*V33)
    eps[2]=rho23-rho2*rho3*(thetaZ**2*nuZ**2+sgmZ**2)/math.sqrt(V22*V33)
    return eps
def optVGvasicek(theta,sgm,nu,cv,x0):
    optLM=sc.optimize.least_squares(getVGvasicek,x0,method='trf',args=(theta,sgm,nu,cv),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
    return optLM
def VGmle(pars,data):
#    mle=0.0
#    N=data.shape[0]
#    for j in range(N):
#        try:
#            dj=np.log(VGdensity(data[j],pars[0],pars[1],pars[2],0.0,1.0,pars[2],1.0))
#            mle+=dj
#        except:
#            dj=0.0
    mle=sum([np.log(VGdensity(el,pars[0],1.0/(1.+math.exp(-pars[1])),pars[2],0.0,1.0,pars[2],1.0)) if el!=0.0 else 0.0 for el in data])
    return -mle
def VGmleopt(data,x0,disp):
    optNM={'xtol': 1e-5, 'ftol':1e-5,'disp': disp,'maxfev':10000}
    out=sc.optimize.minimize(VGmle,x0,args=(data),method='Nelder-Mead',options=optNM)
    pars=out.x[0],1.0/(1.+math.exp(-out.x[1])),out.x[2]
    return [pars,out]

@jit(nopython=True,parallel=True)
def VGnormcdf(X,theta,sgm,g):
    #phii=lchfuni(theta,nu,sgm,-1j)
    #xb=np.dot(X,b.T)
    d1=(X+theta*g)/(sgm*np.sqrt(g))
    #Nd1=scs.norm.cdf(d1)
    #Nd1=(1.0+sc.special.erf(d1/math.sqrt(2.0)))*0.5
    Nd1=appNormCDF(d1)
    #Nd1=normcdff(d1,0.0,1.0)
    #Nd2=scs.norm.cdf(d1-sgm*np.sqrt(g))
    return Nd1
    

@jit(nopython=True,parallel=True)
def VGnormpdf(X,theta,sgm,g):
    d1=(X+theta*g)/(sgm*math.sqrt(g))
    Nd1=appNormPDF(d1)
    return Nd1

def fVGfull(X,theta,sgm,nu,tau):
    GL=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    f=lambda g: VGnormpdf(X,theta,sgm,g)*gampdf(g,tau,nu,nu)
    #gampdf(g,tau,nu,nu)
    vgpdf=GL(f,1e-6,200.)
    return vgpdf
def fVGfullFD(X,theta,sgm,nu,tau):
    h=1e-4
    pdfvg=(FVGfull(X+h,theta,sgm,nu,tau)-FVGfull(X-h,theta,sgm,nu,tau))/(2.*h)
    return pdfvg
def FVGfull(X,theta,sgm,nu,tau):
    #phii=chfuni(theta,nu,sgm,-1j)
    GL=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    #GL=GaussKronrodNonAdaptive(1e-9,10000,1e-9)
    f=lambda g: VGnormcdf(X,theta,sgm,g)*gampdf(g,tau,nu,nu)
    #scs.gamma.pdf(g,tau/nu,0.0,nu)
    #scs.gamma.pdf(g,tau/nu,0.0,nu)
    #gampdf(g,tau,nu,nu)
    #vgcdf=sc.integrate.quad(f,1e-9,np.inf,epsabs=1.49e-09, epsrel=1.49e-06,full_output=1)[0]
    #vgcdf=sc.integrate.fixed_quad(f,0.0,10.)
    vgcdf=sc.integrate.quad(f,0.0,np.inf,epsabs=1.49e-06, epsrel=1.49e-06,full_output=1)
    #GL(f,1e-6,200.)
    
    return vgcdf

def VGvasicekmleC(pars,data,tau,Xs,mth):
    theta=pars[0];sgm=1.0/(1.+math.exp(-pars[1]));nu=pars[2];betas=pars[3:]
    #1.0/(1.+math.exp(-pars[1]))
    N=np.shape(data)[0];eps=np.zeros(N);
    xb=np.dot(Xs,betas.T)
    for j in range(N):
        eps[j]=data[j]-FVGfull(xb[j],theta,sgm,nu,tau)
        
    #pd=[FVGfull(x,theta,sgm,nu,tau) for x in data]
    #eps=pd-data
    #cr=np.dot(eps.T,eps)
    if (mth=='lm'):
        cr=eps
    else:
        cr=np.sqrt(np.dot(eps.T,eps))
    return cr
def VGvasicekPD(pars,tau,Xs):
    theta=pars[0];sgm=1.0/(1.+math.exp(-pars[1]));nu=pars[2];betas=pars[3:]
    #1.0/(1.+math.exp(-pars[1]))
    N=np.shape(Xs)[0];#eps=np.zeros(N);
    pdp=np.zeros(N)
    xb=np.dot(Xs,betas.T)
    for j in range(N):
        #eps[j]=data[j]-FVGfull(xb[j],theta,sgm,nu,tau)
        pdp[j]=FVGfull(xb[j],theta,sgm,nu,tau)
    #pd=[FVGfull(x,theta,sgm,nu,tau) for x in data]
    #eps=pd-data
    #cr=np.dot(eps.T,eps)
    
    return pdp
def VGvasmleopt(data,x0,disp,tau,Xs,mth):
    optNM={'xtol': 1e-5, 'ftol':1e-5,'disp': disp,'maxfev':10000}
    optsCG = {'maxiter' : None, 'disp' : True,'gtol' : 1e-5,'norm' : np.inf, 'eps' : 1.4901161193847656e-07}  # default value.
    if (mth=='lm'):
        out=sc.optimize.least_squares(VGvasicekmleC,x0,args=(data,tau,Xs,mth),method='lm',xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
    elif(mth=='nm'):
        out=sc.optimize.minimize(VGvasicekmleC,x0,args=(data,tau,Xs,mth),method='Nelder-Mead',options=optNM)
    elif(mth=='cg'):
        out=sc.optimize.minimize(VGvasicekmleC,x0,args=(data,tau,Xs,mth),method='CG',options=optsCG )
    else:
        out=sc.optimize.minimize(VGvasicekmleC,x0,args=(data,tau,Xs,mth),method='BFGS',options=optsCG )
    pars=out.x[0],1.0/(1.+math.exp(-out.x[1])),out.x[2],out.x[3:]
    #
    return [pars,out]

def VGvasicekCDF(X,thetaS,sgmS,nuS,thetaZ,sgmZ,nuZ,rho):
    #d=lambda u:
    #thetaS=math.sqrt(1-rho*rho)*thetaS;sgmS=math.sqrt(1-rho*rho)*sgmS
    #thetaZ=rho*thetaZ;sgmZ=math.sqrt((rho*sgmZ)**2)
    f=lambda u:FVGfull((X-rho*u)/(math.sqrt(1-rho*rho)),thetaS,sgmS,nuS,1.0)*VGdensity(u,thetaZ,sgmZ,nuZ,0.0,1.0,nuZ,1.0)
    #fVGfull(u,thetaZ,sgmZ,nuZ,1.0)
    #VGdensity(u,thetaZ,sgmZ,nuZ,0.0,1.0,nuZ,1.0)
    I=sc.integrate.quad(f,0.0,3.0)[0]
    return I

@jit(nopython=True,parallel=True) 
def BMbarrier(At,r,L,sgm,thetaS,tau):
    theta=r-0.5*sgm**2+thetaS
    B=np.log(L/At)
    xL=(B-theta*tau)/(sgm*np.sqrt(tau))
    xR=(B+theta*tau)/(sgm*np.sqrt(tau))
    PD=appNormCDF(xL)+np.exp((2.0*theta*B)/sgm**2)*appNormCDF(xR)
    return PD

@jit(nopython=True,parallel=True) 
def BMbarrierNt(At,r,L,sgm,tau):
    theta=r-0.5*sgm**2
    B=np.log(L/At)
    xL=(B-theta*tau)/(sgm*np.sqrt(tau))
    xR=(B+theta*tau)/(sgm*np.sqrt(tau))
    PD=appNormCDF(xL)+np.exp((2.0*theta*B)/sgm**2)*appNormCDF(xR)
    return PD


@jit(nopython=True,parallel=True) 

def BMbarrierlm(x,At,r,L,tau,cdsP,typ):
    if typ:
        PD=[BMbarrier(At,r,L,x[0],x[1],t) for t in tau]
    else:
        PD=[BMbarrierNt(At,r,L,x[0],t) for t in tau]
    PD=np.array(PD)
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    eps=(sp-np.array(cdsP))
    return eps

def BMbarriero(x,At,r,L,tau,cdsP,plot,typ):
    if typ:
        PD=[BMbarrier(At,r,L,x[0],x[1],t) for t in tau]
    else:
        PD=[BMbarrierNt(At,r,L,x[0],t) for t in tau]
    PD=np.array(PD)
    sp=(-np.log((1.0-PD)+PD*0.5)/tau)*10000
    eps=(sp-np.array(cdsP))
    fn=np.sqrt(np.dot(eps.T,eps))
    if plot:
         
            fcds,=plt.plot(tau,cdsP,'ro')
            ffit,=plt.plot(tau,sp)
            plt.xlabel('Maturity')
            plt.ylabel('PD')
            plt.legend([fcds,ffit],['Market Values ','Curve Fit'])
    return [fn,sp,cdsP,PD]
#np.dot(eps.T,eps)
def calibrateBM(At,r,L,tau,cdsP,typ,pinit):
    if (typ):
        x0=pinit
    else:
        x0=pinit[0]
    optLM=sc.optimize.least_squares(BMbarrierlm,x0,method='lm',args=(At,r,L,tau,cdsP,typ),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
    return optLM.x

#@jit(nopython=True,parallel=True) 
def simvas(lam,mu,rt,sgm,nsim,M,CIR):
    dt=1.0/M
    dWt=np.random.randn(nsim,M)*np.sqrt(dt)
    rtv=np.ones_like(dWt);At=np.ones_like(dWt);Bt=np.ones_like(dWt)
    MMA=np.ones_like(dWt);Pt=np.ones_like(dWt)
    tv=np.cumsum(np.ones_like(dWt)*dt,axis=1);
    #np.cumsum(np.ones_like(dWt)*dt,axis=1);
    #Bt=1.0/np.exp(-lam*tv);At=(Bt-tv)*(mu-sgm**2/(2.0*lam**2))-sgm**2*Bt**2/(4.0*lam)
    rtv[:,0]=rtv[:,0]*rt
    for j in range(1,M):
        #tv[:,j]+=tv[:,j-1]
        if CIR:
            rtv=cytCIRsim(np.array([lam,mu,sgm]),nsim,M,dt,rt)
            rtv=np.array(rtv)
        else:
            rtv[:,j]=rtv[:,j-1]+lam*(mu-rtv[:,j-1])*dt+dWt[:,j]*sgm
        MMA[:,j]=MMA[:,j-1]*np.exp(dt*0.5*rtv[:,j]+rtv[:,j-1])
        Bt[:,j]=1.0/np.exp(-lam*tv[:,j]-dt);
        At[:,j]=(Bt[:,j]-tv[:,j])*(mu-sgm**2/(2.0*lam**2))-sgm**2*Bt[:,j]**2/(4.0*lam)
        Pt[:,j]=np.exp(At[:,j]-Bt[:,j]*rtv[:,j])
    return [rtv,MMA,Pt]
def calcCVA(lam,mu,rt,sgm,nsim,M,PD1,PD2,plot):
    #np.random.seed(123)
    drtv=simvas(lam,mu,rt,sgm,nsim,M,CIR)
    FVm=np.zeros((nsim,M),dtype=np.float)
    CVA=np.zeros((nsim,M),dtype=np.float)
    fr=[(1.0-drtv[2][c][-1:])/np.sum(drtv[2][c]/M) for c in range(nsim)]
    for c in range(nsim):
        FVm[c,:]=[(1.0-drtv[2][c][(M-1)-r])-fr[c]*np.sum(drtv[2][c][r:(M)]/M) if r<(M-1) else 0.0  for r in range(M)]
    FVm=np.maximum(FVm,0.0)
    FVM=np.reshape(FVm,(nsim,M))
    for c in range(nsim):
        CVA[c,:]=[FVm[c,r]/drtv[1][c][r]*pd1*(1-pd2) for r in range(M) for pd1,pd2 in zip(PD1,PD2)]
    if plot:
        plt.figure()
        plt.plot(FVM.T)
        plt.figure()
        for c in range(nsim):
            plt.plot(CVA[c,:])
    
    return [CVA,FVM]
def snG(u,lamdaskew):
    #u=np.real(u)
    dlt=lamdaskew
    x=u*dlt/np.sqrt(1+dlt**2)
    if (np.real(np.log(x))<0.0):
        g=np.exp(-u*u/2.0)*(1.0-sc.integrate.quad(lambda y:np.exp((y*y)/2.0)*np.sqrt(2.0/np.pi),1e-6,np.log(x))[0])
    else:
        g=np.exp(-u*u/2.0)*(1.0+sc.integrate.quad(lambda y:np.exp((y*y)/2.0)*np.sqrt(2.0/np.pi),1e-6,np.log(x))[0])
    return g
@jit(nopython=True,parallel=True)
def cf_svj(St,r,q,vt,lt,u,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump):
    
    
        dlt=lamdaskew
        xj = rhoxv*sgmv*u*1j-kappa
        xsj = rhoxv*sgmv*(1+u*1j)-kappa
        cxj=rhoxl*nu*u*1j-alpha
        csxj=rhoxl*nu*(1+u*1j)-alpha
        gamj=np.sqrt(xj**2+(sgmv**2)*(u**2+1j*u))
        gamcj=np.sqrt(cxj**2+a*(nu**2)*(u**2+1j*u))
        gamsj=np.sqrt(xsj**2+(sgmv**2)*(u**2-1j*u))
        gamcsj=np.sqrt(csxj**2+a*nu**2*(u**2-1j*u))
        xt=np.log(St)
#        if (logskewjump):
#            charj=np.real((lamda*tau)*(snG(u,sgmj)-1.0))
                          #(2.0*np.exp(-u**2/2.0)*scs.norm.cdf(dlt*1j*u/np.sqrt(1+dlt**2)))-1.0)
            #
                  #*scs.norm.cdf(u*dlt/np.sqrt(1+dlt**2)))
#            charsj=np.real((lamda*tau)*(snG(u-1j,sgmj)-1.0))
                           #(2.0*np.exp(-(u-1j)**2/2.0)*scs.norm.cdf(dlt*1j*(u-1j)/np.sqrt(1+dlt**2)))-1.0)
            #
                   #*scs.norm.cdf((u-1j)*dlt/np.sqrt(1+dlt**2)))
#        else:
            
        charsj=(lamda*tau)*(((1+jump)*(1+jump)**(1j*u))*(np.exp((sgmj**2)*(-u**2+1j*u)/2.0)-1.0))
            #(np.exp(np.log(1+jump)*(1j*u+1.0)+(-sgmj**2)*(u**2-1j*u)/2.0)-1.0)   
        charj=(lamda*tau)*((1+jump)**(1j*u)*(np.exp((sgmj**2)*(-u**2-1j*u)/2.0)-1.0))  
            
#        gamj = np.sqrt(np.real(xj**2+sgmv**2*(u**2+u*1j)))
#        cgamj=np.sqrt(np.real(cxj**2+a*beta**2*(u**2+u*1j)))
#        gamsj = np.sqrt(np.real(xj**2+sgmv**2*(u**2-u*1j)))
#        csgamj=np.sqrt(np.real(cxj**2+a*beta**2*(u**2-u*1j)))
                #Aj=-2*(uj*phi*1j - 0.5*phi**2 )/(xj+(gamj*(1+np.exp(gamj*tau))/(1-np.exp(gamj*tau))))
        #Bj=np.real((r-lamda*jump)*1j*phi*tau-(kappa*theta*tau*(xj-gamj)))/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log(1+0.5*(xj-gamj)*(1-np.exp(gamj*tau))/gamj)
        Bj=-(u**2+1j*u)/(-xj+gamj*(np.cosh(gamj*tau/2.0)/np.sinh(gamj*tau/2.0)))
        Bsj=(-u**2+1j*u)/(-xsj+gamsj*(np.cosh(gamsj*tau/2.0)/np.sinh(gamsj*tau/2.0)))
        #Cj=np.real((r-lamda*jump)*1j*phi*tau-(kappa*theta*tau*(xj-gamj)))/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log(1+0.5*(xj-gamj)*(1-np.exp(gamj*tau))/gamj)
        Cj=-a*(u**2+1j*u)/(-xj+gamcj*(np.cosh(gamcj*tau/2.0)/np.sinh(gamcj*tau/2.0)))
        Csj=a*(-u**2+1j*u)/(-xsj+gamcsj*(np.cosh(gamcsj*tau/2.0)/np.sinh(gamcsj*tau/2.0)))
#        if (logskewjump):
#            jump=2.0*(dlt/np.sqrt(1+dlt**2))*scs.norm.pdf(0.0)
#            #np.sqrt(2.0/np.pi)*(dlt/np.sqrt(1+dlt**2))
#            Aj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xj/gamj)*(np.sinh(gamj*tau/2.0))+np.cosh(gamj*tau/2.0))+\
#            tau*alpha*beta*(-cxj)/(nu**2)-(2.0*alpha*beta/nu**2)*np.log((-cxj/gamcj)*(np.sinh(gamcj*tau/2.0))+np.cosh(gamcj*tau/2.0)) 
#             
#            Asj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xsj+gamsj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xsj/gamsj)*(np.sinh(gamsj*tau))+np.cosh(gamsj*tau))+\
#            tau*alpha*beta*(-csxj)/nu**2-(2.0*alpha*beta/nu**2)*np.log((-csxj/gamcsj)*(np.sinh(gamcsj*tau/2.0))+np.cosh(gamcsj*tau/2.0))
#        else:
        Aj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xj/gamj)*(np.sinh(gamj*tau/2.0))+np.cosh(gamj*tau/2.0))+\
        tau*alpha*beta*(-cxj)/(nu**2)-(2.0*alpha*beta/nu**2)*np.log((-cxj/gamcj)*(np.sinh(gamcj*tau/2.0))+np.cosh(gamcj*tau/2.0))              
            
        Asj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xsj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xsj/gamsj)*(np.sinh(gamsj*tau/2.0))+np.cosh(gamsj*tau/2.0))+\
        tau*alpha*beta*(-csxj)/nu**2-(2.0*alpha*beta/nu**2)*np.log((-csxj/gamcsj)*(np.sinh(gamcsj*tau/2.0))+np.cosh(gamcsj*tau/2.0))
        
        
        f2 = np.exp(Aj+Bj*vt+Cj*lt+1j*u*xt+charj)
        f1=np.exp(Asj+Bsj*vt+Csj*lt+1j*u*xt+charsj)
        return [f1,f2,Aj,Asj]

def batesfactoroption(St,K,r,q,vt,lt,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump):

        integ1=sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*np.log(K))*cf_svj(St,r,q,vt,lt,u,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[0]/(1j*u))*(1.0/np.pi),0.0,50.0)
        #sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*np.log(K))*cf_svj(St,r,vt,lt,u+(0.0j+1j*0.0),a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[0]/(1j*u))*(1.0/(1.0*np.pi)),1e-6,500.0)
        integ2=sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*np.log(K))*cf_svj(St,r,q,vt,lt,u,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[1]/(1j*u))*(1.0/np.pi),0.0,50.0)
        #sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*np.log(K))*cf_svj(St,r,vt,lt,u+(1j*0.0),a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[1]/(1j*u))*(1.0/(1.0*np.pi)),1e-6,500.0)
        #P1 = 0.5+(1/np.pi)*sum(np.real(np.exp(-1j*u*np.log(K))*f1/(1j*u))*du);
        P1=integ1[0]+0.5
        P2=integ2[0]+0.5

        #P2 = 0.5+(1/np.pi)*sum(np.real(np.exp(-1j*u*np.log(K))*f2/(1j*u))*du);
        
        Call = np.exp(-q*tau)*St*P1-K*np.exp(-r*tau)*P2
        Put = K*np.exp(-r*tau)*(1.0-min(P2,1.0))-St*(1.0-min(P1,1.0))*np.exp(-q*tau)
        IS=(-1.0/tau)*np.log(1.0-Put/(K*np.exp(-r*tau)))
        return [Put,Call,P1,P2,IS]
def batesfactoroptionPD(St,K,r,q,vt,lt,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump):
    integ2=sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*np.log(K))*cf_svj(St,r,q,vt,lt,u,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[1]/(1j*u))*(1.0/np.pi),0.0,50.0)
        #sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*np.log(K))*cf_svj(St,r,vt,lt,u+(1j*0.0),a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[1]/(1j*u))*(1.0/(1.0*np.pi)),1e-6,500.0)
        #P1 = 0.5+(1/np.pi)*sum(np.real(np.exp(-1j*u*np.log(K))*f1/(1j*u))*du);
    P2=integ2[0]+0.5
    return 1.0-P2
        
@jit(nopython=True,parallel=True) 
def bcf_svj(St,r,q,vt,u,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump):
        xj = rhoxv*sgmv*u*1j-kappa
        xsj = rhoxv*sgmv*(1+u*1j)-kappa
        gamj=np.sqrt(xj**2+(sgmv**2)*(u**2+1j*u))
        gamsj=np.sqrt(xsj**2+(sgmv**2)*(u**2-1j*u))
        xt=np.log(St)
#        if (logskewjump):
#            charj=((lamda*tau)*(snG(u,sgmj)-1.0)).real
#                          #(2.0*np.exp(-u**2/2.0)*scs.norm.cdf(dlt*1j*u/np.sqrt(1+dlt**2)))-1.0)
#            #
#                  #*scs.norm.cdf(u*dlt/np.sqrt(1+dlt**2)))
#            charsj=np.real((lamda*tau)*(snG(u-1j,sgmj)-1.0)).real
                           #(2.0*np.exp(-(u-1j)**2/2.0)*scs.norm.cdf(dlt*1j*(u-1j)/np.sqrt(1+dlt**2)))-1.0)
            #
                   #*scs.norm.cdf((u-1j)*dlt/np.sqrt(1+dlt**2)))
#        else:
        charsj=(lamda*tau)*(((1+jump)*(1+jump)**(1j*u))*(np.exp((sgmj**2)*(-u**2+1j*u)/2.0)-1.0))
            #(np.exp(np.log(1+jump)*(1j*u+1.0)+(-sgmj**2)*(u**2-1j*u)/2.0)-1.0)   
        charj=(lamda*tau)*((1+jump)**(1j*u)*(np.exp((sgmj**2)*(-u**2-1j*u)/2.0)-1.0))
            #(np.exp(np.log(1+jump)*(1j*u)+(-sgmj**2)*(u**2+1j*u)/2.0)-1.0)
            #((1+jump)**(1j*u)*np.exp((sgmj**2)*(-u**2-1j*u)/2.0)-1.0)
            
        Bj=-(u**2+1j*u)/(-xj+gamj*(np.cosh(gamj*tau/2.0)/np.sinh(gamj*tau/2.0)))
        Bsj=(-u**2+1j*u)/(-xsj+gamsj*(np.cosh(gamsj*tau/2.0)/np.sinh(gamsj*tau/2.0)))
#        if (logskewjump):
#            jump=2.0*(sgmj/np.sqrt(1+sgmj**2))*scs.norm.pdf(0.0)
#            #np.sqrt(2.0/np.pi)*(dlt/np.sqrt(1+dlt**2))
#            Aj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xj/gamj)*(np.sinh(gamj*tau/2.0))+np.cosh(gamj*tau/2.0))
#             
#            Asj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xsj+gamsj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xsj/gamsj)*(np.sinh(gamsj*tau))+np.cosh(gamsj*tau))
#        else:
        Aj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xj/gamj)*(np.sinh(gamj*tau/2.0))+np.cosh(gamj*tau/2.0))
            
        Asj=(u*1j)*(r-q-lamda*jump)*tau+tau*kappa*theta*(-xsj)/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log((-xsj/gamsj)*(np.sinh(gamsj*tau/2.0))+np.cosh(gamsj*tau/2.0))
        f2 = np.exp(Aj+Bj*vt+1j*u*xt)*np.exp(charj)
        f1=np.exp(Asj+Bsj*vt+1j*u*xt)*np.exp(charsj)
        return [f1,f2,Aj,Asj]
    
    
#        xj = rho*sgmv*phi*1j-kj
#        xt=np.log(St)
#        gamj = np.sqrt(np.real(xj**2-(2*sgmv**2)*(uj*phi*1j-0.5*phi**2)))
#        Bj=-2*(uj*phi*1j - 0.5*phi**2 )/(xj+(gamj*(1+np.exp(gamj*tau))/(1-np.exp(gamj*tau))))
#        Aj=(r-q-lamda*jump)*1j*phi*tau-(kappa*theta*tau*(xj-gamj))/sgmv**2-(2.0*kappa*theta/sgmv**2)*np.log(1+0.5*(xj-gamj)*(1-np.exp(gamj*tau))/gamj)
#        fj = np.exp(Aj+Bj*vt + 1j*phi*xt+(lamda*tau*((1+jump)**(uj+0.5)*(1+jump)**(1j*phi)*np.exp((sgmj**2)*(uj*1j*phi-0.5*phi**2))-1)))
            
#        return fj
    
def batescall(St,K,r,q,vt,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump,lkappa,ltheta,lsgm,lqt,liq,liqToP):
    
        
        
        #dphi=0.01;
        #dv=0.01
        #maxphi=500;
        #nels=int(maxphi/dphi)
    lt=lqt
        #k1=kappa-rho*sgmv;
        #k2=kappa;
        #phi=np.linspace(0.000001,maxphi,nels).T
        #v=phi
        #x=sqrt(lkappa**2-2*lsgm**2*1j*phi);
        #A2=exp(lkappa**2*ltheta*tau/lsgm**2)/(np.cosh(x*tau*0.5)+lkappa/x*np.sinh(x*tau*0.5))**(2.0*lkappa*ltheta/lsgm**2);
        #B2=2.0*1j*u/(lkappa+x*np.cosh(x*tau*0.5)/np.sinh(x*tau*0.5))
        #l2=A2*exp(-B2*l0);
    if (liq):
        f1=lambda phi: np.real(np.exp(-1j*phi*np.log(K))*bcf_svj(St,r,q,vt,phi,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump)[0]*CIRchf(lkappa,ltheta,lsgm,lt,tau,phi)/(1j*phi))
        f2 = lambda phi: np.real(np.exp(-1j*phi*np.log(K))*bcf_svj(St,r,q,vt,phi,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump)[1]*CIRchf(lkappa,ltheta,lsgm,lt,tau,phi)/(1j*phi))
        integ2=sc.integrate.quad(f2,1e-10,100.0)[0]/(np.pi)
        integ1=sc.integrate.quad(f1,1e-10,100.0)[0]/(np.pi)    
        P1=0.5+integ1
        P2=0.5+integ2
        #P2 = 0.5+((1.0/np.pi**2))*sum(p2);
        if (liqToP):
            Stil=np.real(CIRchf(lkappa,ltheta,lsgm,lt,tau,-1j))*St
            Call = np.exp(-q*tau)*Stil*P1-K*np.exp(-r*tau)*P2
            Put=K*np.exp(-r*tau)*(1.0-P2)-Stil*(1.0-P1)*np.exp(-q*tau)
            IS=(-1.0/tau)*np.log(1.0-Put/(K*np.exp(-r*tau)))
        else:
            Call = np.exp(-q*tau)*St*P1-K*np.exp(-r*tau)*P2
            Put=K*np.exp(-r*tau)*(1.0-P2)-St*(1.0-P1)*np.exp(-q*tau)
            IS=(-1.0/tau)*np.log(1.0-Put/(K*np.exp(-r*tau)))
            
    else:
        f1=lambda phi: np.real(np.exp(-1j*phi*np.log(K))*bcf_svj(St,r,q,vt,phi,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump)[0]/(1j*phi))
        f2 = lambda phi: np.real(np.exp(-1j*phi*np.log(K))*bcf_svj(St,r,q,vt,phi,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump)[1]/(1j*phi))
        integ2=sc.integrate.quad(f2,1e-10,100.0)[0]/(np.pi)
        integ1=sc.integrate.quad(f1,1e-10,100.0)[0]/(np.pi)    
        P1=0.5+integ1
        P2=0.5+integ2
        #P2 = 0.5+((1.0/np.pi**2))*sum(p2);
        Put=K*np.exp(-r*tau)*(1.0-P2)-St*(1.0-P1)*np.exp(-q*tau)
        Call = np.exp(-q*tau)*St*P1-K*np.exp(-r*tau)*P2
        IS=(-1.0/tau)*np.log(1.0-Put/(K*np.exp(-r*tau)))
        #p1=np.real(np.exp(-1j*phi*np.log(K))*f1/(1j*phi))*dphi
        #p2=np.real(np.exp(-1j*phi*np.log(K))*f2/(1j*phi))*dphi
        
        #P1 = 0.5+(1.0/(np.pi**2))*sum(p1);
      
    return [Put,Call,P1,P2,IS]
def coth(x):
    return np.cosh(x)/np.sinh(x)
def CIRchf(kappa,beta,sgmv,Lt,tau,u):
    gamma=np.sqrt(kappa**2-2*sgmv**2*1j*u)
    Atu=np.exp(kappa**2*beta*tau/sgmv**2)/((np.cosh(gamma*tau/2.0)+(kappa/gamma)*np.sinh(gamma*tau/2.0))**(2.0*kappa*beta/sgmv**2))
    Btu=(2.0*1j*u)/(kappa+gamma*coth(gamma*tau/2.0))
    cf=(np.exp(np.log(Atu)-Btu*Lt))
    return cf
def CIRpdf(kappa,beta,sgmv,Lt,lt,tau):  
    pdf=sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*Lt)*CIRchf(kappa,beta,sgmv,lt,tau,u)),1e-6,1000.0,maxp1=100)[0]/(np.pi)
    return pdf
def CIRcdf(kappa,beta,sgmv,Lt,lt,tau,a):
    cdf=np.exp(a*Lt)*sc.integrate.quad(lambda u: np.real(np.exp(-1j*u*Lt)*CIRchf(kappa,beta,sgmv,lt,tau,u+1j*a)/(a-1j*u)),-np.infty,np.infty)[0]/(np.pi*2.0)
    return cdf    

def batesPD(St,K,r,q,vt,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump):    
    #f1=lambda phi: np.real(np.exp(-1j*phi*np.log(K))*bcf_svj(St,r,q,vt,phi,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump)[0]/(1j*phi))
    f2=lambda phi: np.real(np.exp(-1j*phi*np.log(K))*bcf_svj(St,r,q,vt,phi,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump)[1]/(1j*phi))
    integ2=sc.integrate.quad(f2,1e-10,100.0)[0]/(np.pi)
    #integ1=sc.integrate.quad(f1,1e-10,100.0)[0]/(np.pi)    
    #P1=0.5+integ1
    P2=0.5+integ2
    return 1.0-P2
def BatesFactorPD(pars,parsl,lamdaskew,St,K,r,q,logskewjump,cdsP,taus,bates,liq,liqToP,opt):
    if (bates):
        vt,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda=pars
        rhoxv=2.0/(1.0+math.exp(-pars[4]))-1.0
        lkappa,ltheta,lsgm,lqt=parsl
        pd=[batesPD(St,K,r,q,vt,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump) for tau in taus]
            #batescall(St,K,r,q,vt,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump,lkappa,ltheta,lsgm,lqt,liq,liqToP)[4] for tau in taus]
            #batesPD(St,K,r,q,vt,tau,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda,lamdaskew,logskewjump) for tau in taus
    else:
        vt,lt,a,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda=pars
        a=1.0/(1.0+math.exp(-pars[2]))
        pd=[1.0-batesfactoroptionPD(St,K,r,q,vt,lt,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump) for tau in taus]
    #pd=np.array(pd)
    pd=np.array(pd)
    sp=(-np.log((1.0-pd)+pd*0.5)/taus)*10000
    m=np.shape(cdsP)[0]
    eps=(sp-cdsP)
    if opt:
        return eps
    #np.sqrt(np.dot(eps,eps.T))
    else:
        rmse=math.sqrt(np.dot(eps,eps.T))
        return [eps,sp,rmse]
        
    

#np.dot(eps,eps.T)
def caliBatesCDS(pars,St,K,r,q,tau,lamdaskew,logskewjump,cdsP,bates,mth):
    if bates:
        vt,kappa,theta,sgmv,rhoxv,sgmj,jump,lamda=pars
        others=np.ones(4),logskewjump,St,K,r,q,False,cdsP,tau,bates,False,False,True
    else:
        vt,lt,a,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda=pars
        
        others=np.ones(4),logskewjump,St,K,r,q,False,cdsP,tau,bates,False,False,True
    if (mth=='lsq'):
        prs=sc.optimize.least_squares(BatesFactorPD,pars,args=(others),method='lm',xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
    else:
        
        optsCG = {'maxiter' : None, 'gtol' : 1e-4,'norm' : np.inf, 'eps' : 1.4901161193847656e-06,'disp':True}
        optionsBFGS={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': True, 'return_all': True}
        optNM={'xtol': 1e-5, 'ftol':1e-5,'disp': True,'maxfev':10000}
        prs=sc.optimize.minimize(BatesFactorPD,pars,args=(others),method='BFGS',options=optionsBFGS)
    return prs
def CIRpary(yt,dt):
    #cdef struct parset:
     #   double a,b,sgm
    T=np.shape(yt)[0]
    A=np.sum(yt[1:]/yt[:-1]-1.0)
    Sumyt1=np.sum(yt[:-1])
    #Sum2yt1=np.dot(yt[:-1].T,yt[:-1])
    B=(1.0-np.float(T))
    C=np.sum(1.0/yt[:-1])
    D=(yt[-1]-yt[0])
    E=Sumyt1
    F=np.sum(yt[1:]/yt[:-1])
    Btil=C*D*dt+2.0*A*B*dt-B**2*dt-B*F*dt
    #Atil=A*C*dt-C*B*dt-F*C*dt
    G=D*B*dt+A*E*dt
    b=-G/Btil
    #bb=[(-Btil+math.sqrt(Btil**2-4.0*G*Atil))/(2.0*Atil),(-Btil-math.sqrt(Btil**2-4.0*G*Atil))/(2.0*Atil)]
    #if (np.abs(bb[0])<np.abs(bb[1])):
     #   b=bb[0]
    #else:
     #   b=bb[1]
    #a=A/(Sumyt1-np.float(T)*b)
    a=(A/(B+b*C))/dt
    mu=yt[:-1]*(1.0-a*dt)+a*b*dt
    scasgmsq=np.dot((yt[1:]-mu).T,((yt[1:]-mu)/(dt*yt[:-1])))
    sgm=math.sqrt((1.0/(T-1))*scasgmsq)
    return a,b,sgm

