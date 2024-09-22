# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:54:21 2018

@author: Administrator
"""

import scipy.stats as scs
import scipy as sc
import statsmodels.api as sm
from scipy.optimize import fmin_bfgs,fmin_powell,fmin_slsqp
import pandas_datareader as pdrd
import numpy as np
import numexpr as ne
import pandas as pd
import seaborn as sns
from numba import jit
#,autojit,generated_jit
import matplotlib.pyplot as plt
import random as rnd
import math
import cmath
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cython
import os
import re
import sqlite3 as sqlt
os.chdir('C:\\Users\\Administrator\\Desktop\\abdl\\calibre_HCJ')
from runbarrierfuns import *
q=0.0;St=100.0; r=0.05; sgm=0.3; sgmY=0.3;sgmZ=0.3;a=0.6;theta=0.03;thetaY=0.02;thetaZ=0.02;thetaA=0.05;thetaD=0.01;sgmA=0.3;sgmD=0.4;
rhoAD=.5;nu=0.2;nuY=0.2;nuZ=0.1;tau=1.0;At=100.;Dt=100.;smp=0;L=1.0;nsim=10000;steps=252
smp=0;theta=0.001;sgm=0.20722;nu=0.50;q=0.0;r=0.04;B=50.0;At=100;tau=1.;lm=False;smp=False;lamda=0.1
#############################################
BB=False
Sm=np.exp(m)
PDmc=np.sum(np.min(Sm,axis=1)<B)/nsim
############################################################
Smn=np.exp(mn)
Smp=np.exp(mp)
IBp=np.min(Smp,axis=1)<L
IB=0.5*((np.min(Smp,axis=1)<L)*1.0+(np.min(Smn,axis=1)<L)*1.)
#IB=0.5*(IBp+IBn)
PDmc=np.sum(np.min(Smp,axis=1)<L)/nsim
PDmc=np.mean(IB)
###############################################
nu=np.linspace(0.5,0.9,5)
tau=np.linspace(0.0,1.0,10)
smp=True
#VGcallBarrierVGD(At,r,L,nu,theta,sgm,tau,q,smp)
PD=[VGcallBarrierVGD(At,r,L,nun,theta,sgm,tt,q,smp) for nun in nu for tt in tau]
PDm=np.reshape(PD,(nu.size,tau.size))*100.0
plt.plot(tau,PDm.T)
#######################################################################################
cdsP1=np.array([3.,10.,20.,23.,32.])
cdsP2=np.array([75,154,203,225,238])
cdsP4=np.array([73.79,107.44,141.8,162.48,178.53])
cdsP2=np.array([44.23,57.83,89.07,116.37,143.73,164.43,177.15,186.25])
cdsP3=np.array([27.39,40.71,77.00,101.89,125.11,142.28,148.53,151.79])
tau1=[0.5,1.,2.,3.,4.,5.,7.,10.]

#cdsP=GScds.iloc[30,:]
tau2=np.array([1.,3,5.,7.,10.])
#np.array([10,7.,5.,3.,1.])
r=0.04
#Deutschecds=pd.concat((pd.read_csv('dbiCDS1YR.csv',names=['Date',re.findall(r'\d+',"dbiCDS1YR.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                       pd.read_csv('dbiCDS3YR.csv',names=['Date',re.findall(r'\d+',"dbiCDS3YR.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                       pd.read_csv('dbi5YRCDS.csv',names=['Date',re.findall(r'\d+',"dbiCDS5YR.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                        pd.read_csv('dbiCDS7YR.csv',names=['Date',re.findall(r'\d+',"dbiCDS7YR.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                        pd.read_csv('dbiCDS10YR.csv',names=['Date',re.findall(r'\d+',"dbiCDS10YR.csv")[0]+'Yr'],index_col='Date').dropna()),axis=1)
#
#GScds=pd.concat((pd.read_csv('GS1y.csv',names=['Date',re.findall(r'\d+',"GS1y.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                       pd.read_csv('GS3y.csv',names=['Date',re.findall(r'\d+',"GS3y.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                       pd.read_csv('GS5y.csv',names=['Date',re.findall(r'\d+',"GS5y.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                        pd.read_csv('GS7y.csv',names=['Date',re.findall(r'\d+',"GS7y.csv")[0]+'Yr'],index_col='Date').dropna(),\
#                        pd.read_csv('GS10y.csv',names=['Date',re.findall(r'\d+',"GS10y.csv")[0]+'Yr'],index_col='Date').dropna()),axis=1)

#cdsdat1y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='1y')
#cdsdat2y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='2y')
#cdsdat3y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='3y')
#cdsdat4y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='4y')
#cdsdat5y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='5y')
#cdsdat7y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='7y')

#JPdata=pd.read_excel('cds_data.xlsx',index_col='Date',sheetname='jp')
#DBdata=pd.read_excel('cds_data.xlsx',index_col='Date',sheetname='db')
#UBSdata=pd.read_excel('cds_data.xlsx',index_col='Date',sheetname='ubs')
#cdsP1=UBSdata['2018-01-03':'2018-01-03'].values[0]
#cdsP2=DBdata['2018-01-03':'2018-01-03'].values[0]
##cdsdat10y=pd.read_excel('cdsler_full.xlsx',index_col='Date',sheetname='10y')
#cols=cdsdat1y.columns
#cdsTstrc1=pd.concat((cdsdat1y['UBS AG'],cdsdat2y['UBS AG'],\
#                   cdsdat3y['UBS AG'],cdsdat4y['UBS AG'],\
#                   cdsdat5y['UBS AG'],cdsdat7y['UBS AG']),axis=1).dropna()
#cdsTstrc2=pd.concat((cdsdat1y['Citigroup Inc'],cdsdat2y['Citigroup Inc'],\
#                   cdsdat3y['Citigroup Inc'],cdsdat4y['Citigroup Inc'],\
#                   cdsdat5y['Citigroup Inc'],cdsdat7y['Citigroup Inc']),axis=1).dropna()
#
#cdsTstrc3=pd.concat((cdsdat1y[cols[1]],cdsdat2y[cols[1]],\
#                   cdsdat3y[cols[1]],cdsdat4y[cols[1]],\
#                   cdsdat5y[cols[1]],cdsdat7y[cols[1]]),axis=1).dropna()
#cdsP1=cdsTstrc1['2018-01':].iloc[0,:]
#cdsP2=cdsTstrc2['2018-01':].iloc[0,:]
#cdsP3=cdsTstrc3['2018-01':].iloc[0,:]
tau2=np.array([1.,2,3.,4.,5.,7.])
#dat.to_sql('cdstable',confull,if_exists='replace',index_label='id')
sprdat=pd.concat((dat.Date,dat.Ticker,dat.Spread6m,dat.Spread1y,dat.Spread2y,dat.Spread1y,dat.Spread2y,\
                  dat.Spread3y,dat.Spread4y,dat.Spread5y,dat.Spread7y,\
                  dat.Spread10y,dat.Spread15y,dat.Spread20y,dat.Spread30y),axis=1).dropna()
sprdat.to_sql('cdstable',confull,if_exists='replace',index_label='id')
ubsdat=pd.concat((dat[(dat.ShortName=='UBS AG')].Spread6m,dat[(dat.ShortName=='UBS AG')].Spread1y,dat[(dat.ShortName=='UBS AG')].Spread2y,dat[(dat.ShortName=='UBS AG')].Spread3y,dat[(dat.ShortName=='UBS AG')].Spread4y,dat[(dat.ShortName=='UBS AG')].Spread5y,dat[(dat.ShortName=='UBS AG')].Spread7y,dat[(dat.ShortName=='UBS AG')].\
                      Spread10y,dat[(dat.ShortName=='UBS AG')].\
                      Spread15y,dat[(dat.ShortName=='UBS AG')].\
                      Spread20y,dat[(dat.ShortName=='UBS AG')].Spread30y),axis=1).dropna()
dbdat=pd.concat((dat[(dat.ShortName=='DB bk')].Spread6m,dat[(dat.ShortName=='UBS AG')].Spread1y,dat[(dat.ShortName=='UBS AG')].Spread2y,dat[(dat.ShortName=='UBS AG')].Spread3y,dat[(dat.ShortName=='UBS AG')].Spread4y,dat[(dat.ShortName=='UBS AG')].Spread5y,dat[(dat.ShortName=='UBS AG')].Spread7y,dat[(dat.ShortName=='UBS AG')].\
                      Spread10y,dat[(dat.ShortName=='UBS AG')].\
                      Spread15y,dat[(dat.ShortName=='UBS AG')].\
                      Spread20y,dat[(dat.ShortName=='UBS AG')].Spread30y),axis=1).dropna()

#######################################################################################
opt=pd.read_csv('optiondata.txt',sep=' ', header=0)
optd={'Strikes':opt['Strike'][opt['Call/Put']==1],'Prems':opt['Premium'][opt['Call/Put']==1]}

##################################################################
def bp(tau,A):
    return 100.0*np.exp(-A*r*tau)
x=nu,thetaA,thetaD,sgmA,sgmD,rhoAD

#1./(1+math.exp(-x[3])),1./(1+math.exp(-x[4]))

At=100.0;Dt=50.0;q=0.0;Li=50.0;smp=False;lm=False;St=100.0;r=0.032;q=0.021;K=50.;S=False;F=False
r=0.04;q=0.

    
######################################################################################################################################33        
x0=getinitpars(S,F)
x0=np.random.rand(7)
tau3=[0.5,1.,2.,3.,4.,5.,7.,10.,15.,20.,30.];tau=1.
optcal=sc.optimize.minimize(VGcallFopt,x0,args=(At,optd['Strikes'],r,tau,q,optd['Prems']),method='Nelder-Mead',options={'xtol': 1e-5, 'ftol':1e-5,'disp': True,'maxfev':10000})
#optcal=sc.optimize.minimize(VGcallFopt,x0,args=(At,optd['Strikes'],r,tau,q,optd['Prems']),method='Nelder-Mead',options={'xtol': 1e-5, 'ftol':1e-5,'disp': True,'maxfev':10000})
cdsP=np.double(calibdat.iloc[0,2:].values*100)
opt=sc.optimize.least_squares(VGSBlm,x0,method='lm',args=(At,Dt,r,Li,tau1,q,smp,cdsP3,S,mrt),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
optnm=sc.optimize.minimize(VGSBlm,x0,args=(At,Dt,r,Li,tau1,q,smp,cdsP3,S,mrt),method='Nelder-Mead',options={'xtol': 1e-5, 'ftol':1e-5,'disp': True,'maxfev':10000})


fusDBb=VGSBo(pars1,At,Dt,r,Li,tau1,q,smp,cdsP2,S,mrt)
fusDB=VGcallFo(pars3,At,Dt,r,tau1,q,cdsP2,mrt)
fusENIb=VGSBo(pars2,At,Dt,r,Li,tau1,q,smp,cdsP3,S,mrt)
fusENI=VGcallFo(pars4,At,Dt,r,tau1,q,cdsP3,mrt)


dctfus={'RMSEnomerton %':[fusDBb[0],fusENIb[0]],'RMSEmerton %':[fusDBmrt[0],fusENImrt[0]]}
writeParsLtx(dctfus,'VGrmse')
parsd={'Model CDS Spread':fusDBmrt[1],'Market CDS Spread':cdsP2}
writeParsLtx(parsd,'marketvsmodelCDSd_mrt')
parsd={'Model CDS Spread':fusENImrt[1],'Market CDS Spread':cdsP3}
writeParsLtx(parsd,'marketvsmodelCDSen_imrt')
########################################################################################################
At=100.0;Dt=50.;Li=50.;smp=True;lm=False;mrt=False;St=100.0;r=0.032;q=0.021;K=50.;S=False;F=False;tau=1.0
r=0.0045;q=0.0
x0=[2.,0.25,0.35]
x0=[-0.3,-0.01,1.5,1.3,0.2,0.25]
tau2=[1.,3.,5.,7.,10.]
optNM={'xtol': 1e-5, 'ftol':1e-5,'disp': True,'maxfev':10000};bname='UBS';mrt=False;S=False
pars1=calibrate(At,Dt,r,Li,tau1,q,smp,cdsP2,S,F,'LM',True,'B',True,mrt,'DB')
pars2=calibrate(At,Dt,r,Li,tau1,q,smp,cdsP3,S,F,'LM',True,'B',True,mrt,'ENI')
pars3=calibrate(At,Dt,r,Li,tau1,q,smp,cdsP2,S,F,'LM',True,'nB',True,mrt,'DB')
pars4=calibrate(At,Dt,r,Li,tau1,q,smp,cdsP3,S,F,'LM',True,'nB',True,mrt,'ENI')
At=113.76;r=0.0045;q=0.0018;tau=1.
pars3=calibrate(At,Dt,r,Li,tau,q,smp,optd,S,F,'LM',True,'opt',True,mrt,'Brent')
##############################################################################################################################
###################################################################################################################
days=range(1,28)
months=range(1,12)
smonths=[right('0'+str(m),2) for m in months]
sdays=[right('0'+str(d),2) for d in days]
wmonths=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#wmonths=['Aug']
confull=sqlt.connect('cdsql.db');
conn=sqlt.connect('pars.db');
conoAs=sqlt.connect('parsnoAs.db');
conaug=sqlt.connect('cdsql2008.db');
conaug.commit()
for sday in sdays:
    for month in wmonths:
        dirc='C:\\Users\\Administrator\\Desktop\\abdl\\calibre_HCJ\\CDS_08\\CDS_08-'+sday+month+'08.csv'
        try:
            dat=pd.read_csv(dirc,sep=",",header=1);dat.to_sql('cdsql2008',conaug,if_exists='append')
            del dat;
        except:
            pass
        
#create_connection('cdsql.db')
#sadece float64 datatype çekiyor
dt.query('Ticker=="DB"').select_dtypes(include=["float64"]).iloc[10,:].plot()

sqltxt=""" SELECT  Spread6m, Spread1y,Spread2y,Spread3y,Spread4y,Spread5y,Spread7y,Spread10y FROM cdstable WHERE (ticker like 'UBS%'and Ccy like 'USD%' and Tier like 'SNR%' and DocClause like 'CR')
OR (ticker like 'DB'  and Ccy like 'USD%' and Tier like 'SNR%' and DocClause like 'CR'); """

sqltxtkillperc= """SELECT Ticker,Date, cast(replace(Spread6m,'%','') as float) as Spread6m, cast(replace(Spread1y,'%','') as float) as Spread1y, cast(replace(Spread2y,'%','') as float) as Spread2y,cast(replace(Spread3y,'%','') as float)as Spread3y,cast(replace(Spread4y,'%','') as float) as Spread4y,cast(replace(Spread5y,'%','') as float) as Spread5y,cast(replace(Spread7y,'%','') as float) as Spread7y,cast(replace(Spread10y,'%','') as float) as Spread10y FROM cdsql2008 WHERE (ticker like 'DB%'and Ccy like 'USD%' and Tier like 'SNR%' and DocClause like 'CR' and Date like '%DEC%')
OR (ticker like 'UBS%'  and Ccy like 'USD%' and Tier like 'SNR%' and DocClause like 'CR'and Date like '%DEC%');"""

sqltxtkillperc2= """SELECT Ticker,Date, cast(replace(Spread6m,'%','') as float) as Spread6m, cast(replace(Spread1y,'%','') as float) as Spread1y, cast(replace(Spread2y,'%','') as float) as Spread2y,cast(replace(Spread3y,'%','') as float)as Spread3y,cast(replace(Spread4y,'%','') as float) as Spread4y,cast(replace(Spread5y,'%','') as float) as Spread5y,cast(replace(Spread7y,'%','') as float) as Spread7y,cast(replace(Spread10y,'%','') as float) as Spread10y FROM cdstableaug WHERE (Ccy like 'USD%' and Tier like 'SNR%' and DocClause like 'CR');"""

sqltxt=""" SELECT  Date,Ticker,cast(replace(Spread6m,'%','') as float) as Spread6m, cast(replace(Spread1y,'%','') as float) as Spread1y, cast(replace(Spread2y,'%','') as float) as Spread2y,cast(replace(Spread3y,'%','') as float)as Spread3y,cast(replace(Spread4y,'%','') as float) as Spread4y,cast(replace(Spread5y,'%','') as float) as Spread5y,cast(replace(Spread7y,'%','') as float) as Spread7y,cast(replace(Spread10y,'%','') as float) as Spread10y FROM cdstable 
WHERE (Ticker like 'GE' and Ccy like 'USD%')
ORDER BY Date; """
 #and Tier like 'SNR%' and DocClause like 'CR'
cdstick=pd.read_clipboard(header=None);
cdsdata=pd.DataFrame();tmp=cdsdata=pd.DataFrame();
for j in range(cdstick.shape[0]):
    sqltxt=""" SELECT  Date,Ticker,cast(replace(Spread6m,'%','') as float) as Spread6m, cast(replace(Spread1y,'%','') as float) as Spread1y, cast(replace(Spread2y,'%','') as float) as Spread2y,cast(replace(Spread3y,'%','') as float)as Spread3y,cast(replace(Spread4y,'%','') as float) as Spread4y,cast(replace(Spread5y,'%','') as float) as Spread5y,cast(replace(Spread7y,'%','') as float) as Spread7y,cast(replace(Spread10y,'%','') as float) as Spread10y FROM cdstable 
    WHERE (Ticker like '""" +cdstick[0][j]+ """' and Ccy like 'USD%')
    ORDER BY Date; """
    print(sqltxt)
    tmp=pd.read_sql(sqltxt,confull);
    cdsdata=pd.concat([cdsdata,tmp],axis=0)
cdsdata.to_csv('cdsdata.csv')
sqltxt2=""" SELECT  * from params"""
calibdat=pd.read_sql(sqltxtkillperc,confull);calibdat=calibdat[calibdat.Ticker=='DB']
paramsql=pd.read_sql(sqltxt2,conn)
#calibdat.Spread6m=calibdat.Spread6m.astype(float)
tau3=[0.5,1.,2.,3.,4.,5.,7.,10.];plotg=False;B='B';disp=True;mrt=True;method='LM';bname='DB'
calibratedata=calibdat.sort_values(by=['Date'])
cdsP=np.double(calibdat.iloc[:,2:].values*100.)
cdsP=np.vstack((cdsP2,cdsP3))
def getcalpars(At,Dt,r,Li,tau3,q,smp,cdsP,S,F,method,plotg,B,disp,mrt,bname):
    try:
        parsd=[calibrate(At,Dt,r,Li,tau3,q,smp,el,S,F,method,plotg,B,disp,mrt,bname) for el in cdsP]
        parsd=np.array(parsd)
        return parsd         
    except:
        pass
dts=pd.to_datetime(calibdat.Date)

parsd=getcalpars(At,Dt,r,Li,tau3,q,smp,cdsP,S,F,method,plotg,B,disp,mrt,bname)
fns=[VGSBo(np.array(parsde),At,Dt,r,Li,tau3,q,smp,el,S,mrt)[0] for el,parsde in zip(cdsP,parsd)]

if mrt:
    As=[VGSBo(np.array(parsde),At,Dt,r,Li,tau3,q,smp,el,S,mrt)[2][4] for el,parsde in zip(cdsP,parsd)]
    PDs=[VGSBo(np.array(parsde),Ass,Dt,r,Li,tau3,q,smp,el,S,mrt)[2][5] for el,parsde,Ass in zip(cdsP,parsd,As)]
else:
    PDs=[VGSBo(np.array(parsde),At,Dt,r,Li,tau3,q,smp,el,S,mrt)[2][3] for el,parsde in zip(cdsP,parsd)]
plt.figure()
dfpars=pd.DataFrame(parsd,dts)
dfpars.columns=['nu','theta','sigma','pAs']
dfpars.iloc[:,:-1].plot()
plt.figure()
[plt.plot(tau3,pltel) for pltel in PDs]
dfAs=pd.DataFrame(As,index=dts);dfAs.columns=['Implied Asset Values'];dfAs.plot()
dfpars.nu.plot();dfparsnmrt.nu.plot()
plt.legend(['Merton','NoMerton']);plt.title(r"Estimated $\nu$ Parameters")
#parameter to SQL
As=np.array(As)
dctpars={'Ticker':calibdat.iloc[:,0].values,'nu':parsdmer[:,0],'theta':parsdmer[:,1],'sigma':parsdmer[:,2],'As':As};
dctAs={'Ticker':calibdat.iloc[0:20,0].values,'DB Implied Asset Values':As[0:20]};dfAs=pd.DataFrame(dctAs,dtsdec);dfAs.plot()
dctfus={'Assets':As,'nu':prsfus[:,0],'theta':prsfus[:,1],'sigma':prsfus[:,2]};
dfpars=pd.DataFrame(dctpars);dfpars.to_sql('params',conn,if_exists='replace');
pd.read_sql(sqltxt2,conn)[:][:1]
#prediction
prdsnmrt=[VGSBo(np.array(parsde),At,Dt,r,Li,tau3,q,smp,el,S,False)[1] for el,parsde in zip(cdsP,prsdnm)]
prdsmrt=[VGSBo(np.array(parsde),Ass,Dt,r,Li,tau3,q,smp,el,S,True)[1] for el,parsde,Ass in zip(cdsP,parsdm,As)]
prdsnmrt=np.array(prdsnmrt)
prdsmrt=np.array(prdsmrt)
dctprds={'0_5y':prds[:,0],'1y':prds[:,1],'2y':prds[:,2]};conn=sqlt.connect('pars.db')
#################################### VG MODEL vs BM MODEL #######################################################################################################
bmp1=calibrateBM(At,r,Li,tau1,cdsP2,True,x);ltau=np.linspace(0.01,1.0,20);L=Li;smp=False
bmp2=calibrateBM(At,r,Li,tau1,cdsP3,True,x)
sPDbm1=[BMbarrier(At,r,L,bmp1[0],bmp1[1],tt) for tt in ltau]
sPDbm2=[BMbarrier(At,r,L,bmp2[0],bmp2[1],tt) for tt in ltau]
sPD1=[VGcallBarrierVGD(At,r,L,pars1[0],pars1[1],pars1[2],tt,0.0,False) for tt in ltau]
sPD2=[VGcallBarrierVGD(At,r,L,pars2[0],pars2[1],pars2[2],tt,0.0,False) for tt in ltau]
L=1.0
sPD1S=[VGcallSBarrierVGD(At,Dt,r,L,pars1[0],pars1[1],pars1[2],1./(1+math.exp(-pars1[3])),1./(1+math.exp(-pars1[4])),2.0*(1./(1+math.exp(-pars1[5])))-1.,tt,q,smp) for tt in ltau]
sPD2S=[VGcallSBarrierVGD(At,Dt,r,L,pars2[0],pars2[1],pars2[2],1./(1+math.exp(-pars2[3])),1./(1+math.exp(-pars2[4])),2.0*(1./(1+math.exp(-pars2[5])))-1.,tt,q,smp) for tt in ltau]

plt.plot(ltau,sPD1,'-o',color='r')
plt.plot(ltau,sPDbm1,'-o')
plt.plot(ltau,sPD1S,'+',color='b')
plt.figure()
plt.plot(ltau,sPDbm2,'-o')
plt.plot(ltau,sPD2,'-o',color='r')
plt.plot(ltau,sPD2S,'+',color='b')
plt.legend(['PDbrowniamotion','PDVarianceGamma'])
##########################################CVA Stres Through Parameters############################################################################################
pd0=[VGcallfactorBarrierVGD(At,r,Li,nuY1,nuZ,thetaY1,thetaZ,sgmY1,sgmZ,aS[0],tt,q,smp) for tt in ltau]
pd1=[VGcallfactorBarrierVGD(At,r,Li,nuY2,nuZ,thetaY2,thetaZ,sgmY2,sgmZ,aS[1],tt,q,smp) for tt in ltau]
cp=[VGfacall(St,r,K,thetaY3,thetaZ,nuY3,nuZ,sgmY3,sgmZ,tt,q,aS[2])[0] for tt in ltau]
cvastr=[(1.0-el1)*el2*el3 for el1,el2,el3 in zip(pd0,pd1,cp)]
cva=[(1.0-el1)*el2*el3 for el1,el2,el3 in zip(pd0,pd1,cp)]
plt.plot(ltau,cva);plt.plot(ltau,cvastr,'--',color="r")
plt.legend(['Normal Condition','Stres Condition']);plt.title("CVA Scenario Analysis")
########################################################################################################################################
fgs=[plotC(np.array(parsde),At,Dt,r,Li,tau3,q,smp,calibdat.iloc[el,:]*100,S,F) for el,parsde in zip(range(calibdat.shape[0]),parsd)]
plt.plot(np.reshape(np.repeat(np.array(parsd).mean(axis=0),repeats=13),(3,13)).T)
plt.plot(np.array(parsd),'+')
parsdnb=[calibrate(At,Dt,r,Li,tau3[:-3],q,smp,ubsdat.iloc[el,:-3]*10000.,S,F,'LM',True,'nB',True) for el in range(ubsdat.shape[0])]
###########################################################################################################
q=0.0016;r=0.0045;tau=1.;Ft=113.76;
pars3=calibrate(Ft,Dt,r,Li,tau,q,smp,optd,S,F,'NM',True,'opt',True)
#pars3=5.0/(1.0+math.exp(-mpars3[0])),pars3[1],1.0/(1+math.exp(-pars3[2]))
moptP=VGcallFoptout(pars3,Ft,optd['Strikes'],r,tau,q)
###########################################################################################

#############################################################################################
K = np.linspace(10.,99.,25)
K2=np.linspace(10.,99.,25)
K3=100.
Dt = np.linspace(10.,99.,25)
taus=np.linspace(1,20.,25)

if (S):
    nu=opt.x[0];thetaA=opt.x[1];thetaD=opt.x[2];sgmA=1.0/(1+math.exp(-opt.x[3]));sgmD=1.0/(1+math.exp(-opt.x[4]));rhoAD=2.0/(1+math.exp(-opt.x[5]))-1.0
else:
    if(F):
        nuY=optF.x[0];nuZ=optF.x[1];thetaY=optF.x[2];thetaZ=optF.x[3];sgmY=1.0/(1+math.exp(-optF.x[4]));sgmZ=1.0/(1+math.exp(-optF.x[5]));a=optF.x[6]
    else:    
        nu=opt.x[0];theta=opt.x[1];sgm=opt.x[2]



pars1=(nuY,nuZ,thetaY,thetaZ,sgmY,sgmZ,a)
pars=(nuY,nuZ,thetaY,thetaZ,sgmY,sgmZ,a)
(nu,thetaA,thetaD,sgmA,sgmD,rhoAD)
parsS1=(nu,theta,sgm)
parsS2=(nu,theta,sgm)
#################################PLAIN BARRIER #################################################################################    

VGcallSBout=[VGcallSBarrierVGD(At,DD,r,L,nu,thetaA,thetaD,sgmA,sgmD,rhoAD,tt,q,smp) for DD in Dt for tt in taus]
VGcallPDSB=np.reshape(VGcallSBout,(Dt.size,taus.size))*100.0
sp=(-np.log((1.0-0.01*VGcallPDSB)+0.01*VGcallPDSB*0.5)/taus)*10000
VGcallPDSSB=np.reshape(sp,(Dt.size,taus.size))
makesurf(VGcallPDSB,False,Dt,taus,True,False,False,'Maturity','VG-Cox Stochastic Barrier PD')
makesurf(VGcallPDSSB,False,Dt,taus,True,False,False,'Maturity','VG-Cox Stochastic Barrier CDS Spread')

#################################STOCHASTIC BARRIER########################

def makeallPD(St,r,K,q,taus,smp,pars1,pars2,srf,paramD,sprf):
    nu1,theta1,sgm1=pars1[:3]
    nu2,theta2,sgm2=pars2[:3]
    nu3,theta3,sgm3=pars3[:3]
    
    nuY1,thetaY1,sgmY1,nuZ,thetaZ,sgmZ=paramD['nuS'][0],paramD['thetaS'][0],paramD['sigmaS'][0],paramD['nuZ,thetaZ,sigmaZ'][0],paramD['nuZ,thetaZ,sigmaZ'][1],paramD['nuZ,thetaZ,sigmaZ'][2]
    nuY2,thetaY2,sgmY2,nuZ,thetaZ,sgmZ=paramD['nuS'][1],paramD['thetaS'][1],paramD['sigmaS'][1],paramD['nuZ,thetaZ,sigmaZ'][0],paramD['nuZ,thetaZ,sigmaZ'][1],paramD['nuZ,thetaZ,sigmaZ'][2]
    nuY3,thetaY3,sgmY3,nuZ,thetaZ,sgmZ=paramD['nuS'][2],paramD['thetaS'][2],paramD['sigmaS'][2],paramD['nuZ,thetaZ,sigmaZ'][0],paramD['nuZ,thetaZ,sigmaZ'][1],paramD['nuZ,thetaZ,sigmaZ'][2]
    
    VGcallBout1=[VGcallBarrierVGD(St,r,Kk,nu1,theta1,sgm1,tt,q,smp) for Kk in K for tt in taus]
    VGcallPDB1=np.reshape(VGcallBout1,(K.size,taus.size))*100.0
    sp11=(-np.log((1.0-0.01*VGcallPDB1)+0.01*VGcallPDB1*0.5)/taus)*10000
    
    
    VGcallBout2=[VGcallBarrierVGD(St,r,Kk,nu2,theta2,sgm2,tt,q,smp) for Kk in K for tt in taus]
    VGcallPDB2=np.reshape(VGcallBout2,(K.size,taus.size))*100.0
    sp12=(-np.log((1.0-0.01*VGcallPDB2)+0.01*VGcallPDB2*0.5)/taus)*10000
    
#    VGcallBout3=[VGcallBarrierVGD(St,r,Kk,nu3,theta3,sgm3,tt,q,smp) for Kk in K for tt in taus]
#    VGcallPDB3=np.reshape(VGcallBout3,(K.size,taus.size))*100.0
#    sp13=(-np.log((1.0-0.01*VGcallPDB3)+0.01*VGcallPDB3*0.5)/taus)*10000
    PD1=[VGcallfactorBarrierVGD(At,r,Kk,nuY1,nuZ,thetaY1,thetaZ,sgmY1,sgmZ,a1,tt,q,smp) for Kk in K for tt in taus]
    PD1m=np.reshape(PD1,(K.size,taus.size))*100.0
    PD2=[VGcallfactorBarrierVGD(At,r,Kk,nuY2,nuZ,thetaY2,thetaZ,sgmY2,sgmZ,a2,tt,q,smp) for Kk in K for tt in taus]
    PD2m=np.reshape(PD2,(K.size,taus.size))*100.0
#    PD3=[VGcallfactorBarrierVGD(At,r,Kk,nuY3,nuZ,thetaY3,thetaZ,sgmY3,sgmZ,a3,tt,q,smp) for Kk in K for tt in taus]
#    PD3m=np.reshape(PD3,(K.size,taus.size))*100.0
    #VGcallm=[VGfacall(St,r,Kk,thetaY,thetaZ,nuY,nuZ,sgmY,sgmZ,tt,q,a)[0] for Kk in K for tt in taus]
    sp1=(-np.log((1.0-0.01*PD1m)+0.01*PD1m*0.5)/taus)*10000
    sp2=(-np.log((1.0-0.01*PD2m)+0.01*PD2m*0.5)/taus)*10000
#    sp3=(-np.log((1.0-0.01*PD3m)+0.01*PD3m*0.5)/taus)*10000
    
    if srf:
        makesurf(VGcallPDB1,False,K,taus,True,False,True,'Maturity','PD Curve')
        makesurf(PD1m,False,K,taus,True,False,True,'Maturity','PD Curve')
        makesurf(VGcallPDB2,False,K,taus,True,False,True,'Maturity','PD Curve')
        makesurf(PD2m,False,K,taus,True,False,True,'Maturity','PD Curve')
#        makesurf(VGcallPDB3,False,K,taus,True,False,True,'Maturity','PD Curve')
#        makesurf(PD3m,False,K,taus,True,False,True,'Maturity','PD Curve')
        if sprf:
            makesurf(sp11,False,K,taus,True,False,True,'Maturity','CDS Spread Curve')
            makesurf(sp1,False,K,taus,True,False,True,'Maturity','CDS Spread Curve')
            makesurf(sp12,False,K,taus,True,False,True,'Maturity','CDS Spread Curve')
            makesurf(sp2,False,K,taus,True,False,True,'Maturity','CDS Spread Curve')
#            makesurf(sp13,False,K,taus,True,False,True,'Maturity','CDS Spread Curve')
#            makesurf(sp3,False,K,taus,True,False,True,'Maturity','CDS Spread Curve')
    ############################FACTOR PARS####################################
    
#    lchfuni(theta3,nu3,sgm3,-1j),lchfuni(thetaZ,nuZ,sgmZ,-1j*a3)+lchfuni(thetaY3,nuY3,sgmY3,-1j)
#    lchfuni(theta2,nu2,sgm2,-1j),lchfuni(thetaZ,nuZ,sgmZ,-1j*a2)+lchfuni(thetaY2,nuY2,sgmY2,-1j)
#    lchfuni(theta1,nu1,sgm1,-1j),lchfuni(thetaZ,nuZ,sgmZ,-1j*a1)+lchfuni(thetaY1,nuY1,sgmY1,-1j)
    
############################FACTOR BARRIER####################################

    
#    print(sc.linalg.norm(PD3m-VGcallPDB3))
    print(sc.linalg.norm(PD2m-VGcallPDB2))
    print(sc.linalg.norm(PD1m-VGcallPDB1))
    return [VGcallPDB1,VGcallPDB2,sp11,sp12,PD1m,PD2m,sp1,sp2]
srf=True;sprf=True
mm=makeallPD(St,r,K,q,taus,smp,pars1,pars2,srf,out[1],sprf)
##############################################################################
vgc=VGfacall(At,r,K3,thetaY3,thetaZ,nuY3,nuZ,sgmY3,sgmZ,1.0,q,aS[2])[0]
CVA=[(1.0-pd1*0.01)*pd2*0.01*vgc for pd1,pd2 in zip(PD1,PD2,VGcallm)]
CVAm=np.reshape(CVA,(K.size,taus.size))*100.0
makesurf(CVAm,False,K,taus,True,False,True,'Maturity','CVA Curve')
#####################################CVA Calculate##################################################
#nuY1,nuZ1,thetaY1,thetaZ1,sgmY1,sgmZ1,a1=pars1
#K=50.
tau=1.;r=0.032;q=0.021
PD1=[VGcallfactorBarrierVGD(At,r,Kk,nuY1,nuZ,thetaY1,thetaZ,sgmY1,sgmZ,aS[0],tau,q,smp) for Kk in K]
#PD1m=np.reshape(PD1,(K.size,taus.size))*100.0
PD2=[VGcallfactorBarrierVGD(At,r,Kk,nuY2,nuZ,thetaY2,thetaZ,sgmY2,sgmZ,aS[1],tau,q,smp) for Kk in K]
#PD2m=np.reshape(PD2,(K.size,taus.size))*100.0
Ato=113.76;r=0.0045;q=0.0018
vgc=VGfacall(Ato,r,K3,thetaY3,thetaZ,nuY3,nuZ,sgmY3,sgmZ,1.0,q,aS[2])[0]
CVA=[(1.0-pd1)*pd2*vgc for pd1 in PD1 for pd2 in PD2]
CVAm=np.reshape(CVA,(K.size,K2.size))
makesurf(CVAm,False,K,K2,True,False,False,'Debt Barrier','CVA Curve')
##########################################################################################
def BatesFactorPD(vt,lt,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,St,K,r,q,logskewjump,cdsP,taus):
    pd=[1.0-batesfactoroption(St,K,r,q,vt,lt,a,tau,kappa,theta,sgmv,alpha,beta,nu,rhoxv,rhoxl,sgmj,jump,lamda,lamdaskew,logskewjump)[3] for tau in taus]
    sp=-(np.log(pd*0.5+1.0-pd))/taus
    eps=sp-cdsP
    return eps




#######################################################################################
############################## VGMONTE CARLO SIMULATION ##################################
    #np.cumsum(np.random.randn((nsim,M))*np.sqrt(dt),axis=1)

taus=[0.1,0.25,0.5,0.75,1.,2.5,5.,7.5,10.,12.5,15,17.5,20.]
#np.linspace(0.0001,50.,30)    
PDec=[PDminBM(r,sgm,At,L,tau) for tau in taus]
PDc=[simaxBM(nsim,M,tt,mx,r,sgm,At,L) for tt in taus]
BrMC,=plt.plot(taus,PDc,'+')
BrE,=plt.plot(taus,PDec)
plt.xlabel('Maturity')
plt.ylabel('PD')
plt.legend([BrMC,BrE],['BMotionMC','BMotionExact'])
plt.title('BMotion Barrier Credit Risk Model MC Simulation vs Exact Formula')

#########################################################################################
q=0.00;r=0.0421;sgm=0.242;At=100.;L=50.;tau=1.;T=1.;theta1=-0.25;sgm1=0.20721;nu1=0.6
nsim=10000;M=252;mx=False;lamda=0
#theta=[-0.4,-0.25,-0.1,0.0,0.001];
nu=0.5;lamda=0.0;
#q=0.0;
lm=False;smp=False;vgo=[];vgd=[];mx=False
taus=[0.1,0.25,0.5,0.75,1.,2.5,5.,7.5,10.,12.5,15,17.5,20.]
#,22.5,25.,27.5,30.
nsimm=np.ones_like(taus)*nsim
#np.linspace(0.1,20.,10)
bs=[];vgd=[];vgo=[]
bs=[VGmx(tau,nu1)[0] for j in range(nsim) for tau in taus]
#bs=[VGmx(tau,nu)[0] for ns,tau in zip(nsimm,taus)]
bsm=np.reshape(bs,(nsim,np.array(taus).size))

for col in range(bsm.shape[1]):
    vgo.append(genwVG(At,L,r,q,theta1,sgm1,nu1,taus[col],nsim,bsm[:,col],False)[0])
    vgd.append(VGcallBarrierVGD(At,r,L,nu1,theta1,sgm1,taus[col],q,smp))
plt.figure(0)    
vgsimp,=plt.plot(taus,vgo,'+')
vgep,=plt.plot(taus,vgd)
plt.xlabel('Maturity')
plt.ylabel('PD')
plt.legend([vgsimp,vgep],['VGMC','VGExact'])
plt.title('VG Credit Risk Model MC Simulation vs Exact Formula')
###############################################
plt.figure(1)
spex=(-np.log((1.0-np.array(vgd))+np.array(vgd)*0.5)/np.array(taus))*10000
spsim=(-np.log((1.0-np.array(vgo))+np.array(vgo)*0.5)/np.array(taus))*10000
cdsPsim,=plt.plot(taus,spsim,'+')
cdsPex,=plt.plot(taus,spex)
plt.xlabel('Maturity')
plt.ylabel('CDS Spread')
plt.legend([vgsimp,vgep],['VGMC','VGExact'])
plt.title('VG Credit Risk Model CDS MC Simulation vs Exact Formula')

###################################################################################################
taus=[0.25,0.5,0.75,1.,2.5,5.,7.5,10.,12.5,15,17.5,20.,22.5,25.,27.5,30.];mx=False
#np.linspace(0.1,20.,10)
bs=[VGmx(tau,nu)[0] for j in range(nsim) for tau in taus]
bsm=np.reshape(bs,(nsim,np.array(taus).size))

for col in range(bsm.shape[1]):
    vgo.append(genwVGSB(At,Dt,L,r,thetaA,thetaD,sgmA,sgmD,rhoAD,nu,taus[col],nsim,bsm[:,col]))
    vgd.append(VGcallSBarrierVGD(At,Dt,r,L,nu,thetaA,thetaD,sgmA,sgmD,rhoAD,taus[col],q,smp))
    
    
vgsimp,=plt.plot(taus,vgo,'+')
vgep,=plt.plot(taus,vgd)
plt.xlabel('Maturity')
plt.ylabel('PD')
plt.legend([vgsimp,vgep],['VGMC','VGExact'])
plt.title('VG-Cox Stochastic Barrier Model MC Simulation vs Exact Formula')
