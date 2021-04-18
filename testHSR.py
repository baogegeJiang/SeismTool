import os
import sys
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from imp import reload
from obspy import UTCDateTime,read
from SeismTool.io import seism,dataLib
from matplotlib import pyplot as plt
import numpy as np
from SeismTool.HSR import hsr
from SeismTool.SurfDisp import fk
from plotTool import figureSet
from SeismTool.mapTool import mapTool as mt
import mpl_toolkits.basemap as basemap
from SeismTool.plotTool import figureSet as fs
stations =  seism.StationList('hsrStd')

#通过速度校正，前后分别叠加
#寻找有关频率的多普勒效应


timeL=UTCDateTime(2019,8,30).timestamp+np.arange(6)*86400
stations0 =  seism.StationList('hsrStd')
stations = stations0[:1]+stations0[-1:]
stations = stations0
reload(hsr)
h = hsr.hsr()
fs.init(key='ZGKX')
bSecL=np.arange(12)*4;eSecL=np.arange(12)*4+8
#h.handleDays(stations,timeL,'../hsrRes/V3/',rotate=25)
h.handleDays(stations,timeL,'../hsrRes/V11/',rotate=25,bSecL=bSecL,eSecL=eSecL,isPool=True)

minDTL=[-100,0,-100]
maxDTL=[ 100,100,0]
for comp in [0,1,2]:
    for i in [1,2]:
        for j in range(len(bSecL)):
            bSec=bSecL[j]
            eSec=eSecL[j]
            minDT=minDTL[i]
            maxDT=maxDTL[i]
            head = '%d_%d_%d_%d~%d'%(comp,minDT,maxDT,bSec,eSec)
            fL0L2,specAtL2,specBeL2,specAfL2,FFAtL2,FFBeL2,FFAfL2,dTL2=\
                h.loadStationsSpec(stations,'../hsrRes/V6/%d_%d/'%(bSec,eSec),comp=comp,isAdd=True,minDT=minDT,maxDT=maxDT,dF=0.2,bandStr='')
            h.showSpec(fL0L2,specAtL2,specBeL2,specAfL2,stations,head=head,workDir='../hsrRes/V6/fig/',maxF=20,v=3000)
reload(hsr)
h = hsr.hsr()
version='V11'
fversion ='8'#3
FBeLSM=[[]for i in range(3)]
FAfLSM=[[]for i in range(3)]
SBeLSM=[[]for i in range(3)]
SAfLSM=[[]for i in range(3)]
vLSM=[[]for i in range(3)]
SBeLNM=[[]for i in range(3)]
SAfLNM=[[]for i in range(3)]
FBeLNM=[[]for i in range(3)]
FAfLNM=[[]for i in range(3)]
vLNM=[[]for i in range(3)]
for comp in [0,1,2]:
    for j in range(len(bSecL)):
        bSec=bSecL[j]
        eSec=eSecL[j]        
        fL0LS,specAtLS,specBeLS,specAfLS,FFAtLS,FFBeLS,FFAfLS,dTLS=\
        h.loadStationsSpec(stations,'../hsrRes/%s/%d_%d/'%(version,bSec,eSec),comp=comp,isAdd=True,minDT=0,maxDT=100,dF=0.075*3.2,isRemove=True,perN=6)
        fL0LN,specAtLN,specBeLN,specAfLN,FFAtLN,FFBeLN,FFAfLN,dTLN=\
        h.loadStationsSpec(stations,'../hsrRes/%s/%d_%d/'%(version,bSec,eSec),comp=comp,isAdd=True,minDT=-100,maxDT=0,dF=0.075*3.2,isRemove=True,perN=6)
        head = 'N_%d_%d~%d'%(comp,bSec,eSec)
        FBeLN,FAfLN,SBeLN,SAfLN,vLN=h.showSpec(fL0LS,specAtLN,specBeLS,specAfLN,stations,head=head,workDir='../hsrRes/%s/fig%s/'%(version,fversion),maxF=25,v=3000,isPlot=False)
        head = 'S_%d_%d~%d'%(comp,bSec,eSec)
        FBeLS,FAfLS,SBeLS,SAfLS,vLS=h.showSpec(fL0LS,specAtLS,specBeLN,specAfLS,stations,head=head,workDir='../hsrRes/%s/fig%s/'%(version,fversion),maxF=25,v=3000,isPlot=False)
        FBeLNM[comp].append(FBeLN)
        FAfLNM[comp].append(FAfLN)
        SBeLNM[comp].append(SBeLN)
        SAfLNM[comp].append(SAfLN)
        vLNM[comp].append(vLN)
        FBeLSM[comp].append(FBeLS)
        FAfLSM[comp].append(FAfLS)
        SBeLSM[comp].append(SBeLS)
        SAfLSM[comp].append(SAfLS)
        vLSM[comp].append(vLS)
bSecL1=bSecL.copy()
eSecL1=eSecL.copy()
bSecL1[0]=-1000
eSecL1[0]=-1000
bSecL1[1]=-1000
eSecL1[1]=-1000
reload(hsr)
h = hsr.hsr()
h.plotFV(FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL1,eSecL1,np.arange(1)*40/80*0,workDir='../hsrRes/',head='plotFV—single',marker='.',strL='ac')
h.plotFV(FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL1,eSecL1,np.arange(17)*40/80,workDir='../hsrRes/',head='plotFV',strL='bd')

h.plotSV2(SBeLNM,SAfLNM,vLNM,SBeLSM,SAfLSM,vLSM,bSecL1,eSecL1,np.arange(17)*40/80*0,workDir='../hsrRes/V11/fig2/',head='plotSVReLa')
#基频与桥频之差，反映轨道形状
#有的随车速变化，有的不随；
#有的有多普勒效应，有的没有
#仅算直行之部分（根据车速），多普勒算波速
#有的分量是面波明显，有的分量是S波明显，有的是面波，起波速不同
#车频的主要能量集中在Z
#桥频的主要能量集中在R
#来前，来时，来后之能量变化
#不同频率随分量的变化不同



#syn source
FK = fk.FK(exePath='../hsrRes/fkRun/',resDir='../hsrRes/fk/')
distance=np.arange(60)*32/1000+16
expnt=10
dt=0.01
M=[1e15,0,90]
dura=0.08
rise=0.5
depth=0.0001
modelFile='data/hsrModel'
FK.test(distance=distance,modelFile=modelFile,dt=dt,depth=depth,expnt=expnt,dura=dura,M=M,rise=rise,dk=0.1)
#thickness vs vp/vs [rho Qs Qp]
#0.2  0.6 2.5
#0.4  1.6 1.9
#0.8  2.2 1.75
#1    2.6 1.72
#4.5  3.2 1.72

#10   3.6 1.72

##syn NB

###handle
reload(hsr)
h = hsr.hsr()
h.plotER()
#t0=1.567133837665162086e+09+10
dt = 1000*stations[1].dist(stations[0])/80
t0=1.567129671405785799e+09+dt
T3=seism.Trace3(seism.getTrace3ByFileName(stations[1].\
    getFileNames(int(t0/86400)*86400),freq=[0.5, 95],delta0=0.004))
reload(hsr)
h= hsr.hsr()
t3=T3.slice(t0-50,t0+60)
t31=t3.rotate(25)
data = t31.Data()[:,0]

h.plotERR(t3=t31,time=t0)
reload(hsr)
h = hsr.hsr()
h.plotWS(data,time0=-50,head='real',fMax=15,delta=0.004,whiteL=[2.5,5,7.5,10,12.5])
h.plotWS(data,time0=-50,head='realHigh',fMin=40,fMax=60,delta=0.004,whiteL=[50,45,55])
h.plotWS(data,time0=-50,head='realHighMid',fMin=20,fMax=30,delta=0.004,whiteL=[20,25,30])
h.plotWS(data,time0=-50,head='realHighMidMid',fMin=8,fMax=22,delta=0.004,whiteL=[10,15,20])
h.plotWS(data,time0=-50,head='all',fMax=100,delta=0.004)
reload(hsr)
h = hsr.hsr()
h.plotr()
1.567142294544028521e+09

t1 = 1.567132350323159933e+09-dt
t3_=T3.slice(t1-50,t1+60)
t31=t3_.rotate(25)

###plot Two Direction
bSec = -40
eSec = 40
bT=12
eT=20
v=80

y=[0,0]
x=[bSec*v,eSec*v]
squareX = [bT*v,bT*v,eT*v,eT*v,bT*v]
squareX_ = [-bT*v,-bT*v,-eT*v,-eT*v,-bT*v]
squareT= [bT,bT,eT,eT,bT]
squareT_= [-bT,-bT,-eT,-eT,-bT]
squareY = [0.5,   -0.5,  -0.5  , 0.5,  0.5]
plt.close()
fig=plt.figure(figsize=[4,6])
figureSet.init()
plt.subplot(3,1,1)
plt.plot(x,y,'k',linewidth=2)
plt.plot([0],[0],'^y',markersize=10)
plt.plot(squareX,squareY,'b')
plt.plot(squareX_,squareY,'r')
plt.xlim(x)
plt.xlabel('distance/m')
plt.ylim([-1,1])
plt.yticks([])
figureSet.setABC('(a)',[0.01,0.98],c='k')
plt.subplot(3,1,2)
data = t31.Data()[:,0]
data/=data.max()*1.2
timeL=np.arange(len(data))*t31.delta-50
plt.plot(timeL,data,'k',linewidth=0.5)
plt.plot(squareT,squareY,'b')
plt.plot(squareT_,squareY,'r')
plt.xlim([bSec,eSec])
plt.ylim([-1,1])
plt.xlabel('time/s')
plt.ylabel('A')
figureSet.setABC('(b)',[0.01,0.98],c='k')
plt.subplot(3,1,3)
data = t31_.Data()[:,0]
data/=data.max()*1.2
timeL=np.arange(len(data))*t31.delta-50
plt.plot(timeL,data,'k',linewidth=0.5)
plt.plot(squareT,squareY,'r')
plt.plot(squareT_,squareY,'b')
plt.xlim([eSec,bSec])
plt.ylim([-1,1])
plt.xlabel('time/s(distance/km)')
plt.ylabel('A')
figureSet.setABC('(c)',[0.01,0.98],c='k')
fig.tight_layout()
plt.savefig('../hsrRes/twoD.eps')
############plot stations and high speed rail
reload(hsr)
h= hsr.hsr()
hsr.plotStaRail(stations,mt,basemap)
####plotERR

h.plotERR(t3=t31,time=t0)



'''
figureSet.init()
head = '1.567142294544028521e+09'
plt.close()
data = t31.Data()
plt.figure(figsize=[12,8])
plt.subplot(2,1,1)
plt.plot(np.arange(len(data))*0.01-40,data[:,0],'k',linewidth=0.5)
#plt.plot(timeL,rBe,'b',linewidth=0.5)
plt.xlabel('t/s')
plt.ylabel('Disp')
plt.xlim([-40,40])
plt.subplot(2,1,2)
dataS,TL,FL=hsr.SFFT(data[:,0],0.01,10,800)
plt.pcolormesh(TL-40,FL[:int(800*15/100)],np.log(np.abs(dataS[:int(800*15/100)])/np.std(dataS[:int(800*15/100)]).max(axis=0,keepdims=True)+1e-3),cmap='hot')
plt.xlabel('t/s')
plt.ylabel('f/Hz')
plt.xlim([-40,40])
plt.ylim([0,15])
plt.tight_layout()
plt.savefig('../hsrRes/rtReal%d.jpg'%t0,dpi=500)
stations = stations0[:1]+stations0[-1:]
T3L=[]
day=timeL[0]
for station in stations:
    print('loading', station)
    T3L.append(seism.Trace3(seism.getTrace3ByFileName(station.\
        getFileNames(day),freq=[0.5, 90],delta0=0.005)))
'''


reload(hsr)
h = hsr.hsr()

allSBeLSM=[[]for i in range(3)]
allSAfLSM=[[]for i in range(3)]
allvLSM=[[]for i in range(3)]
allSBeLNM=[[]for i in range(3)]
allSAfLNM=[[]for i in range(3)]
allvLNM=[[]for i in range(3)]

allSBeLSM1=[[]for i in range(3)]
allSAfLSM1=[[]for i in range(3)]
allvLSM1=[[]for i in range(3)]
allSBeLNM1=[[]for i in range(3)]
allSAfLNM1=[[]for i in range(3)]
allvLNM1=[[]for i in range(3)]
for comp in [0,1,2]:
    for j in range(len(bSecL)):
        bSec=bSecL[j]
        eSec=eSecL[j]        
        fL0LS,specAtLS,specBeLS,specAfLS,FFAtLS,FFBeLS,FFAfLS,dTLS=\
        h.loadStationsSpec(stations,'../hsrRes/V11/%d_%d/'%(bSec,eSec),comp=comp,isAdd=False,minDT=0,maxDT=100,dF=0.15,)
        fL0LN,specAtLN,specBeLN,specAfLN,FFAtLN,FFBeLN,FFAfLN,dTLN=\
        h.loadStationsSpec(stations,'../hsrRes/V11/%d_%d/'%(bSec,eSec),comp=comp,isAdd=False,minDT=-100,maxDT=0,dF=0.15,)
        FBeLN,FAfLN,SBeLN,SAfLN,vLN=h.anSpec(fL0LS,specAtLN,specBeLS,specAfLN,stations,maxF=20,v=3000)
        FBeLS,FAfLS,SBeLS,SAfLS,vLS=h.anSpec(fL0LS,specAtLS,specBeLN,specAfLS,stations,maxF=20,v=3000)
        allSBeLNM[comp].append(SBeLN)
        allSAfLNM[comp].append(SAfLN)
        allvLNM[comp].append(vLN)
        allSBeLSM[comp].append(SBeLS)
        allSAfLSM[comp].append(SAfLS)
        allvLSM[comp].append(vLS)
        fL0LS,specAtLS,specBeLS,specAfLS,FFAtLS,FFBeLS,FFAfLS,dTLS=\
        h.loadStationsSpec(stations,'../hsrRes/V11/%d_%d/'%(bSec,eSec),comp=comp,isAdd=False,minDT=0,maxDT=100,dF=0.15,)
        fL0LN,specAtLN,specBeLN,specAfLN,FFAtLN,FFBeLN,FFAfLN,dTLN=\
        h.loadStationsSpec(stations,'../hsrRes/V11/%d_%d/'%(bSec,eSec),comp=comp,isAdd=False,minDT=-100,maxDT=0,dF=0.15,)
        FBeLN1,FAfLN1,SBeLN1,SAfLN1,vLN1=h.anSpec(fL0LS,specAtLN,specBeLS,specAfLN,stations,minF=9,maxF=10.2,avoidF=30,fmax=12,v=3000)
        FBeLS1,FAfLS1,SBeLS1,SAfLS1,vLS1=h.anSpec(fL0LS,specAtLS,specBeLN,specAfLS,stations,minF=9,maxF=10.2,avoidF=30,fmax=12,v=3000)
        allSBeLNM1[comp].append(SBeLN1)
        allSAfLNM1[comp].append(SAfLN1)
        allvLNM1[comp].append(vLN1)
        allSBeLSM1[comp].append(SBeLS1)
        allSAfLSM1[comp].append(SAfLS1)
        allvLSM1[comp].append(vLS1)

reload(hsr)
h = hsr.hsr()
h.plotSV2(allSBeLNM,allSAfLNM,allvLNM,allSBeLSM,allSAfLSM,allvLSM,bSecL1,eSecL1,np.arange(17)*40/80*0,workDir='../hsrRes/',head='plotSVReLa')
h.plotSV2(allSBeLNM1,allSAfLNM1,allvLNM1,allSBeLSM1,allSAfLSM1,allvLSM1,bSecL1,eSecL1,np.arange(17)*40/80*0,workDir='../hsrRes/',head='plotSVReLa1')