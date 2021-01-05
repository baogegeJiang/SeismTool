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
stations =  seism.StationList('hsrStd')

#通过速度校正，前后分别叠加
#寻找有关频率的多普勒效应


timeL=UTCDateTime(2019,8,30).timestamp+np.arange(6)*86400
stations0 =  seism.StationList('hsrStd')
stations = stations0[:1]+stations0[-1:]
stations = stations0
reload(hsr)
h = hsr.hsr()

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
        h.loadStationsSpec(stations,'../hsrRes/V11/%d_%d/'%(bSec,eSec),comp=comp,isAdd=True,minDT=0,maxDT=100,dF=0.15,)
        fL0LN,specAtLN,specBeLN,specAfLN,FFAtLN,FFBeLN,FFAfLN,dTLN=\
        h.loadStationsSpec(stations,'../hsrRes/V11/%d_%d/'%(bSec,eSec),comp=comp,isAdd=True,minDT=-100,maxDT=0,dF=0.15,)
        head = 'N_%d_%d~%d'%(comp,bSec,eSec)
        FBeLN,FAfLN,SBeLN,SAfLN,vLN=h.showSpec(fL0LS,specAtLN,specBeLS,specAfLN,stations,head=head,workDir='../hsrRes/V11/fig1/',maxF=20,v=3000,isPlot=False)
        head = 'S_%d_%d~%d'%(comp,bSec,eSec)
        FBeLS,FAfLS,SBeLS,SAfLS,vLS=h.showSpec(fL0LS,specAtLS,specBeLN,specAfLS,stations,head=head,workDir='../hsrRes/V11/fig1/',maxF=20,v=3000,isPlot=True)
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
h.plotFV(FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL1,eSecL1,np.arange(17)*40/80,workDir='../hsrRes/V11/fig2/',head='plotFV')
h.plotFV(FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL1,eSecL1,np.arange(1)*40/80*0,workDir='../hsrRes/V11/fig2/',head='plotFV—single',marker='.')
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
T3=seism.Trace3(seism.getTrace3ByFileName(stations[0].\
    getFileNames(1.567123401723240137e+09),freq=[0.5, 45],delta0=0.01))
t3=T3.slice(1.567123893571964979e+09-40,1.567123893571964979e+09+60)
t31=t3.rotate(25)
from matplotlib import pyplot as plt
plt.close()
data = t31.Data()
plt.figure(figsize=[8,4])
plt.subplot(2,1,1)
plt.plot(np.arange(len(data))*0.01-40,data[:,0],'r',linewidth=0.5)
#plt.plot(timeL,rBe,'b',linewidth=0.5)
plt.xlabel('t/s')
plt.ylabel('Disp')
plt.xlim([-40,40])
plt.subplot(2,1,2)
dataS,TL,FL=hsr.SFFT(data[:,0],0.01,10,800)
plt.pcolor(TL-40,FL[:int(800*0.08)],np.abs(dataS)[:int(800*0.08)]/np.abs(dataS.std(axis=0,keepdims=True)))
plt.xlabel('t/s')
plt.ylabel('f/Hz')
plt.xlim([-40,40])
plt.ylim([0,8])
plt.savefig('../hsrRes/rtReal%s.jpg'%head,dpi=500)
stations = stations0[:1]+stations0[-1:]
T3L=[]
day=timeL[0]
for station in stations:
    print('loading', station)
    T3L.append(seism.Trace3(seism.getTrace3ByFileName(station.\
        getFileNames(day),freq=[0.5, 90],delta0=0.005)))
reload(hsr)
h= hsr.hsr(fL0=np.arange(0,100,0.01))
