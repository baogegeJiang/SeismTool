from mathTool.mathFunc import rotate
from HSR.hsr import handleDay
import os
import sys
from imp import reload
from obspy import UTCDateTime,read
from SeismTool.io import seism,dataLib
from matplotlib import pyplot as plt
import numpy as np
from SeismTool.HSR import hsr
from SeismTool.SurfDisp import fk
from plotTool import figureSet
from SeismTool.mapTool import mapTool as mt
from SeismTool.mathTool.mathFunc_bak import Line
import mpl_toolkits.basemap as basemap
from SeismTool.plotTool import figureSet as fs
stations =  seism.StationList('hsrStd')
pL = [[stations[1]['la'],stations[1]['lo']],[stations[-1]['la'],stations[-1]['lo']]]
line = Line(pL,H=1000)
DL = np.array([ line.l([station['la'],station['lo']])for station in stations])
DL = DL-DL[0]
DL*=1000
T0=1.567129671405785799e+09
T0 =1.567174409828289032e+09
T0=1.567141351420457602e+09
NS=1
T3L = [seism.Trace3(seism.getTrace3ByFileName(station.\
    getFileNames(int(T0/86400)*86400),freq=[0.5, 40],delta0=0.01)) for station in stations]
reload(hsr)
h= hsr.hsr(fL0=np.arange(4.5,5.5,0.005),uL=np.arange(1,10000,50))
T0=1.567141733064932823e+09
T0=1.567129671405785799e+09
T0=1.567129671405785799e+09
specBeL,specAtL,specAfL,FFBe,FFAt,FFAf,uL = h.getOne(T3L,T0,DL,NS=NS,bSec=28,eSec=36,rotate=25,comp=0)
h.plotUSpec(FFBe,specBeL,'testUSpecBe.jpg',f0=3.2)
h.plotUSpec(FFAf,specAfL,'testUSpecAf.jpg',f0=3.2)
reload(hsr)
h= hsr.hsr(fL0=np.arange(4.5,5.5,0.005))
specBeLL,specAtLL,specAfLL,FFBeL,FFAtL,FFAfL,uLL =[],[],[],[],[],[],[]
for T0 in [1.567141733064932823e+09,1.567129671405785799e+09,1.567129671405785799e+09,1.567206717342578650e+09,1.567204486511776209e+09,1.567202587522882462e+09,1.567171759178528547e+09,1.567171436670357704e+09,1.567169209026289940e+09]:
    specBeL,specAtL,specAfL,FFBe,FFAt,FFAf,uL = h.getOne(T3L,T0,DL,NS=NS,bSec=16,eSec=24,rotate=25,comp=0)
    specBeLL.append(specBeL)
    specAfLL.append(specAfL)
    uLL.append(uL)

specBeL = np.abs(np.array(specBeLL)).mean(axis=0)
specAfL = np.abs(np.array(specAfLL)).mean(axis=0)
h.plotUSpec(3.2,specBeL,'testUSpecBeM.jpg',f0=3.2)
h.plotUSpec(3.2,specAfL,'testUSpecAfM.jpg',f0=3.2)
reload(hsr)
h= hsr.hsr(fL0=np.arange(4.7,5.3,0.002),uL=np.arange(1000,10000,50))
specBeLN,specAfLN,specBeLS,specAfLS= h.handleDayUSpec(stations,T0,comp=0,rotate=25,bSec=20,eSec=28,T3L=T3L,DL=DL,maxDV=1.5)
h.plotUSpec(3.2,specBeLN,'testUSpecBeN.jpg',f0=3.2)
h.plotUSpec(3.2,specAfLN,'testUSpecAfN.jpg',f0=3.2)
h.plotUSpec(3.2,specBeLS,'testUSpecBeS.jpg',f0=3.2)
h.plotUSpec(3.2,specAfLS,'testUSpecAfS.jpg',f0=3.2)
reload(hsr)
h= hsr.hsr(fL0=np.arange(4.7,5.3,0.002),uL=np.arange(1500,5000,25))
bSecL = np.arange(10,80,1)
fBeNL =[]
fAfNL =[]
fBeSL =[]
fAfSL =[]
vN = []
vS = []
for bSec in  bSecL:
    specBeLN,specAfLN,specBeLS,specAfLS= h.handleDayUSpec(stations,T0,comp=0,rotate=25,bSec=bSec,eSec=bSec+10,T3L=T3L,DL=DL,maxDV=2)
    fBeNL.append(h.findBF(specBeLN,f0=5,f1=5.3))
    fAfNL.append(h.findBF(specAfLN,f0=4.7,f1=5))
    fBeSL.append(h.findBF(specBeLS,f0=5,f1=5.3))
    fAfSL.append(h.findBF(specAfLS,f0=4.7,f1=5))
    vN.append(80*(fBeNL[-1]+fAfSL[-1])/(fBeNL[-1]-fAfSL[-1]))
    vS.append(80*(fBeSL[-1]+fAfNL[-1])/(fBeSL[-1]-fAfNL[-1]))

h.plotUSpec(3.2,specBeLN,'testUSpecBeND.jpg',f0=3.2)
h.plotUSpec(3.2,specAfLN,'testUSpecAfND.jpg',f0=3.2)
h.plotUSpec(3.2,specBeLS,'testUSpecBeSD.jpg',f0=3.2)
h.plotUSpec(3.2,specAfLS,'testUSpecAfSD.jpg',f0=3.2)
plt.close()
plt.subplot(2,1,1)
plt.plot(bSecL*80,fBeNL,'.b')
plt.plot(-bSecL*80,fBeSL,'.b')
fMN = 2/(1/np.array(fBeNL)+1/np.array(fAfSL))
fMS = 2/(1/np.array(fBeSL)+1/np.array(fAfNL))
plt.plot(-bSecL*80,fMS,'.k')
plt.plot(+bSecL*80,fMN,'.k')
plt.plot(-bSecL*80,fAfNL,'.r')
plt.plot(+bSecL*80,fAfSL,'.r')
#plt.xlim([-5000,5000])
plt.subplot(2,1,2)
plt.plot(bSecL*80,vN,'.k')
plt.plot(-bSecL*80,vS,'.k')
plt.ylim([0,4000])
#plt.xlim([-5000,5000])
plt.savefig('testUV.jpg',dpi=300)
######
######
reload(hsr)
h= hsr.hsr()
hsr.plotStaRail2(mt,basemap)

#通过速度校正，前后分别叠加
#寻找有关频率的多普勒效应


timeL=UTCDateTime(2019,8,30).timestamp+np.arange(0,6,1)*86400
stations0 =  seism.StationList('hsrStd')
stations = stations0[:1]+stations0[-1:]
stations = stations0
reload(hsr)
h = hsr.hsr()
fs.init(key='ZGKX')
bSecL=np.arange(12)*4;eSecL=np.arange(12)*4+8
#bSecL=np.arange(24)*2;eSecL=np.arange(24)*2+4
#bSecL=np.arange(10)*5;eSecL=np.arange(10)*5+10
#h.handleDays(stations,timeL,'../hsrRes/V3/',rotate=25)
#h.handleDays(stations,timeL,'../hsrRes/V11/',rotate=25,bSecL=bSecL,eSecL=eSecL,isPool=True)
h.handleDays(stations,timeL,'../hsrRes/V30/',rotate=25,bSecL=bSecL,eSecL=eSecL,isPool=False,compL=[0])

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
version='V30'
fversion ='10'#3
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
perN=1000#6
isPlot=False
for comp in [0]:
    for j in range(len(bSecL)):
        bSec=bSecL[j]
        eSec=eSecL[j]        
        fL0LS,specAtLS,specBeLS,specAfLS,FFAtLS,FFBeLS,FFAfLS,dTLS=\
        h.loadStationsSpec(stations,'../hsrRes/%s/%d_%d/'%(version,bSec,eSec),comp=comp,isAdd=True,minDT=0,maxDT=100,dF=0.05*3.2,isRemove=False,perN=perN)
        fL0LN,specAtLN,specBeLN,specAfLN,FFAtLN,FFBeLN,FFAfLN,dTLN=\
        h.loadStationsSpec(stations,'../hsrRes/%s/%d_%d/'%(version,bSec,eSec),comp=comp,isAdd=True,minDT=-100,maxDT=0,dF=0.05*3.2,isRemove=False,perN=perN)
        head = 'North_%d_%d~%d'%(comp,bSec,eSec)
        FBeLN,FAfLN,SBeLN,SAfLN,vLN=h.showSpec(fL0LS,specAtLN,specBeLS,specAfLN,stations,head=head,workDir='../hsrRes/%s/fig%s/'%(version,fversion),maxF=25,v=3000,isPlot=isPlot)
        head = 'South_%d_%d~%d'%(comp,bSec,eSec)
        FBeLS,FAfLS,SBeLS,SAfLS,vLS=h.showSpec(fL0LS,specAtLS,specBeLN,specAfLS,stations,head=head,workDir='../hsrRes/%s/fig%s/'%(version,fversion),maxF=25,v=3000,isPlot=isPlot)
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
#h.testDenseFFT()
h.plotFV(FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL1,eSecL1,np.arange(1)*40/80*0,workDir='../hsrRes/',head='plotFV—single1',marker='.',strL='ac')
h.plotFV(FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL1,eSecL1,np.arange(0,17)*40/80,workDir='../hsrRes/',head='plotFV1',strL='bd',maxStd=1000)#300
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
t0=1.567133837665162086e+09+10
dt = 1000*stations[1].dist(stations[0])/80

T3=seism.Trace3(seism.getTrace3ByFileName(stations[1].\
    getFileNames(int(t0/86400)*86400),freq=[0.5, 40],delta0=0.01))
reload(hsr)
h= hsr.hsr()
t0=1.567129671405785799e+09+dt
t3=T3.slice(t0-50,t0+60)
t31=t3.rotate(25)
data = t31.Data()[:,0]

t1 = 1.567132350323159933e+09-dt
t3_=T3.slice(t1-50,t1+60)
t31_=t3_.rotate(25)
reload(hsr)
h = hsr.hsr()
t2 =1.567174409828289032e+09-dt
t3__=T3.slice(t2-50,t2+60)
t31__=t3__.rotate(25)
reload(hsr)
h = hsr.hsr()
h.plotERR5(t3=t31,t3_=t31__,time=t0,time1=t2)
h.plotERR5(t3=t31,t3_=t31__,time=t0,time1=t2,U=[3000,1000],head='1000')
#5.09 4.81 80.5781012943 79.6618012036 80.1 2832.81256202
#5.2 4.91 82.0775014427 81.3278013685 81.7 2848.32346797
#
reload(hsr)
h = hsr.hsr()
h.plotWS(data,time0=-50,head='real',fMax=15,delta=0.01,whiteL=[2.5,5,7.5,10,12.5])
h.plotWS(data,time0=-50,head='realHigh',fMin=40,fMax=60,delta=0.01,whiteL=[50,45,55])
h.plotWS(data,time0=-50,head='realHighMid',fMin=20,fMax=30,delta=0.01,whiteL=[20,25,30])
h.plotWS(data,time0=-50,head='realHighMidMid',fMin=8,fMax=22,delta=0.01,whiteL=[10,15,20])
h.plotWS(data,time0=-50,head='all',fMax=50,delta=0.01,isSpec=True)

reload(hsr)
h = hsr.hsr()
h.plotr(head='syn',linewidth=0.1)

1.567142294544028521e+09



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

figureSet.init()
fig=plt.figure(figsize=[4,4])
plt.subplot(3,1,1)
plt.plot(x,y,'k',linewidth=2)
plt.plot([0],[0],'^y',markersize=10)
plt.plot(squareX,squareY,'b')
plt.plot(squareX_,squareY,'r')
plt.xlim(x)
plt.xlabel('$distance$/m')
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
plt.xlabel('$time$/s')
plt.ylabel('$D$')
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
plt.xlabel('$time$/s')
plt.ylabel('$D$')
figureSet.setABC('(c)',[0.01,0.98],c='k')
#fig.tight_layout()
plt.savefig('../hsrRes/twoD.eps')
############plot stations and high speed rail
reload(hsr)
h= hsr.hsr()
hsr.plotStaRail(stations,mt,basemap)
####plotERR
reload(hsr)
h= hsr.hsr()
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