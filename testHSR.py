import os
import sys
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from imp import reload
from obspy import UTCDateTime,read
from SeismTool.io import seism,dataLib
from matplotlib import pyplot as plt
import numpy as np
from SeismTool.HSR import hsr
stations =  seism.StationList('hsrStd')
T3=  seism.Trace3(seism.getTrace3ByFileName(stations[1].getFileNames(UTCDateTime(2019,9,5)),freq=[0.5, 90],delta0=0.002))
T3=seism.Trace3(T3)
timeL,vL=T32[0].getDetec(minValue=30000)
time=timeL[8]
t3=T32[0].slice(time-5,time+5)
#t3=T32[0]
data = t3.Data()
delta = t3.Delta()
plt.close()
plt.subplot(2,1,1)
plt.plot(np.arange(len(data))*delta,data[:,2],linewidth=0.3)
plt.subplot(2,1,2)
F=np.fft.fft(data[:,0])
f = np.arange(len(F))/(delta*len(F))
plt.plot(f,np.abs(F),linewidth=0.3)
plt.xlim([0,100])
plt.savefig('tmp/at.jpg',dpi=600)
#通过速度校正，前后分别叠加
#寻找有关频率的多普勒效应
reload(hsr)
h = hsr.hsr()
h.findFF(t3)
specBe = 
specAf
T3L=[]  
for i in range(5):
    T3L.append(seism.Trace3(seism.getTrace3ByFileName(stations[1].\
        getFileNames(UTCDateTime(2019,8,30)+i*86400),freq=[0.5, 90],delta0=0.002)))

specBe,specAf = h.getAdd(T3)
specAt=h.fL0*0
specBe=h.fL0*0
specAf=h.fL0*0
for T3 in T3L:
    if len(T3)<=0:
        continue
    tmpAt,tmpBe,tmpAf= h.getAdd(T3)
    specAt+=tmpAt
    specBe+=tmpBe
    specAf+=tmpAf
plt.close()
plt.plot(h.fL0,specAt/specAt.std(),'k',linewidth=0.3)
plt.plot(h.fL0,specBe/specBe.std(),'b',linewidth=0.3)
plt.plot(h.fL0,specAf/specAf.std(),'r',linewidth=0.3)
plt.xlim([0,100])
plt.savefig('tmp/adjust.jpg',dpi=600)

T32=[]  
for i in [1,2]:
    T32.append(seism.Trace3(seism.getTrace3ByFileName(stations[i].\
        getFileNames(UTCDateTime(2019,8,30)),freq=[0.5, 90],delta0=-1)))
#T32=T3L[-2:]
eL = h.getCatolog(T32)
tmpAt,tmpBe,tmpAf= h.getAdd(T32[0],eL[:,0])
plt.close()
plt.plot(h.fL0,tmpAt/tmpAt.std(),'k',linewidth=0.3)
plt.plot(h.fL0,tmpBe/tmpBe.std(),'b',linewidth=0.3)
plt.plot(h.fL0,tmpAf/tmpAf.std(),'r',linewidth=0.3)
plt.xlim([0,5])
plt.savefig('tmp/adjust_5.jpg',dpi=600)
plt.xlim([0,20])
plt.savefig('tmp/adjust_20.jpg',dpi=600)
plt.xlim([0,100])
plt.savefig('tmp/adjust_100.jpg',dpi=600)

FFAt=h.FindFF(tmpAt,h.fL0,2.0,2.5,avoidF=3.2)
FFBe=h.FindFF(tmpBe,h.fL0,2.3,2.7,avoidF=3.2)
FFAf=h.FindFF(tmpAf,h.fL0,2.3,2.7,avoidF=3.2)
(FFAf+FFBe)/(FFAf-FFBe)*80


tmpAt,tmpBe,tmpAf= h.getAdd(T32[1],eL[:,0],3.,3.2)
FFAt=h.FindFF(tmpAt,h.fL0,2.0,2.5,avoidF=3.2,fmax=6)
FFBe=h.FindFF(tmpBe,h.fL0,2.3,2.7,avoidF=3.2,fmax=6)
FFAf=h.FindFF(tmpAf,h.fL0,2.3,2.7,avoidF=3.2,fmax=6)
l1=(FFAf+FFBe)/(FFAf-FFBe)*80


tmpAt,tmpBe,tmpAf= h.getAdd(T32[1],eL[:,0],3.0,3.3)
FFAt=h.FindFF(tmpAt,h.fL0,2.0,2.5,avoidF=3.2,fmax=6)
FFBe=h.FindFF(tmpBe,h.fL0,2.3,2.7,avoidF=3.2,fmax=6)
FFAf=h.FindFF(tmpAf,h.fL0,2.3,2.7,avoidF=3.2,fmax=6)
l2=(FFAf+FFBe)/(FFAf-FFBe)*80
plt.close()
plt.plot(h.fL0,tmpAt/tmpAt.std(),'k',linewidth=0.3)
plt.plot(h.fL0,tmpBe/tmpBe.std(),'b',linewidth=0.3)
plt.plot(h.fL0,tmpAf/tmpAf.std(),'r',linewidth=0.3)
plt.xlim([0,5])
plt.savefig('tmp/adjust_5.jpg',dpi=600)
plt.xlim([0,20])
plt.savefig('tmp/adjust_20.jpg',dpi=600)
plt.xlim([0,100])
plt.savefig('tmp/adjust_100.jpg',dpi=600)

timeL=UTCDateTime(2019,8,30).timestamp+np.arange(5)*86400
stations =  seism.StationList('hsrStd')
reload(hsr)
h = hsr.hsr()
h.handleDay(stations[1:2]+stations[-1:],timeL[0],
T3L=T32,workDir='../hsrRes/V0/')
h.handleDays(stations[1:2]+stations[-1:],timeL,'../hsrRes/V0/')
