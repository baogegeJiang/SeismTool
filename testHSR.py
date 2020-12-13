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
T3=  seism.Trace3(seism.getTrace3ByFileName(stations[0].getFileNames(UTCDateTime(2019,9,5)),freq=[0.5, 90],delta0=0.005))
T3=seism.Trace3(T3)
timeL,vL=T3.getDetec(minValue=30000)
time=timeL[12]
t3=T3.slice(time-5,time+5)
t3=T3.slice(UTCDateTime(2019,9,5),UTCDateTime(2019,9,5)+8*3600)
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
plt.savefig('tmp/all.jpg',dpi=600)
#通过速度校正，前后分别叠加
#寻找有关频率的多普勒效应
h = hsr.hsr()
h.findFF(t3)
specBe = 
specAf
T3L=[]  
for i in range(5):
    T3L.append(seism.Trace3(seism.getTrace3ByFileName(stations[0].getFileNames(UTCDateTime(2019,9,5)+i*86400),freq=[0.5, 90],delta0=0.005)))

specBe,specAf = h.getAdd(T3)
specBe=h.fL0*0
specAf=h.fL0*0
for T3 in T3L:
    if len(T3)<=0:
        continue
    tmpBe,tmpAf = h.getAdd(T3)
    specBe+=tmpBe
    specAf+=tmpAf
plt.close()
plt.plot(h.fL0,specBe/specBe.std(),'b',linewidth=0.3)
plt.plot(h.fL0,specAf/specAf.std(),'r',linewidth=0.3)
plt.xlim([0,5])
plt.savefig('tmp/adjust.jpg',dpi=600)