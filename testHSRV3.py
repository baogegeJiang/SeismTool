from SeismTool.HSR import hsr
from imp import reload
from obspy import UTCDateTime,read
from SeismTool.io import seism,dataLib
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

stations =  seism.StationList('hsrStd')
#stations =  seism.StationList(seism.StationList('../stations/XA.Loc.201908.txt1'))
stations.set('net','XA')
stations.set('compBase','BH')
stations.set('nameMode','RDDS')
T0=1.567141351420457602e+09
T3L = [seism.Trace3(seism.getTrace3ByFileName(station.\
    getFileNames(int(T0/86400)*86400),freq=[0.5, 40],delta0=0.01)) for station in stations]
reload(seism)
reload(hsr)
#T3L =[seism.Trace3(t3)for t3 in T3L]
#x3 = T3L[0].cross(T3L[1].slice(T0,T0+3))
h=hsr.hsr()
eL=h.getCatolog(T3L[1:])
dTime=120
bTime=-5
N=int((dTime-1)*100)
dataL= []
index=0
data = np.zeros(N)
count=0
for e in eL[:]:
    time0,time1,v0,v1=e
    time0,time1=[min([time0,time1]),max([time0,time1])]
    t3L0 =[t3.slice(time0+bTime,time1+dTime)for t3 in T3L]
    t3L1 =[t3.slice(time0+bTime,time1)for t3 in T3L]
    print(e)
    for i in [2]:
        for j in [2]:
            if stations[i].dist(stations[j])*1e3>200:
                continue
            t30=t3L0[i]
            t31=t3L1[j]
            if t30.Data().size==0 or t31.Data().size==0:
                continue
            v= h.findFF(t31)*25
            print(v)
            if v>83 or v<81:
                continue
            tmp= t30.cross(t31).Data()[:,2]/1e9
            if len(tmp)>=N:
                data+=tmp[:N]
            count+=1
            print(count)
data/=count
#dataL.append(data)
#data0=data.copy()
f=[1.5,45]
plt.close()
b,a=signal.butter(4,f,'bandpass',fs=100)
data1=signal.filtfilt(b,a,data)
plt.plot(np.arange(data1.size)/t30[0].stats['sampling_rate'],data1,linewidth=0.3)
plt.xlim([20,120])
plt.ylim([-0.03,0.03])
plt.savefig('hsrFig/%d_xx.jpg'%index,dpi=300)
#index+=1

h.plotWS(data,time0=0,head='test',delta=0.01,fMin=0,fMax=50,xlim=[0,120],whiteL=[],ylabel0='$D$ (count)',isSpec=True,linewidth=0.5,fileName='hsrFig/%d_ws.jpg'%index)

dTime=500
bTime=10
eTime=60
N=int((dTime-1)*100)
#eL=h.getCatolog(T3L[1:])
dataL= np.zeros([len(stations),N])
countL= np.zeros([len(stations),1])
eL=np.array(eL[:-5])
fL = eL[:,2]
jL =fL.argsort()
V0=1
for j in jL:
    e = eL[j]
    time0,time1,v0,v1=e
    if v0<V0*1.002:
        pass
        #continue
    V0 = v0
    if time0>time1:
        continue
    time0,time1=[min([time0,time1]),max([time0,time1])]
    print(e)
    for i in range(len(stations)):
        T3 =  T3L[i]
        t3 = T3.slice(time0-5,time1+5)
        time = h.MeanTime(t3)
        v= h.findFF(t3)*25
        print(v)
        if v>83 or v<81:
            #continue
            pass
        t30 = T3.slice(time+bTime,time+eTime+dTime)
        t31 = T3.slice(time+bTime,time+eTime)
        tmp= t30.cross(t31).Data()[:,2]/1e9
        if len(tmp)>=N:
            dataL[i]+=tmp[:N]
            countL[i,0]+=1
            print(countL[i])

dataL/=countL
dataL/=dataL.std(axis=1,keepdims=True)
f=[2,10]
mul=0.4
plt.close()
b,a=signal.butter(4,f,'bandpass',fs=100)
for i in range(len(stations)):
    data1=signal.filtfilt(b,a,dataL[i])
    plt.plot(np.arange(data1.size)/t30[0].stats['sampling_rate'],data1*mul+i,'k',linewidth=0.3)

plt.plot(np.arange(data1.size)/t30[0].stats['sampling_rate'],dataL.mean(axis=0)*mul*3+i+5,'r',linewidth=0.3)

plt.xlim([0,500])
plt.ylim([-1,25])
plt.savefig('hsrFig/%d_xxall.jpg'%index,dpi=300)        

time0,time1,v0,v1=eL[3]
t3 = T3L[0].slice(time0-20,time0+20)
data = t3.rotate(25).Data()[:,0]
delta = t3.Delta()
f = np.arange(len(data))/len(data)*1/delta
spec = np.fft.fft(data)
l = np.arange(16)*25
v = np.arange(75,85,0.1)
E = np.exp(-2*np.pi*f.reshape([-1,1,1])*l.reshape([1,-1,1])/v.reshape([1,1,-1])).sum(axis=1)+1e-6
specV = E*spec.reshape([-1,1])
r=np.fft.ifft(specV,axis=0)
plt.close()
plt.figure(figsize=[10,10])
plt.pcolormesh(delta*np.arange(len(data)),v,np.abs(r.transpose()),rasterized=True)
plt.savefig('hsrFig/vR.jpg',dpi=300)

#用较短的截取时间窗来做







'''
reload(dataLib)
reload(seism)

stations1 = seism.StationList(seism.StationList('hsrSz')[::7])
#stations =  seism.StationList(seism.StationList('../stations/XA.Loc.201908.txt1'))
stations1.set('net','SZ')
stations1.set('compBase','BH')
stations1.set('nameMode','SZ')
T3L1 = [seism.Trace3(seism.getTrace3ByFileName(station.\
    getFileNames(UTCDateTime(2018,1,29).timestamp),freq=[0.5, 40],delta0=0.01)) for station in stations1]
reload(hsr)
h=hsr.hsr()
eL=h.getCatolog(T3L1[2:],minValue=20000,minDis=stations1[0].dist(stations1[-1])*1000,refV=70)
eL = eL
eL[:,2]*=25
eL[:,3]*=25
print(eL)
np.savetxt('eventM',eL)

timeL0 = h.meanTime(T3L1[2],T3L1[2].getDetec(20000,20)[0])
np.savetxt('eventM',timeL0)
plt.close()
i=2
plt.plot(np.arange(len(T3L[i].Data()))*T3L[i].Delta(),T3L[i].Data(),'k')
plt.xlim([6e3,7e3])
plt.ylim([-50000,50000])
plt.savefig('test.jpg',dpi=500)

stations1 = seism.StationList(seism.StationList('hsrSzT')[::7])
#stations =  seism.StationList(seism.StationList('../stations/XA.Loc.201908.txt1'))
stations1.set('net','SZ')
stations1.set('compBase','BH')
stations1.set('nameMode','SZ')
T3L1 = [seism.Trace3(seism.getTrace3ByFileName(station.\
    getFileNames(UTCDateTime(2018,1,30).timestamp),freq=[0.5, 40],delta0=0.01)) for station in stations1]
reload(hsr)
h=hsr.hsr()
eL=h.getCatolog(T3L1[2:],minValue=20000,minDis=stations1[0].dist(stations1[-1])*1000,refV=70)
eL = eL
eL[:,2]*=25
eL[:,3]*=25
print(eL)
np.savetxt('event',eL)

timeL0 = h.meanTime(T3L1[2],T3L1[2].getDetec(10000,20)[0])
np.savetxt('eventT',timeL0)
'''