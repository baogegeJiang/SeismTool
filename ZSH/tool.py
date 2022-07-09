import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import time
def decompose(st):
    sourceKey='trace_identification_code'
    keyXS = 'SourceX'
    keyYS=  'SourceY'
    sourceIdD = {}
    for i in range(len(st.header)):
        head = st.header[i]
        keys = head.keys()
        IXS = keys.index(keyXS)
        IYS = keys.index(keyYS)
        sourceId = '%d %d'%(head[IXS],head[IYS])
        if sourceId not in sourceIdD:
            sourceIdD[sourceId]=[]
        sourceIdD[sourceId].append(i)
    return sourceIdD
def decompose_(st):
    keyXS = 'source_coordinate_x'
    keyYS=  'source_coordinate_y'
    sourceIdD = {}
    for ST in st:
        head = ST.stats['segy']['trace_header']
        sourceId = '%d %d'%(head[keyXS],head[keyYS])
        if sourceId not in sourceIdD:
            sourceIdD[sourceId]=[]
        sourceIdD[sourceId].append(ST)
    return sourceIdD

def toGather(st,maxV=2,minV=0.5,tail=0.25,mul=4):
    keyXG = 'GroupX'
    keyYG=  'GroupY'
    keyXS = 'SourceX'
    keyYS=  'SourceY'
    #traceKey = 'trace_sequence_number_within_line'
    #sourceKey='energy_source_point_number'
    #traceKey = 'trace_sequence_number_within_line'
    data = np.zeros([len(st),len(st[0].data)],dtype='float32')
    timeL=np.arange(len(st[0].data),dtype='float32')/st[0].stats.sampling_rate
    xg   =      np.zeros([len(st)])
    yg   =      np.zeros([len(st)])
    xs   =      np.zeros([len(st)])
    ys   =      np.zeros([len(st)])
    az   =      np.zeros([len(st)])
    dis   =      np.zeros([len(st)])
    sourceL  = []
    traceL   = []
    snrL =[]
    i=0
    for ii in range(len(st)):
        ST = st[ii]
        head = ST.stats['segy']['trace_header']
        xg[i] = head[keyXG]/1e3
        yg[i] = head[keyYG]/1e3
        xs[i] = head[keyXS]/1e3
        ys[i] = head[keyYS]/1e3
        dis[i]=((xg[i]-xs[i])**2+(yg[i]-ys[i])**2)**0.5
        az[i] = (90-np.angle((xg[i]-xs[i])+1j*(yg[i]-ys[i]))/np.pi*180)%360
        time0 = dis[i]/maxV-tail
        time1 = dis[i]/minV+tail
        i0 = np.abs(time0-timeL).argmin()
        i1 = np.abs(time1-timeL).argmin()
        data[i,i0:i1] = ST.data[i0:i1]
        #data[i] = ST.data
        source = '%d %d'%(xs[i],ys[i])
        #traceL.append(head[traceKey])
        #sourceL.append(source)
        snrL.append(ST.data[i0:i1].std()/np.concatenate([ST.data[:int(i0/3)],ST.data[i1:]],axis=0).std())
        i=i+1
    return gather(data[:i],xg[:i],yg[:i],xs[:i],ys[:i],dis[:i],az[:i],timeL,snrL)
def xcorrFrom0(a,b,fromI=0):
    la = a.size
    lb = b.size
    x =  signal.correlate(a,b,'full')
    return x[lb-1+fromI:]
class gather:
    def __init__(self,data,xg,yg,xs,ys,dis,az,timeL,snrL):
        self.data=data
        self.xg = xg
        self.yg = yg
        self.xs = xs
        self.ys = ys
        self.dis = dis
        self.az  = az
        self.timeL = timeL
        self.snrL =np.array(snrL)
    def pair(self,minDis=0.5,maxDis=3,minDDis=0.25,maxDDis=3,vmin=1.5/2.2,vmax=5/2.2,maxTheta=5,maxCount = 3072,fromT = -384/125,minSnr=0,deltaM=0.075,deltaD=0.075):
        delta = self.timeL[1]-self.timeL[0]
        data=[]
        ddis =[]
        xm =[]
        ym =[]
        az = []
        timeL = np.arange(3072)*delta+fromT
        fromI = int(fromT/delta)
        count=0
        sTime = time.time()
        for i in range(len(self.data)):
            if self.snrL[i]<minSnr:
                continue
            XM0=0
            YM0=999999999
            DDis0=9999999
            for j in range(len(self.data)):
                if self.snrL[j]<minSnr:
                    continue
                if self.dis[i]<self.dis[j] or (self.az[i]-self.az[j]+maxTheta)%360>2*maxTheta or np.abs(self.dis[i]-self.dis[j])<minDDis or np.abs(self.dis[i]-self.dis[j])>maxDDis or self.dis[i]>maxDis or self.dis[j]<minDis:
                    continue
                XM = (self.xg[i]+self.xg[j])/2
                YM = (self.yg[i]+self.yg[j])/2
                DDis = np.abs(self.dis[i]-self.dis[j])
                if ((XM-XM0)**2+(YM-YM0)**2)**0.5<deltaM:
                    if np.abs(DDis-DDis0)<deltaD:
                        continue
                XM0=XM
                YM0=YM
                DDis0=DDis
                Data = np.zeros([maxCount,2],dtype='float32')
                xx=xcorrFrom0(self.data[i],self.data[j],fromI=fromI)
                Data[:len(xx),0]=xx/np.abs(xx).max()
                Data[(timeL>DDis/vmax)*(timeL<DDis/vmin),1]=1
                data.append(Data)
                ddis.append(DDis)
                az.append((self.az[i]+self.az[j])/2)
                xm.append(XM)
                ym.append(YM)
                count+=1
                if count%5000==0:
                    print('find',count,'use',time.time()-sTime)
        return Pair(data,ddis,xm,ym,timeL,az)
def calSpec(data,f,timeL):
    return np.exp(-1j*np.pi*2*f.reshape([-1,1])*timeL.reshape([1,-1]))*data.reshape([1,-1]).sum(axis=1)

class Pair:
    def __init__(self,data,ddis,xm,ym,timeL,az):
        self.data=np.array(data)
        self.ddis=np.array(ddis)
        self.xm =xm
        self.ym=ym
        self.timeL = timeL
        self.az=az
    def plot(self,n=5):
        plt.close()
        plt.figure(figsize=(8,8))
        for i in range(n):
            plt.plot(self.timeL,self.data[i,:,0],'k',linewidth=0.5)
            plt.plot(self.timeL,self.data[i,:,1],'r',linewidth=0.5)
        plt.savefig('plot/xx.jpg',dpi=300)
        plt.close()
    def predict(self,model):
        yout= model.predict(self.data.reshape([len(self.data),-1,1,2]))
    def toV(self,model,vL=np.arange(0.6,2.2,0.01),batch_size=5000):
        vM =[]
        delta = self.timeL[1]-self.timeL[0]
        stime = time.time()
        for i in range(0,len(self.data),batch_size):
            data = self.data[i:min(len(self.data),i+batch_size)]
            yout= model.predict(data.reshape([len(data),-1,1,2]))
            if i%5000==0:
                print(i,'in',len(self.data),time.time()-stime)
            for j in range(batch_size):
                if i+j>=len(self.data):
                    break
                indexL = np.round((self.ddis[i+j]/vL-self.timeL[0])/delta).astype(np.int)
                vM.append(yout[j,indexL])
        print(time.time()-stime,(time.time()-stime)/len(self.data))
        return np.array(vM)

class Velocity:
    def __init__(self,v,xm,ym,ddis,vL=np.arange(0.6,2.2,0.01)):
        self.v=v
        self.xm=xm
        self.ym =ym
        self.ddis = ddis
        self.vL = vL

