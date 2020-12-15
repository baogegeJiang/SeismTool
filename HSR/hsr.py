from ..io import seism
import numpy as np
from obspy import UTCDateTime
from glob import glob
import os
class hsr:
    def __init__(self,f0=3.2,fmax=50,fL0=np.arange(0,100,0.01)):
        self.f0=f0
        self.fmax= fmax
        self.fL0=fL0
    def findFF(self,T3):
        spec,fL=T3.getSpec()
        return self.FindFF(spec,fL)
    def FindFF(self,spec,fL,minF=2.7,maxF=3.6,avoidF=-1,fmax=50):
        N=fmax/self.f0
        df0=(fL[1]-fL[0])
        df=df0/N
        FFL = np.arange(minF,maxF,df)
        FF0=-1
        S=-1
        if avoidF>0:
            spec=spec.copy()
            N = fL/avoidF
            dN=np.abs(N-np.round(N))
            spec[dN<0.15]=0
        for FF in  FFL:
            FL=np.arange(FF,fmax,FF)
            indexL=((FL-fL[0])/df0).astype(np.int)
            s = np.abs(spec)[indexL].sum()
            if s>=S:
                S=s
                FF0=FF
        return FF0
    def adjustSpec(self,T3,f,f0=3.2):
        spec,fL=T3.getSpec(isNorm=True)
        fL=fL/f*f0
        df0=(fL[1]-fL[0])
        indexL=((self.fL0-fL[0])/df0).astype(np.int)
        return spec[indexL]
    def getAll(self,T3,eL,minF=3.1,maxF=3.3):
        specAt = []
        specBe = []
        specAf = []
        FFAt   = []
        FFBe   = []
        FFAf   = []
        timeL  = eL[:,0]
        v0L    = eL[:,2]
        v1L    = eL[:,3]
        dT     =[]
        for i in range(len(timeL)):
            time = timeL[i]
            v0   = v0L[i]
            v1   = v1L[i]
            if np.abs(v0-v1)>0.1:
                continue
            t3=T3.slice(time-20,time+20)
            time = self.MeanTime(t3)
            t3At=T3.slice(time-5,time+5)
            if t3.bTime>0:
                t3Be=T3.slice(time-55,time-10)
                t3Af=T3.slice(time+10,time+55)
                if t3Af.bTime<0 or t3.bTime<0 or t3.bTime<0:
                    continue
                FFAt.append(self.findFF(t3At))
                specAt.append(self.adjustSpec(t3At,FFAt[-1]))
                FFBe.append(self.findFF(t3Be))
                specBe.append(self.adjustSpec(t3Be,FFBe[-1]))
                FFAf.append(self.findFF(t3Af))
                specAf.append(np.abs(self.adjustSpec(t3Af,FFAf[-1])))
                dT.append(eL[i,1]-eL[i,0])
        return specAt,specBe,specAf,FFAt,FFBe,FFAf,dT
    def meanTime(self,T3,timeL,minValue=15000,minD=5):
        #V = np.abs(T3.Data()).max()/
        timeL,vL=T3.getDetec(minValue=minValue,minD=minD)
        timeLNew=[]
        for time in timeL:
            t3 = T3.slice(time-5,time+5)
            time1 = self.MeanTime(t3)
            if time1>0:
                timeLNew.append(time1)
        return np.array(timeLNew)
    def MeanTime(self,t3):
        data = np.abs(t3.Data())
        if len(data)>0:
            bTime,eTime=t3.getTimeLim()
            delta = t3.Delta()
            indexL = np.arange(len(data))
            iM=(indexL*data[:,2]).sum()/data[:,2].sum()
            time1=iM*delta+bTime.timestamp
            return time1
        else:
            return -1
    def noClose(self,timeL,minD=40):
        timeLNew=[]
        for time in timeL:
            if (np.abs(timeL-time)<minD).sum()==1:
                timeLNew.append(time)
        return np.array(timeLNew)

    def getCatolog(self,T32,minValue=15000,minD=20):
        timeL0 = self.noClose(self.meanTime(T32[0],\
            T32[0].getDetec(minValue,minD)[0]))
        timeL1 = self.noClose(self.meanTime(T32[-1],\
            T32[-1].getDetec(minValue,minD)[0]))
        eL=[]
        for time in timeL0:
            dTime = timeL1-time
            absD  = np.abs(dTime)
            if (absD<minD).sum()==1:
                time0 = time
                time1 = timeL1[absD.argmin()]
                v0 = self.findFF(T32[0].slice(time0-5,time0+5))
                v1 = self.findFF(T32[-1].slice(time1-5,time1+5))
                eL.append([time0,time1,v0,v1])
        return np.array(eL)
    def handleDay(self,stations,day,workDir='../hsrRes/',T3L=[]):
        if len(T3L)==0:
            for station in stations:
                T3L.append(seism.Trace3(seism.getTrace3ByFileName(station.\
                    getFileNames(day),freq=[0.5, 90],delta0=-1)))
        if not os.path.exists(workDir):
            os.makedirs(workDir)
        eL = self.getCatolog(T3L)
        eFile = workDir + UTCDateTime(day).strftime('%Y%m%d')
        np.savetxt(eFile,eL)
        for i  in range(len(T3L)): 
            T3=T3L[i]
            bTime,eTime=T3.getTimeLim()
            if bTime<0:
                continue
            tmpAt,tmpBe,tmpAf,FFAt,FFBe,FFAf,dT= self.getAll(T3,eL)
            station = stations[i]
            staDir = workDir+station.name('.')+'/'
            if not os.path.exists(staDir):
                os.makedirs(staDir)
            file = staDir + UTCDateTime(day).strftime('%Y%m%d')+'_day.npy'
            self.saveSpec(file,self.fL0,tmpAt,tmpBe,tmpAf,FFAt,FFBe,FFAf,dT)
    def saveSpec(self,file, fL0,specAt,specBe,specAf,FFAt,FFBe,FFAf,dT):
        res = np.zeros([len(specAt),4,len(fL0)+1],np.float32)
        for i in range(len(specAt)):
            res[i,0,0]=dT[i]
            res[i,0,1:]=fL0
            res[i,1,0]=FFAt[i]
            res[i,1,1:]=np.abs(specAt[i])
            res[i,2,0]=FFBe[i]
            res[i,2,1:]=np.abs(specBe[i])
            res[i,3,0]=FFAf[i]
            res[i,3,1:]=np.abs(specAf[i])
        np.save(file,res)


    def loadSpec(self,file):
        data   = np.load(file)
        dT     = data[:,0,0]
        fL0    = data[:,0,1:]
        specAt = data[:,1,1:]
        specBe = data[:,2,1:]
        specAf = data[:,3,1:]
        FFAt   = data[:,1,0]
        FFBe   = data[:,2,0]
        FFAf   = data[:,3,0]
        return fL0,specAt,specBe,specAf,FFAt,FFBe,FFAf,dT
    def handleDays(self,stations,days,workDir='../hsrRes/'):
        for day in days:
            self.handleDay(stations,day,workDir)
    def loadStationsSpec(self,stations,workDir):
        dTL     = []
        fL0L    = []
        specAtL = []
        specBeL = []
        specAfL = []
        FFAtL   = []
        FFBeL   = []
        FFAfL   = []
        for station in stations:
            dTL     .append( [])
            fL0L    .append( [])
            specAtL .append( [])
            specBeL .append( [])
            specAfL .append( [])
            FFAtL   .append( [])
            FFBeL   .append( [])
            FFAfL   .append( [])
            staDir = workDir+station.name('.')+'/'
            for file in glob(staDir+'*day.npy'):
                tfL0,tspecAt,tspecBe,tspecAf,tFFAt,tFFBe,tFFAf,tdT=self.loadSpec(file)
                dTL[-1]     .append( tdT)
                fL0L[-1]    .append( tfL0)
                specAtL[-1] .append( tspecAt)
                specBeL[-1] .append( tSpecBe)
                specAfL[-1] .append( tSepcAf)
                FFAtL[-1]   .append( tFFAt)
                FFBeL[-1]   .append( tFFBe)
                FFAfL[-1]   .append( tFFAf)
            if len(f0L[-1])==0:
                continue
            dTL[-1]     =np.append(dTL[-1],axis=0)
            fL0L[-1]    =np.append(fL0L[-1],axis=0)
            specAtL[-1] =np.append(specAtL[-1],axis=0)
            specBeL[-1] =np.append(specBeL[-1],axis=0)
            specAfL[-1] =np.append(specAfL[-1],axis=0)
            FFAtL[-1]   =np.append(FFAtL[-1],axis=0)
            FFBeL[-1]   =np.append(FFBeL[-1],axis=0)
            FFAfL[-1]   =np.append(FFAfL[-1],axis=0)
        return fL0L,specAfL,specBeL,specAfL,FFAtL,FFBeL,FFAfL,dTL
    def add(self,fL0,spec,FF,dTL,minDT=-100,maxDT=100,minFF=3.0,maxFF=3.3):
        res = fL*0
        for i in range(spec):
            if FF[i]<minFF or FF[i]>maxF:
                continue
def toStr(num):
    tmp=''
    for n in num:
        tmp+=str(np.real(n))+' '
    return tmp


        



        


