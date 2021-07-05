from re import L

from matplotlib.cbook import delete_masked_points

from scipy import fft
from ..io import seism
import numpy as np
from obspy import UTCDateTime
from glob import glob
import os
from matplotlib import pyplot as plt
from multiprocessing import pool
import matplotlib
from ..plotTool import figureSet
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import gridspec
from torch import dtype, tensor
from ..mathTool.mathFunc import getDetec
import  torch 
from ..io.parRead import hander,DataLoader
from ..mathTool import mathFunc_bak
#多台联合测量车速似乎由于波场速度复杂，似乎不好测量，
#但如果不矫正直接叠加也可以
#set_per_process_memory_fraction
#torch.cuda.set_per_process_memory_fraction(0.1, 0)
dtype= torch.float32
device='cuda:0'
#matplotlib.rcParams['font.family']='Simhei'
class hsr:
    def __init__(self,f0=3.2,fmax=50,fL0=np.arange(0,25,0.01),uL=np.arange(1,10000,20)):
        self.f0=f0
        self.fmax= fmax
        self.fL0=fL0
        self.uL =uL
    def findFF_old(self,T3,comp=2):
        spec,fL=T3.getSpec(comp=comp)
        return self.FindFF(spec,fL)
    def findFF(self,T3,comp=2,NL=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13]),isBe=False,isAf=False):
        if isBe:
            NL=np.array([1,2,3,5,6,7,9,10,11])
        if isAf:
            NL=np.array([1,2,4,5,6,7,8,9,10,11])
        NL=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        return self.FindFFTimeInt(T3.Data()[:,comp],T3.Delta(),NL=NL)
    def findFFMul(self,T3L,comp=2,NL=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13]),isBe=False,isAf=False):
        if isBe:
            NL=np.array([1,2,3,5,6,7,9,10,11])
        if isAf:
            NL=np.array([1,2,4,5,6,7,8,9,10,11])
        NL=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        return self.FindFFTimeIntMul([T3.Data()[:,comp] for T3 in T3L],T3L[0].Delta(),NL=NL)
    def FindFFTimeInt(self,data,delta,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1,minMul=-1,NL=np.array([1,2,3])):
        #N=int(fmax/self.f0)
        df0= 1/(len(data)*delta)
        df=df0/NL.max()*0.5
        FFL = np.arange(minF,maxF,df)
        FFM = FFL.reshape([-1,1,1])*NL.reshape([1,1,-1])
        FFM = np.round(FFM/df0)*df0
        timeM = (np.arange(data.size)*delta).reshape([1,-1,1])
        dataT = tensor(data.reshape([1,-1,1]),dtype=dtype,device=device)
        timeLT= tensor(timeM,dtype=dtype,device=device)
        piT =tensor(np.pi,dtype=dtype,device=device)
        FFMT=tensor(FFM,dtype=dtype,device=device)
        #3.1933333333333449
        #M  =*torch.exp(-1j*FFMT*timeLT*piT*2)
        MS = dataT*torch.sin(-FFMT*timeLT*piT*2)
        MC = dataT*torch.cos(-FFMT*timeLT*piT*2)
        S =((MS.sum(axis=1)**2+MC.sum(axis=1)**2)**0.5).sum(axis=1).cpu().numpy()
        index = S.argmax()
        if reS:
            return FFL[index],S[index]
        else:
            return FFL[index]
    def FindFFTimeIntMul(self,dataL,delta,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1,minMul=-1,NL=np.array([1,2,3])):
        #N=int(fmax/self.f0)
        N = np.array([len(data)for data in dataL]).min()
        data = np.array([data[:N] for data in dataL ])
        df0= 1/(N*delta)
        df=df0/NL.max()*0.5
        FFL = np.arange(minF,maxF,df)
        FFM = FFL.reshape([1,-1,1,1])*NL.reshape([1,1,1,-1])
        FFM = np.round(FFM/df0)*df0
        timeM = (np.arange(data.shape[1])*delta).reshape([1,-1,1])
        dataT = tensor(data.reshape([-1,1,N,1]),dtype=dtype,device=device)
        timeLT= tensor(timeM,dtype=dtype,device=device)
        piT =tensor(np.pi,dtype=dtype,device=device)
        FFMT=tensor(FFM,dtype=dtype,device=device)
        #3.1933333333333449
        #M  =*torch.exp(-1j*FFMT*timeLT*piT*2)
        MS = dataT*torch.sin(-FFMT*timeLT*piT*2)
        MC = dataT*torch.cos(-FFMT*timeLT*piT*2)
        S =((MS.sum(axis=2)**2+MC.sum(axis=2)**2)**0.5).sum(axis=[0,2]).cpu().numpy()
        index = S.argmax()
        if reS:
            return FFL[index],S[index]
        else:
            return FFL[index]
    def FindFFTime(self,data,delta,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1,minMul=-1,NL=np.array([1,2,3])):
        #N=int(fmax/self.f0)
        df0= 1/(len(data)*delta)*0.5
        df=df0/NL.max()
        FFL = np.arange(minF,maxF,df)
        FFM = FFL.reshape([-1,1,1])*NL.reshape([1,1,-1])
        timeM = (np.arange(data.size)*delta).reshape([1,-1,1])
        dataT = tensor(data.reshape([1,-1,1]),dtype=dtype,device=device)
        timeLT= tensor(timeM,dtype=dtype,device=device)
        piT =tensor(np.pi,dtype=dtype,device=device)
        FFMT=tensor(FFM,dtype=dtype,device=device)
        #3.1933333333333449
        #M  =*torch.exp(-1j*FFMT*timeLT*piT*2)
        MS = dataT*torch.sin(-FFMT*timeLT*piT*2)
        MC = dataT*torch.cos(-FFMT*timeLT*piT*2)
        S =((MS.sum(axis=1)**2+MC.sum(axis=1)**2)**0.5).sum(axis=1).cpu().numpy()
        index = S.argmax()
        if reS:
            return FFL[index],S[index]
        else:
            return FFL[index]
    def FindFFTimeMul(self,dataL,delta,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1,minMul=-1,NL=np.array([1,2,3])):
        #N=int(fmax/self.f0)
        N = min([len(data)]for data in dataL)
        data = np.array([data[:N] for data in dataL ])
        df0= 1/(N*delta)*0.5
        df=df0/NL.max()
        FFL = np.arange(minF,maxF,df)
        FFM = FFL.reshape([1,-1,1,1])*NL.reshape([1,1,1,-1])
        timeM = (np.arange(data.shape[1])*delta).reshape([1,1,-1,1])
        dataT = tensor(data.reshape([-1,1,N,1]),dtype=dtype,device=device)
        timeLT= tensor(timeM,dtype=dtype,device=device)
        piT =tensor(np.pi,dtype=dtype,device=device)
        FFMT=tensor(FFM,dtype=dtype,device=device)
        #3.1933333333333449
        #M  =*torch.exp(-1j*FFMT*timeLT*piT*2)
        MS = dataT*torch.sin(-FFMT*timeLT*piT*2)
        MC = dataT*torch.cos(-FFMT*timeLT*piT*2)
        S =((MS.sum(axis=2)**2+MC.sum(axis=2)**2)**0.5).sum(axis=[0,2]).cpu().numpy()
        index = S.argmax()
        if reS:
            return FFL[index],S[index]
        else:
            return FFL[index]
    def FindFF(self,spec,fL,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1,minMul=-1):
        N=int(fmax/self.f0)
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
            FL=np.arange(FF*mul,fmax,FF)
            indexL=np.round((FL-fL[0])/df0).astype(np.int)
            s = np.abs(spec)[indexL].sum()
            if s>=S:
                S=s
                FF0=FF
        if S<minMul*np.real(spec).std()*2**0.5:
            FF0=-1
        if  reS:
            return FF0,S
        return FF0*mul
    def FindFFDetec(self,spec,fL,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1,minMul=1):
        threshold=minMul*np.real(spec).std()*2**0.5
        iL,vL=getDetec(np.abs(spec),minValue=threshold,minDelta=int(0.1/(fL[1]-fL[0])))
        FL = fL[iL]
        SL = np.abs(spec)[iL]
        index= np.abs(FL-(minF+maxF)/2).argmin()
        f= FL[index]
        S = vL[index]
        if f>minF and f<maxF:
            return f,S
        else:
            return -1,-1

    def adjustSpec_old(self,T3,f,f0=3.2,comp=2):
        spec,fL=T3.getSpec(isNorm=True,comp=comp)
        fL=fL/f*f0
        df0=(fL[1]-fL[0])
        indexL=((self.fL0-fL[0])/df0).astype(np.int)
        return spec[indexL]
    def adjustSpecNp(self,T3,f,f0=3.2,comp=2):
        data = T3.Data()
        data/=data.std()
        data=data[:,comp].reshape([1,-1])
        delta=T3.delta
        timeL=(np.arange(data.size)*delta).reshape([1,-1])
        fL = (self.fL0*f/f0).reshape([-1,1])
        M  = data*np.exp(-1j*fL*timeL*np.pi*2)
        return M.sum(axis=1)
    def adjustSpec(self,T3,f,f0=3.2,comp=2):
        data = T3.Data()
        data/=data.std()
        data=data[:,comp].reshape([1,-1])
        delta=T3.delta
        timeL=(np.arange(data.size)*delta).reshape([1,-1])
        fL = (self.fL0*f/f0).reshape([-1,1])

        dataT = tensor(data,dtype=dtype,device=device)
        timeLT = tensor(timeL,dtype=dtype,device=device)
        fLT= tensor(fL,dtype=dtype,device=device)
        #print(dataT.shape,fLT.shape,timeLT.shape)
        MS  = (dataT*torch.sin(-fLT*timeLT*np.pi*2)).sum(axis=1)
        MC  = (dataT*torch.cos(-fLT*timeLT*np.pi*2)).sum(axis=1)
        return 1j*MS.cpu().numpy()+MC.cpu().numpy()
    def adjustSpecNpV2(self,T3,f,f0=3.2,comp=2):
        data = T3.Data()
        data/=data.std()
        data=data[:,comp].reshape([1,-1])
        delta=T3.delta
        timeL=(np.arange(data.size)*delta).reshape([1,-1])
        fL = (self.fL0*f/f0).reshape([-1,1])
        M  = data*np.exp(-1j*fL*timeL*np.pi*2)
        return M.sum(axis=1),fL.reshape([-1])
    def adjustSpecV2(self,T3,f,f0=3.2,comp=2):
        data = T3.Data()
        data/=data.std()
        data=data[:,comp].reshape([1,-1])
        delta=T3.delta
        timeL=(np.arange(data.size)*delta).reshape([1,-1])
        fL = (self.fL0*f/f0).reshape([-1,1])

        dataT = tensor(data,dtype=dtype,device=device)
        timeLT = tensor(timeL,dtype=dtype,device=device)
        fLT= tensor(fL,dtype=dtype,device=device)
        #print(dataT.shape,fLT.shape,timeLT.shape)
        MS  = (dataT*torch.sin(-fLT*timeLT*np.pi*2)).sum(axis=1)
        MC  = (dataT*torch.cos(-fLT*timeLT*np.pi*2)).sum(axis=1)
        return 1j*MS.cpu().numpy()+MC.cpu().numpy(),fL.reshape([-1])
    def adjustSpecDt(self,T3,f,f0=3.2,comp=2,dt=0,isNormal=True):
        data = T3.Data()
        if isNormal:
            data/=data.std()
        data=data[:,comp].reshape([1,-1])
        delta=T3.delta
        timeL=(np.arange(data.size)*delta).reshape([1,-1])+dt
        fL = (self.fL0*f/f0).reshape([-1,1])

        dataT = tensor(data,dtype=dtype,device=device)
        timeLT = tensor(timeL,dtype=dtype,device=device)
        fLT= tensor(fL,dtype=dtype,device=device)
        #print(dataT.shape,fLT.shape,timeLT.shape)
        MS  = (dataT*torch.sin(-fLT*timeLT*np.pi*2)).sum(axis=1)
        MC  = (dataT*torch.cos(-fLT*timeLT*np.pi*2)).sum(axis=1)
        return 1j*MS.cpu().numpy()+MC.cpu().numpy(),fL.reshape([-1])
    def shiftDt(self,spec,dt,f,f0=3.2):
        fL = (self.fL0*f/f0)
        return spec*np.exp(-1j*fL*dt*np.pi*2)
    def getAll(self,T3,eL,minF=3.1,maxF=3.3,comp=2,bSec=5,eSec=15):
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
            #if np.abs(v0-v1)>0.1:
            #    continue
            t3=T3.slice(time-20,time+20)
            time = self.MeanTime(t3)
            if t3.bTime>0 and time>0:
                t3At=T3.slice(time-5,time+5)
                t3Be=T3.slice(time-eSec,time-bSec)#-25--5
                t3Af=T3.slice(time+bSec,time+eSec)
                if t3At.bTime<0 or t3Be.bTime<0 or t3Af.bTime<0:
                    continue
                FFAt.append(self.findFF(t3At,comp=comp))
                specAt.append(self.adjustSpec(t3At,FFAt[-1],comp=comp))
                FFBe.append(self.findFF(t3Be,comp=comp,isBe=True))
                specBe.append(self.adjustSpec(t3Be,FFBe[-1],comp=comp))
                FFAf.append(self.findFF(t3Af,comp=comp,isAf=True))
                specAf.append(np.abs(self.adjustSpec(t3Af,FFAf[-1],comp=comp)))
                dT.append(eL[i,1]-eL[i,0])
                print('handle',time,FFAt[-1]/3.2*80,FFBe[-1]/3.2*80,FFAf[-1]/3.2*80)
        return specAt,specBe,specAf,FFAt,FFBe,FFAf,dT
    def getOne(self,T3L,time0,DL,NS=1,comp=0,bSec=5,eSec=15,u0=3000,v=80,rotate=0):
        DL = DL-(DL[0]+DL[-1])/2
        DL  = DL*NS
        dtL = DL/u0
        uL = self.uL
        time0 = np.array(time0).mean()
        T3LBe = [T3L[i].slice(time0+dtL[i]-eSec,time0+dtL[i]-bSec).rotate(rotate)for i in range(len(T3L))]
        T3LAt = [T3L[i].slice(time0-5,time0+5).rotate(rotate)for i in range(len(T3L))]
        T3LAf = [T3L[i].slice(time0-dtL[i]+bSec,time0-dtL[i]+eSec).rotate(rotate)for i in range(len(T3L))]
        FFBe=self.findFFMul(T3LBe,comp=comp)
        specLBe = [self.adjustSpecDt(t3,FFBe,comp=comp,isNormal=False)[0]for t3 in T3LBe]
        FFAt=self.findFFMul(T3LAt,comp=comp)
        specLAt = [self.adjustSpecDt(t3,FFAt,comp=comp,isNormal=False)[0]for t3 in T3LAt]
        FFAf=self.findFFMul(T3LAf,comp=comp)
        specLAf = [self.adjustSpecDt(t3,FFAf,comp=comp,isNormal=False)[0]for t3 in T3LAf]
        print(FFBe,FFAt,FFAf)
        specBeL = np.array([ np.array([ self.shiftDt(specLBe[i],T3LBe[i].bTime-T3LBe[0].bTime-DL[i]/u,FFBe) for i in range(len(T3LBe))]).mean(axis=0)for u in uL])
        specAtL = np.array([ np.array([ self.shiftDt(specLAt[i],T3LAt[i].bTime-T3LAt[0].bTime-DL[i]/u,FFAt) for i in range(len(T3LAt))]).mean(axis=0)for u in uL])
        specAfL = np.array([ np.array([ self.shiftDt(specLAf[i],T3LAf[i].bTime-T3LAf[0].bTime+DL[i]/u,FFAf) for i in range(len(T3LAf))]).mean(axis=0)for u in uL])

        return specBeL,specAtL,specAfL,FFBe,FFAt,FFAf,self.uL
    def plotUSpec(self,f,specL,filename,f0=3.2):
        uL=self.uL
        plt.close()
        plt.figure(figsize=[5,5])
        plt.subplot(2,1,1)
        plt.pcolormesh(self.fL0*f/f0,uL,np.array(np.abs(specL)),cmap='jet',shading='gouraud',rasterized=True)
        #plt.xlim([self.fL0[0]*f/f0,fmax])
        plt.xlabel('f/Hz')
        plt.ylabel('u/(m/s)')
        ##plt.colorbar()
        plt.subplot(2,1,2)
        specLNew = specL
        fNew = self.fL0
        i=np.abs(specLNew).max(axis=1).argmax()
        plt.plot(fNew,np.abs(specLNew[i]))
        index = np.abs(specLNew[i]).argmax()
        print(self.fL0[index])
        plt.xlabel('f/Hz')
        print(uL[i])
        plt.xlim([self.fL0[0],self.fL0[-1]])
        plt.savefig(filename,dpi=600)
        plt.close()
    def handleDayUSpec(self,stations,day,workDir='../hsrRes/',comp=0,rotate=0,bSec=20,eSec=28,T3L=[],V=82,maxDV=2,V0=80,f0=3.2,DL=[]):
        if len(T3L)==0:
            T3L=[]
            for station in stations:
                print('loading', station)
                T3L.append(seism.Trace3(seism.getTrace3ByFileName(station.\
                    getFileNames(day),freq=[0.5, 40],delta0=0.01,corners=4)).rotate(rotate))
        if not os.path.exists(workDir):
            os.makedirs(workDir)
        eL = self.getCatolog(T3L)
        eFile = workDir + UTCDateTime(day).strftime('%Y%m%d')
        #np.savetxt(eFile,eL)
        specBeLLN,specAtLLN,specAfLLN,FFBeLN,FFAtLN,FFAfLN,uLLN =[],[],[],[],[],[],[]
        specBeLLS,specAtLLS,specAfLLS,FFBeLS,FFAtLS,FFAfLS,uLLS =[],[],[],[],[],[],[]
        nCountBe=0
        nCountAf=0
        sCountBe=0
        sCountAf=0
        for e in eL:
            time0  = e[0]
            time1  = e[1]
            v0     = e[2]/f0*V0
            v1     = e[3]/f0*V0
            #print(e)
            if np.abs(v0-v1)/v0>1/20:
                continue
            if np.abs((v0+v1)/2-V0)>maxDV*1.5:
                continue
            if time0 < time1:
                specBeL,specAtL,specAfL,FFBe,FFAt,FFAf,uL = self.getOne(T3L,[time0,time1],DL,NS=1,bSec=bSec,eSec=eSec,rotate=rotate,comp=comp)
                if np.abs(FFBe/f0*V0-V)<maxDV:
                    specBeLLN.append(specBeL)
                    nCountBe+=1
                if np.abs(FFAf/f0*V0-V)<maxDV:
                    specAfLLN.append(specAfL)
                    nCountAf+=1
            else:
                specBeL,specAtL,specAfL,FFBe,FFAt,FFAf,uL = self.getOne(T3L,[time0,time1],DL,NS=-1,bSec=bSec,eSec=eSec,rotate=rotate,comp=comp)
                if np.abs(FFBe/f0*V0-V)<maxDV:
                    specBeLLS.append(specBeL)
                    sCountBe+=1
                if np.abs(FFAf/f0*V0-V)<maxDV:
                    specAfLLS.append(specAfL)
                    sCountAf+=1
            print(nCountBe,nCountAf,sCountAf,sCountBe)
        return np.abs(np.array(specBeLLN)).mean(axis=0),np.abs(np.array(specAfLLN)).mean(axis=0),\
            np.abs(np.array(specBeLLS)).mean(axis=0),np.abs(np.array(specAfLLS)).mean(axis=0)
    def findBF(self,specL,f0=-1,f1=10000):
        i0 = np.abs(self.fL0-f0).argmin()
        i1 = np.abs(self.fL0-f1).argmin()
        i=np.abs(specL[:,i0:i1]).max(axis=1).argmax()
        index = np.abs(specL[i,i0:i1]).argmax()+i0
        return self.fL0[index]
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
        data = np.abs(t3.Data()[:,2])
        if len(data)>0:
            bTime,eTime=t3.getTimeLim()
            delta = t3.Delta()
            indexL = np.arange(len(data))
            iM=(indexL*data).sum()/data.sum()
            print(iM,data.argmax(),np.where(data>0)[0].mean(),np.where(data>0)[0])
            time1=iM*delta+bTime.timestamp
            return time1
        else:
            return -1
    def MeanTime_(self,t3):
        data = np.abs(t3.Data()[:,2])
        #data[data<data.max()*0.3] =0 
        if len(data)>0:
            bTime,eTime=t3.getTimeLim()
            delta = t3.Delta()
            indexL = np.arange(len(data))
            iM=(indexL*data).sum()/data.sum()
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
    def handleDay(self,stations,day,workDir='../hsrRes/',compL=[2],rotate=0,bSecL=[5,10,15,20],eSecL=[15,20,25,30],T3L=[]):
        if len(T3L)==0:
            T3L=[]
            for station in stations:
                print('loading', station)
                T3L.append(seism.Trace3(seism.getTrace3ByFileName(station.\
                    getFileNames(day),freq=[0.5, 40],delta0=0.01,corners=4)))
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
            if rotate!=0:
                print('rotating')
                T3 = T3.rotate(rotate)
                print('rotated')
            bTime,eTime=T3.getTimeLim()
            for comp in compL:
                for j in range(len(bSecL)):
                    print('handle',comp)
                    bSec=bSecL[j]
                    eSec=eSecL[j]
                    tmpAt,tmpBe,tmpAf,FFAt,FFBe,FFAf,dT= self.getAll(T3,eL,comp=comp,bSec=bSec,eSec=eSec)
                    station = stations[i]
                    staDir = workDir+'%d_%d/'%(bSec,eSec)+station.name('.')+'/'
                    if not os.path.exists(staDir):
                        os.makedirs(staDir)
                    file = staDir + UTCDateTime(day).strftime('%Y%m%d')+('_day%d.npy'%comp)
                    self.saveSpec(file,self.fL0,tmpAt,tmpBe,tmpAf,FFAt,FFBe,FFAf,dT)
    def plotDay(self,stations,day,workDir='../hsrRes/',comp=2,rotate=0,T3L=[]):
        if len(T3L)==0:
            T3L=[]
            for station in stations:
                print('loading', station)
                T3L.append(seism.Trace3(seism.getTrace3ByFileName(station.\
                    getFileNames(day),freq=[0.5, 45],delta0=0.01)))
        if not os.path.exists(workDir):
            os.makedirs(workDir)
        eL = self.getCatolog(T3L)
        eFile = workDir + UTCDateTime(day).strftime('%Y%m%d')
        np.savetxt(eFile,eL)
        timeL0  = eL[:,0]
        timeL1  = eL[:,1]
        v0L    = eL[:,2]
        v1L    = eL[:,3]
        vL = []
        specL = []
        for T3 in T3L:
            for i in range(len(timeL0)):
                time0=timeL0[i]
                time1=timeL1[i]
                time=(time0+time1)/2
                t3=T3.slice(time-20,time+20) 
                Time = self.MeanTime(t3)
                t3=T3.slice(Time-10,Time+10)
                f=self.findFF(t3,comp=comp)
                v=f/3.2*80*3.6
                #d=stations[0].dist(stations[-1])
                #v= d/np.abs(time1-time0)*3600
                vL.append(v)
                print(v)
                spec,fL=t3.getSpec(isNorm=True,comp=comp)
                spec /= np.real(spec).std()
                specL.append(spec)
        V = np.array(vL)
        indexL=(-V).argsort()
        Vsort = V[indexL]
        N=len(specL[0])-10
        Spec = np.zeros([N,len(V)])
        for i in np.arange(len(indexL)):
            Spec[:,i]=np.abs(specL[i][:N])
        Spec= Spec[:,indexL]
        plt.close()
        plt.figure(figsize=[6,3.2])
        plt.subplot(1,2,1)
        plt.plot(Vsort,'k')
        plt.xlabel('Event Index')
        plt.xlim([0,len(Vsort)])
        plt.ylabel('V (km/h)')
        plt.subplot(1,2,2)
        plt.pcolor(np.arange(len(V)),fL[:N],np.log(Spec),cmap='jet')
        #plt.colorbar()
        clb = plt.colorbar()
        clb.set_label('log(A)')
        plt.xlabel('Event Index')
        plt.ylabel('f (Hz)')
        plt.ylim([0,70])
        plt.savefig('VL.jpg',dpi=300)
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
        print('data shape',data.shape)
        dT     = data[:,0,0]
        fL0    = data[:,0,1:]
        specAt = data[:,1,1:]
        specBe = data[:,2,1:]
        specAf = data[:,3,1:]
        FFAt   = data[:,1,0]
        FFBe   = data[:,2,0]
        FFAf   = data[:,3,0]
        return fL0,specAt,specBe,specAf,FFAt,FFBe,FFAf,dT
    def handleDays(self,stations,days,workDir='../hsrRes/',compL=[0,1,2],rotate=0,bSecL=[5,10,15,20],eSecL=[15,20,25,30],isPool=False):
        #handleDay(self,stations,day,workDir='../hsrRes/',compL=[2],rotate=0,#bSecL=[5,10,15,20],eSecL=[15,20,25,30],T3L=[])
        if not isPool:
            for day in days:
                print('doing',day)
                self.handleDay(stations,day,workDir,compL=compL,rotate=rotate,\
                    bSecL=bSecL,eSecL=eSecL)
        else:
            paraL=[]
            for day in days:
                paraL.append([stations,day,workDir,compL,rotate,\
                    bSecL,eSecL,[]])
            with pool.Pool(6) as p:
                p.map(handleDay,paraL)
            #H=hander(self,paraL)
            #DataLoader(H,batch_size=12,num_workers=6)
            #for tmp in H:
            #    for t in tmp:
            #        pass
    def loadStationSpec(self,station,workDir,comp=2,bandStr=''):
        dT     = []
        fL0    = []
        specAt = []
        specBe = []
        specAf = []
        FFAt   = []
        FFBe   = []
        FFAf  = []
        staDir = workDir+'/'+station.name('.')+'/'
        print(staDir+bandStr+'*_day%d.npy'%comp)
        for file in glob(staDir+'*_day%d.npy'%comp):
            tfL0,tspecAt,tspecBe,tspecAf,tFFAt,tFFBe,tFFAf,tdT=self.loadSpec(file)
            dT     .append( tdT)
            fL0    .append( tfL0)
            specAt .append( tspecAt)
            specBe .append( tspecBe)
            specAf .append( tspecAf)
            FFAt  .append( tFFAt)
            FFBe  .append( tFFBe)
            FFAf  .append( tFFAf)
        if len(fL0)!=0:
            print('find record day',len(dT))
            dT    =np. concatenate(dT,axis=0)
            fL0    =np. concatenate(fL0,axis=0)
            specAt=np. concatenate(specAt,axis=0)
            specBe =np. concatenate(specBe,axis=0)
            specAf =np. concatenate(specAf,axis=0)
            FFAt   =np. concatenate(FFAt,axis=0)
            FFBe   =np. concatenate(FFBe,axis=0)
            FFAf   =np. concatenate(FFAf,axis=0)
        return fL0,specAt,specBe,specAf,FFAt,FFBe,FFAf,dT
    def loadStationsSpec(self,stations,workDir,comp=2,isAdd=False,minDT=-100,maxDT=100,dF=0.1,bandStr='',**kwags):
        dTL     = []
        fL0L    = []
        specAtL = []
        specBeL = []
        specAfL = []
        FFAtL   = []
        FFBeL   = []
        FFAfL   = []
        for station in stations:
            fL0,specAt,specBe,specAf,FFAt,FFBe,FFAf,dT=\
                self.loadStationSpec(station,workDir,comp=comp,bandStr=bandStr)
            dTL     .append( dT)
            fL0L    .append( fL0)
            specAtL .append( specAt)
            specBeL .append(specBe)
            specAfL .append(specAf)
            FFAtL   .append( FFAt)
            FFBeL   .append( FFBe)
            FFAfL   .append( FFAf)
            if isAdd and len(fL0L[-1])>0:
                fL0L[-1]= fL0L[-1][0:kwags['perN']]
                specAtL[-1]=self.add(fL0,specAtL[-1],FFAtL[-1],dTL[-1],\
                    minDT=minDT,maxDT=maxDT,minFF=3.2-dF,maxFF=3.2+dF,F0=FFAtL[-1],dF0=0.05*3.2,**kwags)
                specBeL[-1]=self.add(fL0,specBeL[-1],FFBeL[-1],dTL[-1],\
                    minDT=minDT,maxDT=maxDT,minFF=3.2-dF,maxFF=3.2+dF,F0=FFAtL[-1],dF0=0.05*3.2,**kwags)
                specAfL[-1]=self.add(fL0,specAfL[-1],FFAfL[-1],dTL[-1],\
                    minDT=minDT,maxDT=maxDT,minFF=3.2-dF,maxFF=3.2+dF,F0=FFAtL[-1],dF0=0.05*3.2,**kwags)
        return fL0L,specAtL,specBeL,specAfL,FFAtL,FFBeL,FFAfL,dTL
    def add(self,fL0,spec,FF,dT,minDT=-100,maxDT=100,minFF=3.0,maxFF=3.3,F0=[],dF0=-1,isRemove=False,perN=1):
        if perN > len(spec):
            perN = len(spec)
        res = fL0[0:perN]*0
        count= np.zeros(perN).reshape([-1,1])
        maxSum=0
        for i in range(len(spec)):
            index = int(i/len(spec)*perN)
            if FF[i]<minFF or FF[i]>maxFF:
                continue
            if dT[i]<minDT or dT[i]>maxDT:
                continue
            if len(F0)>0 and np.abs(FF[i]-F0[i])>dF0:
                continue
            if isRemove:
                E = np.abs(self.calE(N=8,L=25,v=25*FF[i],f=fL0[0]).reshape([1,-1]))
                E[E<1]=1
                spec[i:i+1]/= E
            res[index:index+1] += spec[i:i+1]
            maxSum = spec[i:i+1].max()
            count[index]+=1
        print('add',count,'maxSum',maxSum)
        return res/count
    def showSpec(self,fL0L,specAtL,specBeL,specAfL,stations,head='',workDir='',maxF=20,v=3000,isPlot=True):
        n= np.arange(1,10)
        FBeL = []
        FAfL = []
        SBeL=[]
        SAfL=[]
        vL =[]

        for i in range(len(stations)):
            specAt = specAtL[i]
            specBe = specBeL[i]
            specAf = specAfL[i]
            if len(specAt)==0:
                continue
            station = stations[i]
            staName = station.name('.')+'_'+head+'_v=%d'%v
            filename = workDir+'/'+ staName+'.eps'
            figureSet.init()
            plt.close()
            plt.figure(figsize=[3.5,2.5])
            FBeL.append([])
            FAfL.append([])
            SBeL.append([])
            SAfL.append([])
            vL.append([])
            for index in range(len(specBe)):
                FBe,SBe= self.FindFF(specBe[index],fL0L[0][0,:],minF=5.06,maxF=5.21,avoidF=3.2,fmax=9,reS=True,mul=1,minMul=2)
                FAf,SAf = self.FindFF(specAf[index],fL0L[0][0,:],minF=4.8,maxF=4.94,avoidF=3.2,fmax=9,reS=True,mul=1,minMul=2)
                fBeL = FBe*n
                fAfL = FAf*n
                V = (FBe+FAf)/(FBe-FAf+0.000001)*80
                if FBe<0 or FAf<0:
                    V=-10
                FBeL[-1].append(FBe)
                FAfL[-1].append(FAf)
                SBeL[-1].append(SBe)
                SAfL[-1].append(SAf)
                vL[-1].append(V)
            if not isPlot:
                continue
            '''
            for f in fBe:
                plt.plot([f,f],[-1,2000],'-b',linewidth=0.3)
            for f in fAf:
                plt.plot([f,f],[-1,2000],'-r',linewidth=0.3)
            for f in fBeL:
                plt.plot([f,f],[-1,2000],'-.b',linewidth=0.3)
            for f in fAfL:
                plt.plot([f,f],[-1,2000],'-.r',linewidth=0.3)     
            '''    
            for i in range(len(specAt)*0+1):
                hAt=plt.plot(fL0L[i][i,:],specAt[i],'k',linewidth=0.5)
                hBe=plt.plot(fL0L[i][i,:],specBe[i],'b',linewidth=0.5)
                hAf=plt.plot(fL0L[i][i,:],specAf[i],'r',linewidth=0.5)
            #if not np.isinf(vL[-1][0]):
            #    figureSet.setABC('Fbe:%.3f FAf:%.3f V:%d'%(FBeL[-1][0],FAfL[-1][0],int(vL[-1][0])))
            #plt.legend((hAt,hBe,hAf),['at','before','after'])
            plt.xlim([0,maxF])
            #plt.ylim([-1,800])
            plt.title(staName+' V: %.2f'%V)
            plt.xlabel('$f$/Hz')
            plt.ylabel('$A$')
            NS,comp,time = head.split('_') 
            figureSet.setABC('%s %s s'%(NS,time),[0.75,0.95])
            fileDir = os.path.dirname(filename)
            if not os.path.exists(fileDir):
                os.makedirs(fileDir)
            plt.savefig(filename,dpi=300)
            plt.close()
        return FBeL,FAfL,SBeL,SAfL,vL
    def anSpec(self,fL0L,specAtL,specBeL,specAfL,stations,minF=4.6,maxF=5.4,avoidF=3.2,fmax=9,v=3000):
        n= np.arange(1,10)
        fBe = v/(v-80)*3.2*n*25/32
        fAf = v/(v+80)*3.2 *n*25/32
        FBeL = []
        FAfL = []
        SBeL=[]
        SAfL=[]
        vL =[]
        for i in range(len(stations)):
            for j in range(len(specAtL[i])):
                specAt = specAtL[i][j:j+1]
                specBe = specBeL[i][j:j+1]
                specAf = specAfL[i][j:j+1]
                if len(specAt)==0:
                    continue

                FBe,SBe= self.FindFF(specBe[0],fL0L[0][0,:],minF=minF,maxF=maxF,avoidF=avoidF,fmax=fmax,reS=True,mul=1)
                #print(specAf.size,fL0L[0].size)
                FAf,SAf = self.FindFF(specAf[0],fL0L[0][0,:],minF=minF,maxF=maxF,avoidF=avoidF,fmax=fmax,reS=True,mul=1)
                fBeL = FBe*n
                fAfL = FAf*n
                V = (FBe+FAf)/(FBe-FAf)*80
                FBeL.append(FBe)
                FAfL.append(FAf)
                SBeL.append(SBe)
                SAfL.append(SAf)
                vL.append(V)
                '''
                for f in fBe:
                    plt.plot([f,f],[-1,2000],'-b',linewidth=0.3)
                for f in fAf:
                    plt.plot([f,f],[-1,2000],'-r',linewidth=0.3)
                for f in fBeL:
                    plt.plot([f,f],[-1,2000],'-.b',linewidth=0.3)
                for f in fAfL:
                    plt.plot([f,f],[-1,2000],'-.r',linewidth=0.3)     
                '''    
        return FBeL,FAfL,SBeL,SAfL,vL
    def showSpecs(self,fL0L,specAfL,specBeL,specAtL,stations,head='',workDir='',maxF=20):
        specAtL=specAtL.copy()
        specBeL=specBeL.copy()
        specAfL=specAfL.copy()
        specAtL.remove([])
        specBeL.remove([])
        specAfL.remove([])
        specAtL    =np. concatenate(specAtL,axis=0)
        specBeL    =np. concatenate(specBeL,axis=0)
        specAfL     =np. concatenate(specAfL,axis=0)
        specAtL/= specAtL.max()
        specBeL/= specBeL.max()
        specAfL/= specAfL.max()
        N = int((maxF-fL0L[0][0][0])/(fL0L[0][0][1]-fL0L[0][0][0]))
        C = np.concatenate([specBeL[:,:N].reshape([-1,N,1]),\
            specAfL[:,:N].reshape([-1,N,1]),\
            specAtL[:,:N].reshape([-1,N,1]),],axis=2)
        #sC/=C.max(axis=0)
        plt.title('0-%.2f Hz'%maxF)
        plt.imshow(C**0.5,aspect=50)
        plt.savefig(workDir+'/'+'allSta_+%s_%.2f.jpg'%(head,maxF),dpi=300)
        plt.close()
    def addSacsL(self,sacsL,l =np.arange(-60,60)*32/1000+16,timeL=np.arange(-2500,2500)*0.01,v=80):
        distL0 = self.getDistL(sacsL)
        data = np.zeros([len(timeL),3])
        for dist in l:
            sacs,dist0 = self.getSacs(sacsL,disL0,dist)
            time = dist/v
            if dist*dist>0:
                shift =[1,1,1]
            else:
                shift =[-1,-1,1]
            self.addSacs(data,timeL,time,shift)
    def addSacs(self,data,timeL,time,shift):
        delta = timeL[1]-timeL[0]
        for i in range(3):
            sac = sacs[i]
            timeB = time+ sac.stats['sac']['b']
            indexB =int((timeB-timeL[0])/delta)
            data[indexB:indexB+sac.data.shape[0],i]+=sac.data*shift[i]
    def getDistL(self,sacsL,dist):
        distL = []
        for sacs in sacsL:
            distL.append(sacs[0].stats['sac']['distance'])
        return distL
    def getSacs(self,sacsL,disL0,dist):
        index = np.abs(np.abs(distL0)-np.abs(dist)).argmin()
        return sacsL[index],distL0[indexL]
    def realMS(self,X,minX=-1,maxX=30000,minNum=3,maxStd=30000):
        M = np.zeros(len(X))
        S = np.zeros(len(X))
        for i in range(len(X)):
            x = np.array(X[i])
            x = x[x>minX]
            x = x[x<maxX]
            if len(x)<minNum:
                M[i]=-300
                S[i]=0
            else:
                M[i]= x.mean()
                S[i]= x.std()
                if x.std()>maxStd:
                    M[i]= -100000
        return M,S 
    def plotFV(self,FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL,eSecL,dTimeL,workDir,head='',marker='.',strL='ab',maxStd=20000):
        for comp in range(3):
            FBeLNL = np.array(FBeLNM[comp])
            FAfLNL = np.array(FAfLNM[comp])
            vLNL = np.array(vLNM[comp])
            FBeLSL = np.array(FBeLSM[comp])
            FAfLSL = np.array(FAfLSM[comp])
            vLSL = np.array(vLSM[comp])
            midSecL = (eSecL+bSecL)/2
            plt.close()
            figureSet.init()
            plt.figure(figsize=[3.5,3.5])
            plt.subplot(2,1,1)
            plt.ylim([4,6])
            plt.xlim([-5000,5000])
            for i in range(len(dTimeL)):
                dTime=dTimeL[i]
                '''
                plt.plot((midSecL-dTime)*80,FBeLNL[:,i],marker+'b')
                plt.plot((midSecL-dTime)*80,FAfLNL[:,i],marker+'r')
                plt.plot((-midSecL+dTime)*80,FBeLSL[:,i],marker+'b')
                plt.plot((-midSecL+dTime)*80,FAfLSL[:,i],marker+'r')
                '''
                #print(FBeLNL[:,i].shape,*self.realMS(FBeLNL[:,i],minX=3,maxX=6,minNum=3),marker=marker,color='b')
                #print(((midSecL-dTime)*80).shape)
                print(i)
                plt.errorbar((midSecL-dTime)*80,*self.realMS(FBeLNL[:,i],minX=3,maxX=6,minNum=4),fmt=marker+'b',markersize=0.5,capsize=2,elinewidth=0.25,capthick=0.2)
                plt.errorbar((midSecL-dTime)*80,*self.realMS(FAfLNL[:,i],minX=3,maxX=6,minNum=4),fmt=marker+'r',markersize=0.5,capsize=2,elinewidth=0.25,capthick=0.2)
                plt.errorbar((-midSecL+dTime)*80,*self.realMS(FBeLSL[:,i],minX=3,maxX=6,minNum=4),fmt=marker+'b',markersize=0.5,capsize=2,elinewidth=0.25,capthick=0.2)
                plt.errorbar((-midSecL+dTime)*80,*self.realMS(FAfLSL[:,i],minX=3,maxX=6,minNum=4),fmt=marker+'r',markersize=0.5,capsize=2,elinewidth=0.25,capthick=0.2)
            #plt.xlim([-60,60])
            plt.ylabel('$f$/Hz')
            figureSet.setABC('(%s)'%strL[0],[0.01,0.98],c='k')
            plt.subplot(2,1,2)
            plt.ylim([0,5000])
            for i in range(len(dTimeL)):
                dTime=dTimeL[i]
                '''
                plt.plot((midSecL-dTime)*80,vLNL[:,i],marker+'k')
                plt.plot((-midSecL+dTime)*80,vLSL[:,i],marker+'k')
                '''
                plt.errorbar((midSecL-dTime)*80,*self.realMS(vLNL[:,i],minX=1500,maxX=5000,minNum=4,maxStd=maxStd),fmt=marker+'k',markersize=0.5,capsize=2,elinewidth=0.25,capthick=0.2)
                plt.errorbar((-midSecL+dTime)*80,*self.realMS(vLSL[:,i],minX=1500,maxX=5000,minNum=4,maxStd=maxStd),fmt=marker+'k',markersize=0.5,capsize=2,elinewidth=0.25,capthick=0.2)
            plt.xlim([-5000,5000])
            plt.xlabel('$distance$/m')
            plt.ylabel('$v$/(m/s)')
            figureSet.setABC('(%s)'%strL[1],[0.01,0.98],c='k')
            if not os.path.exists(workDir):
                os.makedirs(workDir)
            plt.savefig(workDir+head+'%d.eps'%comp)
    def plotSV(self,SBeLNM,SAfLNM,vLNM,SBeLSM,SAfLSM,vLSM,bSecL,eSecL,dTimeL,workDir,head='',markerL='.*d'):
        plt.close()
        plt.subplot(2,1,1)
        for comp in range(1):
            marker = markerL[comp]
            SBeLNL = np.array(SBeLNM[comp])/(np.array(SBeLNM)**2).sum(axis=0)**0.5
            SAfLNL = np.array(SAfLNM[comp])/(np.array(SAfLNM)**2).sum(axis=0)**0.5
            vLNL = np.array(vLNM[comp])
            SBeLSL = np.array(SBeLSM[comp])/(np.array(SBeLSM)**2).sum(axis=0)**0.5
            SAfLSL = np.array(SAfLSM[comp])/(np.array(SAfLSM)**2).sum(axis=0)**0.5
            vLSL = np.array(vLSM[comp])
            midSecL = (eSecL+bSecL)/2
            plt.xlim([-60,60])
            #plt.ylim([2,3])
            for i in range(len(dTimeL)):
                dTime=dTimeL[i]
                plt.plot(midSecL-dTime,SBeLNL[:,i],marker+'b')
                plt.plot(midSecL-dTime,SAfLNL[:,i],marker+'r')
                plt.plot(-midSecL+dTime,SBeLSL[:,i],marker+'b')
                plt.plot(-midSecL+dTime,SAfLSL[:,i],marker+'r')
        plt.subplot(2,1,2)
        for comp in range(3):
            marker = markerL[comp]
            SBeLNL = np.array(SBeLNM[comp])
            SAfLNL = np.array(SAfLNM[comp])
            vLNL = np.array(vLNM[comp])
            SBeLSL = np.array(SBeLSM[comp])
            SAfLSL = np.array(SAfLSM[comp])
            vLSL = np.array(vLSM[comp])
            midSecL = (eSecL+bSecL)/2
            plt.xlim([-60,60])
            #plt.subplot(2,1,2)
            plt.ylim([0,5000])
            for i in range(len(dTimeL)):
                dTime=dTimeL[i]
                plt.plot(midSecL-dTime,vLNL[:,i],marker+'k')
                plt.plot(-midSecL+dTime,vLSL[:,i],marker+'k')
            plt.xlim([-60,60])
            if not os.path.exists(workDir):
                os.makedirs(workDir)
            plt.savefig(workDir+head+'%d.pdf'%comp)
    def plotSV2(self,SBeLNM,SAfLNM,vLNM,SBeLSM,SAfLSM,vLSM,bSecL,eSecL,dTimeL,workDir,head='',markerL='.*d'):
        plt.close()
        compL='RTZ'
        figureSet.init()
        plt.figure(figsize=[3,3])
        figIndex='bcd'
        for comp in range(3):
            plt.subplot(3,1,comp+1)
            SBeLNL = np.array(SBeLNM[comp])**2/(np.array(SBeLNM)**2).sum(axis=0)
            SAfLNL = np.array(SAfLNM[comp])**2/(np.array(SAfLNM)**2).sum(axis=0)
            SBeLSL = np.array(SBeLSM[comp])**2/(np.array(SBeLSM)**2).sum(axis=0)
            SAfLSL = np.array(SAfLSM[comp])**2/(np.array(SAfLSM)**2).sum(axis=0)
            midSecL = (eSecL+bSecL)/2
            plt.xlim([-60,60])
            beNT=(SBeLNL*0+midSecL.reshape([-1,1])).reshape([-1]).tolist()
            beST=(SBeLSL*0+midSecL.reshape([-1,1])).reshape([-1]).tolist()
            afNT=(SAfLNL*0-midSecL.reshape([-1,1])).reshape([-1]).tolist()
            afST=(SAfLSL*0-midSecL.reshape([-1,1])).reshape([-1]).tolist()
            beNS=(SBeLNL).reshape([-1]).tolist()
            beSS=(SBeLSL).reshape([-1]).tolist()
            afNS=(SAfLNL).reshape([-1]).tolist()
            afSS=(SAfLSL).reshape([-1]).tolist()
            #pc=plt.hist2d(beNT+beST+afNT+afST,beNS+beSS+afNS+afSS,range=[[-50,50],[0,1]],cmap='bwr')
            pc=plt.hist2d(beNT+beST+afNT+afST,beNS+beSS+afNS+afSS,range=[[-50,50],[0,1]],cmap='bwr')
            cb=plt.colorbar()
            cb.set_label('$num$')
            plt.ylabel('$'+compL[comp]+'$'+'/$All$')
            figureSet.setABC('(%s)'%figIndex[comp],[0.01,0.98],c='k')
            if comp==2:
                plt.xlabel('$t$/s')
            else:
                a=plt.xticks()[0][1:-1]
                plt.xticks(a,['' for tmp in a])
        '''
        plt.subplot(5,1,5)
        pc=plt.hist2d(beNT+beST+afNT+afST,beNS+beSS+afNS+afSS,range=[[-50,50],[0,1]],cmap='bwr',cmin=0, cmax=7000)
        cb=plt.colorbar()
        cb.set_label('count')
        '''
        if not os.path.exists(workDir):
            os.makedirs(workDir)
        plt.savefig(workDir+head+'.eps')
    def calE(self,N=16,L=25,v=80,f=np.arange(0,100,0.01)):
        E = (f*0).astype(np.complex128)
        for i in range(N):
            E+=np.exp(-1j*f*np.pi*2*i*L/v)
        return E
    def calr(self,N=16,v=80,U=3000,L=np.arange(100)*32+16,A=1,timeL=np.arange(-40,40,0.01),dd=[4, 6.5, 18.5 ,21],randP=0,D=100):
        r = timeL*0
        UL =  U*(L*0+1)
        if randP>0:
            for i in range(int((L.max()-L.min())/D)):
                index= int(np.random.rand()*L.size)
                index= np.abs(np.abs(L-L.min())-i*D).argmin()
                print(index)
                UL[index:]=UL[index:]+2*(i%2-0.5)*randP*U
        index0 = np.abs(L).argmin()
        print(UL)
        for l in L:
            index = np.abs(L-l).argmin()
            if index>index0:
                U = 1/(1/UL[index0:index]).mean()
            elif index<index0:
                U = 1/(1/UL[index:index0]).mean()
            else:
                U = 1/(1/UL[index0:index0+1]).mean()
            #print(U)
            dt = np.abs(l)/U
            for d in dd:
                t0 = (l+d)/v
                t=t0+dt
                index = np.abs(timeL-t).argmin()
                r[index]+= A/np.abs(l)
            #r[index+1]-= A/np.abs(l)
        return r
    def calR(self,N=16,v=80,U=[3000],L=np.arange(100)*32+16,A=1,timeL=np.arange(-50,50,0.01),dd=[4, 6.5, 18.5 ,21],D=100,randP=0):
        r =self.calr(N=N,v=v,U=U[0],L=L,A=1,timeL=timeL,dd=dd,D=D,randP=randP)*0
        for u in U:
            r += self.calr(N=N,v=v,U=u,L=L,A=1,timeL=timeL,dd=dd,D=D,randP=randP)
        return np.fft.fft(r),np.arange(len(timeL))/len(timeL)*1/(timeL[1]-timeL[0])
    def plotE(self,N=16,L=25,v=80,f=np.arange(0,100,0.01),head=''):
        E = self.calE(N=N,L=L,v=v,f=f)
        plt.close()
        plt.figure(figsize=[4,4])
        plt.plot(f,np.abs(E),linewidth=0.5)
        plt.xlabel('f/Hz')
        plt.ylabel('Amplitude')
        plt.xlim([0,20])
        plt.savefig('../hsrRes/E%s.pdf'%head)
        plt.close()
    def plotR(self,N=16,v=80,U=1000,L=np.arange(100)*32+16,A=1,timeL=np.arange(-50,50,0.01),head=''):
        RAf,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        RBe,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        plt.close()
        plt.figure(figsize=[4,4])
        plt.plot(f,np.abs(RAf),'r',linewidth=0.5)
        plt.plot(f,np.abs(RBe),'b',linewidth=0.5)
        plt.xlabel('f/Hz')
        plt.ylabel('Amplitude')
        plt.xlim([0,20])
        plt.savefig('../hsrRes/R%s.pdf'%head)
        plt.close()
    def plotr(self,N=16,v=80,U=2300,L=np.arange(100)*32+16,A=16,timeL=np.arange(-50,50,0.01),head='',linewidth=0.5):
        rAf= self.calr(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        rBe= self.calr(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        self.plotWS(rAf+rBe,time0=timeL[0],head='syn',whiteL=[2.5,5,7.5,10,12.5,15,17.5,20,25],ylabel0='$D/D_0$',fMax=15,linewidth=linewidth)
        return 
        plt.close()
        plt.figure(figsize=[8,4])
        plt.subplot(2,1,1)
        plt.plot(timeL,rAf,'r',linewidth=0.5)
        plt.plot(timeL,rBe,'b',linewidth=0.5)
        plt.xlabel('t/s')
        plt.ylabel('D/count')
        plt.xlim([-20,20])
        plt.subplot(2,1,2)
        data,TL,FL=SFFT(rAf+rBe,0.01,10,800)
        plt.pcolor(TL+timeL[0],FL,np.abs(data)/np.abs(data.std(axis=0,keepdims=True)))
        plt.xlabel('t/s')
        plt.ylabel('f/Hz')
        plt.xlim([-20,20])
        plt.ylim([0,10])
        plt.savefig('../hsrRes/rt%s.jpg'%head,dpi=500)
        plt.close()
    def plotER(self,N=16,v=80,U=2300,Lc=25,L=np.arange(30,100)*32+16,A=1,timeL=np.arange(-50,50,0.01)\
        ,f=np.arange(0,100,0.01),head=''):
        E = self.calE(N=N,L=Lc,v=v,f=f)
        RAf,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        RBe,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        ERAf = np.conj(E)*RAf
        ERBe = np.conj(E)*RBe
        plt.close()
        plt.figure(figsize=[4,4])
        plt.plot(f[10:],np.abs(ERAf[10:]),'r',linewidth=0.5)
        plt.plot(f[10:],np.abs(ERBe[10:]),'b',linewidth=0.5)
        plt.xlabel('f/Hz')
        plt.ylabel('Amplitude')
        plt.xlim([0,20])
        plt.savefig('../hsrRes/ER%s.pdf'%head)
        plt.close()
    def plotERR(self,N=16,v=80,U=[3000],Lc=25,L=np.arange(30,100)*32+16,A=1,timeL=np.arange(-50,50,0.01)\
        ,f=np.arange(0,100,0.01),head='',t3='',time=0,t3_='',time1=0):
        E = self.calE(N=N,L=Lc,v=v,f=f)
        RAf,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        RBe,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        RAf0,f0= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL,dd=[0])
        RBe0,f0= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL,dd=[0])
        ERAf = np.conj(E)*RAf
        ERBe = np.conj(E)*RBe
        t3Be=t3.slice(time-20,time-10)
        t3At=t3.slice(time-4,time+4)
        t3Af=t3.slice(time+10,time+20)
        fL = np.arange(0,25,0.01)
        #beS,beF=t3Be.getSpec(comp=0,isNorm=False)
        #atS,atF=t3At.getSpec(comp=0,isNorm=False)
        #afS,afF=t3Af.getSpec(comp=0,isNorm=False)
        beS,beF=self.adjustSpecV2(t3Be,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS,atF=self.adjustSpecV2(t3At,3.2,f0=3.2,comp=0)
        afS,afF=self.adjustSpecV2(t3Af,3.2,f0=3.2,comp=0)
        FBe,SBe= self.FindFF(beS,beF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf,SAf= self.FindFF(afS,afF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        t3Be_=t3_.slice(time1-20,time1-10)
        t3At_=t3_.slice(time1-4,time1+4)
        t3Af_=t3_.slice(time1+10,time1+20)
        fVBe = self.findFF(t3Be)
        fVAt = self.findFF(t3At)
        fVAf = self.findFF(t3Af)
        fVBe_ = self.findFF(t3Be_)
        fVAt_ = self.findFF(t3At_)
        fVAf_ = self.findFF(t3Af_)
        #beS_,beF_=t3Be_.getSpec(comp=0,isNorm=False)
        #atS_,atF_=t3At_.getSpec(comp=0,isNorm=False)
        #afS_,afF_=t3Af_.getSpec(comp=0,isNorm=False)
        beS_,beF_=self.adjustSpecV2(t3Be_,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS_,atF_=self.adjustSpecV2(t3At_,3.2,f0=3.2,comp=0)
        afS_,afF_=self.adjustSpecV2(t3Af_,3.2,f0=3.2,comp=0)
        FBe_,SBe_= self.FindFF(beS_,beF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf_,SAf_= self.FindFF(afS_,afF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        print(FBe,FAf,80*fVBe/3.2,80*fVAf/3.2,(FBe+FAf)/(FBe-FAf)*80*(fVBe/2+fVAf/2)/3.2)
        print(FBe_,FAf_,80*fVBe_/3.2,80*fVAf_/3.2,(FBe_+FAf_)/(FBe_-FAf_)*80*(fVBe_/2+fVAf_/2)/3.2)
        plt.close()
        figureSet.init()
        plt.figure(figsize=[4.5,4])
        plt.subplot(6,1,1)
        plt.plot(f0,np.abs(RAf0),'r',linewidth=0.5)
        plt.plot(f0,np.abs(RBe0),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(a)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(6,1,2)
        plt.plot(f,np.abs(RAf),'r',linewidth=0.5)
        plt.plot(f,np.abs(RBe),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(b)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(6,1,3)
        plt.plot(f[10:],np.abs(ERAf[10:]),'r',linewidth=0.5)
        plt.plot(f[10:],np.abs(ERBe[10:]),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylim([-0.1,0.5])
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(c)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(6,1,4)
        self.testDenseFFT()
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(d)',[0.01,0.98],c='k')
        plt.xticks(a,['' for tmp in a])
        plt.subplot(6,1,5)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF,np.abs(afS)/np.abs(afS[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF,np.abs(beS)/np.abs(beS[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF,np.abs(atS)/np.abs(atS[:N]).max(),'k',linewidth=0.3)
        plt.arrow(FAf,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf,FAf],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe,FBe],[0,1.25],'--b',linewidth=0.3)
        plt.ylim([0,1.25])
        #plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(e)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(6,1,6)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF_,np.abs(afS_)/np.abs(afS_[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF_,np.abs(beS_)/np.abs(beS_[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF_,np.abs(atS_)/np.abs(atS_[:N]).max(),'k',linewidth=0.3)
        #plt.fill_between(afF_[np.abs(afF_-FAf_)<0.15], 0.2, (np.abs(afS_)/np.abs(afS_[:N]).max())[np.abs(afF_-FAf_)<0.15], facecolor='r', alpha=0.3)
        #plt.fill_between(beF_[np.abs(beF_-FBe_)<0.15], 0.2, (np.abs(beS_)/np.abs(beS_[:N]).max())[np.abs(beF_-FBe_)<0.15], facecolor='b', alpha=0.3)
        plt.arrow(FAf_,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe_,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf_,FAf_],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe_,FBe_],[0,1.25],'--b',linewidth=0.3)
        plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        plt.ylim([0,1.25])
        figureSet.setABC('(f)',[0.01,0.98],c='k')
        plt.savefig('../hsrRes/ERR%s.eps'%head,dpi=300)
        plt.close()
    
    def plotERR5(self,N=16,v=80,U=[3000],Lc=25,L=np.arange(30,100)*32+16,A=1,timeL=np.arange(-50,50,0.001)\
        ,f=np.arange(0,1000,0.01),head='',t3='',time=0,t3_='',time1=0):
        RAf,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        RBe,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        RAf0,f0= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL,dd=[0])
        RBe0,f0= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL,dd=[0])
        E = self.calE(N=N,L=Lc,v=v,f=f)
        ERAf = np.conj(E)*RAf
        ERBe = np.conj(E)*RBe
        t3Be=t3.slice(time-20,time-10)
        t3At=t3.slice(time-4,time+4)
        t3Af=t3.slice(time+10,time+20)
        fL = np.arange(0,25,0.01)
        #beS,beF=t3Be.getSpec(comp=0,isNorm=False)
        #atS,atF=t3At.getSpec(comp=0,isNorm=False)
        #afS,afF=t3Af.getSpec(comp=0,isNorm=False)
        beS,beF=self.adjustSpecV2(t3Be,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS,atF=self.adjustSpecV2(t3At,3.2,f0=3.2,comp=0)
        afS,afF=self.adjustSpecV2(t3Af,3.2,f0=3.2,comp=0)
        FBe,SBe= self.FindFF(beS,beF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf,SAf= self.FindFF(afS,afF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        t3Be_=t3_.slice(time1-20,time1-10)
        t3At_=t3_.slice(time1-4,time1+4)
        t3Af_=t3_.slice(time1+10,time1+20)
        fVBe = self.findFF(t3Be)
        fVAt = self.findFF(t3At)
        fVAf = self.findFF(t3Af)
        fVBe_ = self.findFF(t3Be_)
        fVAt_ = self.findFF(t3At_)
        fVAf_ = self.findFF(t3Af_)
        #beS_,beF_=t3Be_.getSpec(comp=0,isNorm=False)
        #atS_,atF_=t3At_.getSpec(comp=0,isNorm=False)
        #afS_,afF_=t3Af_.getSpec(comp=0,isNorm=False)
        beS_,beF_=self.adjustSpecV2(t3Be_,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS_,atF_=self.adjustSpecV2(t3At_,3.2,f0=3.2,comp=0)
        afS_,afF_=self.adjustSpecV2(t3Af_,3.2,f0=3.2,comp=0)
        FBe_,SBe_= self.FindFF(beS_,beF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf_,SAf_= self.FindFF(afS_,afF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        print(FBe,FAf,80*fVBe/3.2,80*fVAf/3.2,(FBe+FAf)/(FBe-FAf)*80*(fVBe/2+fVAf/2)/3.2)
        print(FBe_,FAf_,80*fVBe_/3.2,80*fVAf_/3.2,(FBe_+FAf_)/(FBe_-FAf_)*80*(fVBe_/2+fVAf_/2)/3.2)
        plt.close()
        figureSet.init()
        plt.figure(figsize=[4.5,4])
        plt.subplot(5,1,1)
        plt.plot(f0,np.abs(RAf0),'r',linewidth=0.3)
        plt.plot(f0,np.abs(RBe0),'b',linewidth=0.3)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(a)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,2)
        plt.plot(f,np.abs(RAf),'r',linewidth=0.3)
        plt.plot(f,np.abs(RBe),'b',linewidth=0.3)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(b)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,3)
        plt.plot(f[10:],np.abs(ERAf[10:]),'r',linewidth=0.3)
        plt.plot(f[10:],np.abs(ERBe[10:]),'b',linewidth=0.3)
        #plt.xlabel('f/Hz')
        plt.ylim([-0.1,0.5])
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(c)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,4)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF,np.abs(afS)/np.abs(afS[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF,np.abs(beS)/np.abs(beS[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF,np.abs(atS)/np.abs(atS[:N]).max(),'k',linewidth=0.3)
        plt.arrow(FAf,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf,FAf],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe,FBe],[0,1.25],'--b',linewidth=0.3)
        plt.ylim([0,1.25])
        #plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(d)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,5)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF_,np.abs(afS_)/np.abs(afS_[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF_,np.abs(beS_)/np.abs(beS_[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF_,np.abs(atS_)/np.abs(atS_[:N]).max(),'k',linewidth=0.3)
        #plt.fill_between(afF_[np.abs(afF_-FAf_)<0.15], 0.2, (np.abs(afS_)/np.abs(afS_[:N]).max())[np.abs(afF_-FAf_)<0.15], facecolor='r', alpha=0.3)
        #plt.fill_between(beF_[np.abs(beF_-FBe_)<0.15], 0.2, (np.abs(beS_)/np.abs(beS_[:N]).max())[np.abs(beF_-FBe_)<0.15], facecolor='b', alpha=0.3)
        plt.arrow(FAf_,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe_,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf_,FAf_],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe_,FBe_],[0,1.25],'--b',linewidth=0.3)
        plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        plt.ylim([0,1.25])
        figureSet.setABC('(e)',[0.01,0.98],c='k')
        plt.savefig('../hsrRes/ERR%s.eps'%head,dpi=300)
        plt.close()
    def plotERR5_0_300(self,N=16,v=80,U=[3000],Lc=25,L=np.arange(30,100)*32+16,A=1,timeL=np.arange(-50,50,0.001)\
        ,f=np.arange(0,1000,0.01),head='',t3='',time=0,t3_='',time1=0):
        RAf,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        RBe,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        RAf0,f0= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL,dd=[0])
        RBe0,f0= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL,dd=[0])
        E = self.calE(N=N,L=Lc,v=v,f=f)
        ERAf = np.conj(E)*RAf
        ERBe = np.conj(E)*RBe
        t3Be=t3.slice(time-20,time-10)
        t3At=t3.slice(time-4,time+4)
        t3Af=t3.slice(time+10,time+20)
        fL = np.arange(0,25,0.01)
        #beS,beF=t3Be.getSpec(comp=0,isNorm=False)
        #atS,atF=t3At.getSpec(comp=0,isNorm=False)
        #afS,afF=t3Af.getSpec(comp=0,isNorm=False)
        beS,beF=self.adjustSpecV2(t3Be,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS,atF=self.adjustSpecV2(t3At,3.2,f0=3.2,comp=0)
        afS,afF=self.adjustSpecV2(t3Af,3.2,f0=3.2,comp=0)
        FBe,SBe= self.FindFF(beS,beF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf,SAf= self.FindFF(afS,afF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        t3Be_=t3_.slice(time1-20,time1-10)
        t3At_=t3_.slice(time1-4,time1+4)
        t3Af_=t3_.slice(time1+10,time1+20)
        fVBe = self.findFF(t3Be)
        fVAt = self.findFF(t3At)
        fVAf = self.findFF(t3Af)
        fVBe_ = self.findFF(t3Be_)
        fVAt_ = self.findFF(t3At_)
        fVAf_ = self.findFF(t3Af_)
        #beS_,beF_=t3Be_.getSpec(comp=0,isNorm=False)
        #atS_,atF_=t3At_.getSpec(comp=0,isNorm=False)
        #afS_,afF_=t3Af_.getSpec(comp=0,isNorm=False)
        beS_,beF_=self.adjustSpecV2(t3Be_,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS_,atF_=self.adjustSpecV2(t3At_,3.2,f0=3.2,comp=0)
        afS_,afF_=self.adjustSpecV2(t3Af_,3.2,f0=3.2,comp=0)
        FBe_,SBe_= self.FindFF(beS_,beF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf_,SAf_= self.FindFF(afS_,afF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        print(FBe,FAf,80*fVBe/3.2,80*fVAf/3.2,(FBe+FAf)/(FBe-FAf)*80*(fVBe/2+fVAf/2)/3.2)
        print(FBe_,FAf_,80*fVBe_/3.2,80*fVAf_/3.2,(FBe_+FAf_)/(FBe_-FAf_)*80*(fVBe_/2+fVAf_/2)/3.2)
        plt.close()
        figureSet.init()
        plt.figure(figsize=[4.5,4])
        plt.subplot(5,1,1)
        plt.plot(f0,np.abs(RAf0),'r',linewidth=0.3)
        plt.plot(f0,np.abs(RBe0),'b',linewidth=0.3)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(a)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,2)
        plt.plot(f,np.abs(RAf),'r',linewidth=0.3)
        plt.plot(f,np.abs(RBe),'b',linewidth=0.3)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(b)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,3)
        plt.plot(f[10:],np.abs(ERAf[10:]),'r',linewidth=0.3)
        plt.plot(f[10:],np.abs(ERBe[10:]),'b',linewidth=0.3)
        #plt.xlabel('f/Hz')
        plt.ylim([-0.1,0.5])
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(c)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,4)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF,np.abs(afS)/np.abs(afS[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF,np.abs(beS)/np.abs(beS[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF,np.abs(atS)/np.abs(atS[:N]).max(),'k',linewidth=0.3)
        plt.arrow(FAf,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf,FAf],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe,FBe],[0,1.25],'--b',linewidth=0.3)
        plt.ylim([0,1.25])
        #plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(d)',[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(5,1,5)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF_,np.abs(afS_)/np.abs(afS_[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF_,np.abs(beS_)/np.abs(beS_[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF_,np.abs(atS_)/np.abs(atS_[:N]).max(),'k',linewidth=0.3)
        #plt.fill_between(afF_[np.abs(afF_-FAf_)<0.15], 0.2, (np.abs(afS_)/np.abs(afS_[:N]).max())[np.abs(afF_-FAf_)<0.15], facecolor='r', alpha=0.3)
        #plt.fill_between(beF_[np.abs(beF_-FBe_)<0.15], 0.2, (np.abs(beS_)/np.abs(beS_[:N]).max())[np.abs(beF_-FBe_)<0.15], facecolor='b', alpha=0.3)
        plt.arrow(FAf_,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe_,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf_,FAf_],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe_,FBe_],[0,1.25],'--b',linewidth=0.3)
        plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        plt.ylim([0,1.25])
        figureSet.setABC('(e)',[0.01,0.98],c='k')
        plt.savefig('../hsrRes/ERR%s.eps'%head,dpi=300)
        plt.close()
    def plotERR6(self,N=16,v=80,U=[3000],UPS=[3000,1000],Lc=25,L=np.arange(30,50)*32+16,A=1,timeL=np.arange(-50,50,0.001)\
        ,f=np.arange(0,1000,0.01),head='',t3='',time=0,t3_='',time1=0,D=200):
        RAf,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        RBe,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        E = self.calE(N=N,L=Lc,v=v,f=f)
        RAf0,f0= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL,dd=[0])
        RBe0,f0= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL,dd=[0])
        ERAf = np.conj(E)*RAf
        ERBe = np.conj(E)*RBe
        RAfPS,f0= self.calR(N=N,v=v,U=UPS,L=L,A=A,timeL=timeL,randP=0.1,D=D)
        RBePS,f0= self.calR(N=N,v=v,U=UPS,L=-L,A=A,timeL=timeL,randP=0.1,D=D)
        ERAfPS = np.conj(E)*RAfPS
        ERBePS = np.conj(E)*RBePS
        t3Be=t3.slice(time-20,time-10)
        t3At=t3.slice(time-4,time+4)
        t3Af=t3.slice(time+10,time+20)
        fL = np.arange(0,25,0.01)
        #beS,beF=t3Be.getSpec(comp=0,isNorm=False)
        #atS,atF=t3At.getSpec(comp=0,isNorm=False)
        #afS,afF=t3Af.getSpec(comp=0,isNorm=False)
        beS,beF=self.adjustSpecV2(t3Be,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS,atF=self.adjustSpecV2(t3At,3.2,f0=3.2,comp=0)
        afS,afF=self.adjustSpecV2(t3Af,3.2,f0=3.2,comp=0)
        FBe,SBe= self.FindFF(beS,beF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf,SAf= self.FindFF(afS,afF,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        t3Be_=t3_.slice(time1-20,time1-10)
        t3At_=t3_.slice(time1-4,time1+4)
        t3Af_=t3_.slice(time1+10,time1+20)
        fVBe = self.findFF(t3Be)
        fVAt = self.findFF(t3At)
        fVAf = self.findFF(t3Af)
        fVBe_ = self.findFF(t3Be_)
        fVAt_ = self.findFF(t3At_)
        fVAf_ = self.findFF(t3Af_)
        #beS_,beF_=t3Be_.getSpec(comp=0,isNorm=False)
        #atS_,atF_=t3At_.getSpec(comp=0,isNorm=False)
        #afS_,afF_=t3Af_.getSpec(comp=0,isNorm=False)
        beS_,beF_=self.adjustSpecV2(t3Be_,3.2,f0=3.2,comp=0)#t3Be.getSpec(comp=0,isNorm=False)
        atS_,atF_=self.adjustSpecV2(t3At_,3.2,f0=3.2,comp=0)
        afS_,afF_=self.adjustSpecV2(t3Af_,3.2,f0=3.2,comp=0)
        FBe_,SBe_= self.FindFF(beS_,beF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        FAf_,SAf_= self.FindFF(afS_,afF_,minF=4.6,maxF=5.4,avoidF=3.2,fmax=6,reS=True,mul=1)
        print(FBe,FAf,80*fVBe/3.2,80*fVAf/3.2,(FBe+FAf)/(FBe-FAf)*80*(fVBe/2+fVAf/2)/3.2)
        print(FBe_,FAf_,80*fVBe_/3.2,80*fVAf_/3.2,(FBe_+FAf_)/(FBe_-FAf_)*80*(fVBe_/2+fVAf_/2)/3.2)
        plt.close()
        figureSet.init()
        plt.figure(figsize=[4.5,4])
        figCount  = 6
        figIndex  = 1
        strL = 'aabcdefg'
        plt.subplot(figCount,1,figIndex)
        plt.plot(f0,np.abs(RAf0),'r',linewidth=0.5)
        plt.plot(f0,np.abs(RBe0),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(%s)'%strL[figIndex],[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        figIndex+=1
        plt.subplot(figCount,1,figIndex)
        plt.plot(f,np.abs(RAf),'r',linewidth=0.5)
        plt.plot(f,np.abs(RBe),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(%s)'%strL[figIndex],[0.01,0.98],c='k')
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        figIndex+=1
        plt.subplot(figCount,1,figIndex)
        plt.plot(f[10:],np.abs(ERAf[10:]),'r',linewidth=0.5)
        plt.plot(f[10:],np.abs(ERBe[10:]),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylim([-0.1,0.45])
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(%s)'%strL[figIndex],[0.01,0.98],c='k')
        figIndex+=1
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(figCount,1,figIndex)
        plt.plot(f[10:],np.abs(RAfPS[10:]),'r',linewidth=0.5)
        plt.plot(f[10:],np.abs(RBePS[10:]),'b',linewidth=0.5)
        #plt.xlabel('f/Hz')
        plt.ylim([-0.1,0.5])
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(%s)'%strL[figIndex],[0.01,0.98],c='k')
        figIndex+=1
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(figCount,1,figIndex)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF,np.abs(afS)/np.abs(afS[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF,np.abs(beS)/np.abs(beS[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF,np.abs(atS)/np.abs(atS[:N]).max(),'k',linewidth=0.3)
        plt.arrow(FAf,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf,FAf],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe,FBe],[0,1.25],'--b',linewidth=0.3)
        plt.ylim([0,1.25])
        #plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        figureSet.setABC('(%s)'%strL[figIndex],[0.01,0.98],c='k')
        figIndex+=1
        a=plt.xticks()[0]
        plt.xticks(a,['' for tmp in a])
        plt.subplot(figCount,1,figIndex)
        N = int(20*t3.delta*len(afF))
        N=-1
        plt.plot(afF_,np.abs(afS_)/np.abs(afS_[:N]).max(),'r',linewidth=0.3)
        plt.plot(beF_,np.abs(beS_)/np.abs(beS_[:N]).max(),'b',linewidth=0.3)
        #plt.plot(atF_,np.abs(atS_)/np.abs(atS_[:N]).max(),'k',linewidth=0.3)
        #plt.fill_between(afF_[np.abs(afF_-FAf_)<0.15], 0.2, (np.abs(afS_)/np.abs(afS_[:N]).max())[np.abs(afF_-FAf_)<0.15], facecolor='r', alpha=0.3)
        #plt.fill_between(beF_[np.abs(beF_-FBe_)<0.15], 0.2, (np.abs(beS_)/np.abs(beS_[:N]).max())[np.abs(beF_-FBe_)<0.15], facecolor='b', alpha=0.3)
        plt.arrow(FAf_,1.25,0,-0.10,color='r',linewidth=0.5,width=0.01)
        plt.arrow(FBe_,1.25,0,-0.10,color='b',linewidth=0.5,width=0.01)
        #plt.plot([FAf_,FAf_],[0,1.25],'--r',linewidth=0.3)
        #plt.plot([FBe_,FBe_],[0,1.25],'--b',linewidth=0.3)
        plt.xlabel('$f$/Hz')
        plt.ylabel('$A$')
        plt.xlim([0,20])
        plt.ylim([0,1.25])
        figureSet.setABC('(%s)'%strL[figIndex],[0.01,0.98],c='k')
        plt.savefig('../hsrRes/ERR%s.eps'%head,dpi=300)
        plt.close()
    def plotERL(self,N=16,v=80,UL=[500,1000,3000],Lc=25,\
        L=np.arange(30,100)*32+16,A=1,timeL=np.arange(-50,50,0.01)\
        ,f=np.arange(0,100,0.01),head=''):
        E = self.calE(N=N,L=Lc,v=v,f=f)
        ERAf = E*0
        ERBe = E*0
        for U in UL:
            RAft,f= self.calR(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
            RBet,f= self.calR(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
            ERAf += np.conj(E)*RAft
            ERBe += np.conj(E)*RBet
        plt.close()
        plt.figure(figsize=[4,4])
        plt.plot(f[10:],np.abs(ERAf[10:]),'r',linewidth=0.5)
        plt.plot(f[10:],np.abs(ERBe[10:]),'b',linewidth=0.5)
        plt.xlabel('f/Hz')
        plt.ylabel('Amplitude')
        plt.xlim([0,20])
        plt.savefig('../hsrRes/ERL%s.pdf'%head)
        plt.close()
    def plotWS(self,data,time0=-40,head='test',delta=0.01,fMin=0,fMax=10,xlim=[-40,40],whiteL=[],ylabel0='D/count',isSpec=False,linewidth=0.5):
        figureSet.init()
        fMax+=0.5
        #head = '1.567142294544028521e+09'
        plt.close()
        fig=plt.figure(figsize=[4.8,4])
        fig.tight_layout()
        '''
        if isSpec:
            specs=gridspec.GridSpec(3, 1,height_ratios=[1,2,1])
        else:
        '''
        specs=gridspec.GridSpec(2, 1,height_ratios=[1,3])
        ax0 = fig.add_subplot(specs[0])
        #plt.subplot(2,1,1)
        tL = np.arange(len(data))*delta+time0
        ax0.plot(tL[np.abs(tL)<=4],data[np.abs(tL)<=4],'k',linewidth=linewidth)
        ax0.plot(tL[tL<-4],data[tL<-4],'b',linewidth=linewidth)
        ax0.plot(tL[tL>4],data[tL>4],'r',linewidth=linewidth)
        #plt.plot(timeL,rBe,'b',linewidth=0.5)
        #plt.xlabel('t/s')
        plt.ylabel(ylabel0)
        plt.xlim(xlim)
        figureSet.setABC('(a)',[0.01,0.98])
        ax1 = fig.add_subplot(specs[1])
        N=int(8/delta)
        dataS,TL,FL=SFFT(data,delta,10,N)
        pc=ax1.pcolormesh(TL+time0,FL[:int(N*fMax*delta)],np.log(np.abs(dataS[:int(N*fMax*delta)])/np.std(dataS[:int(N*fMax*delta)]).max(axis=0,keepdims=True)+1e-3),cmap='hot',shading='gouraud',rasterized=True,vmin=-3)
        for white in whiteL:
            ax1.plot(TL+time0,(TL+time0)*0+white,'-.',color='cyan',linewidth=0.3)
        plt.xlabel('$t$/s')
        plt.ylabel('$f$/Hz')
        plt.xlim(xlim)
        plt.ylim([fMin,fMax-0.5])
        figureSet.setABC('(b)',[0.01,0.98],c='w')
        ax=plt.gca()
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("bottom", size="7%", pad="25%")
        cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
        cbar.set_label('log($A$)')
        if isSpec:
            ax_divider0 = make_axes_locatable(ax0)
            specAxTmp = ax_divider0.append_axes("right", size="30%", pad="5%")
            specAxTmp.axis('off')
            specAx = ax_divider.append_axes("right", size="30%", pad="5%")
            spec=np.abs(np.fft.fft(data))
            N =len(data)
            fL = np.arange(N)/N*1/delta
            specAx.plot(spec,fL,'k',linewidth=0.5)
            plt.xlabel('$A$')
            plt.ylim([fMin,fMax-0.5])
            a=plt.yticks()[0]
            plt.yticks(a,['' for tmp in a])
            figureSet.setABC('(c)',[0.85,0.98],c='k')
        #plt.tight_layout()
        plt.savefig('../hsrRes/rt%s%d.eps'%(head,time0),dpi=500)
    def testDenseFFT(self,f0=5+1/16,N=800,delta=0.01):
        timeL=(np.arange(N)*delta).reshape([1,-1])
        data =  np.sin(timeL*f0*np.pi*2)
        fL = self.fL0.reshape([-1,1])
        fL0 = (np.arange(N)*1/delta/N).reshape([-1,1])
        M  = (data*np.exp(-1j*fL*timeL*np.pi*2)).sum(axis=1)
        M0  = (data*np.exp(-1j*fL0*timeL*np.pi*2)).sum(axis=1)
        #plt.close()
        #plt.figure(figsize=[3,3])
        plt.plot(fL0[:,0],np.abs(M0)/np.abs(M).max(),'or',markersize=0.5)
        plt.plot(fL[:,0],np.abs(M)/np.abs(M).max(),'k',linewidth=0.3)
        #plt.plot([f0,f0],[0,np.abs(M).max()*1.5],'b',linewidth=0.3)
        plt.xlim([0,10])
        plt.ylim([-0.1,1.25])
        plt.arrow(f0,1.25,0,-0.1,color='b',linewidth=0.5,width=0.01)
        #plt.savefig('../hsrRes/ERRDense.eps',dpi=300)

def handleDay(l):
    stations,day,workDir,compL,rotate,bSecL,eSecL,T3L=l
    h = hsr()
    h.handleDay(stations,day,workDir,compL,rotate,bSecL,eSecL,T3L)
def SFFT(data,dt,dN,dn):
    N0 = len(data)-dn+1
    N1 = int(N0/dN)
    print(N1)
    data1 = np.zeros([dn,N1],np.complex128)
    tL= (np.arange(N1)*dN+dn/2)*dt
    fL=(np.arange(dn)/dn*1/dt)
    for i in range(N1):
        #print([i*dN,(i*dN+dn)])
        data1[:,i]=np.fft.fft(data[i*dN:(i*dN+dn)])
    return data1,tL,fL



def toStr(num):
    tmp=''
    for n in num:
        tmp+=str(np.real(n))+' '
    return tmp
def meanE(num):
    return (num**2).mean()**0.5

def plotWSpec(t3):
    data = t3.Data()
    data/=data.max()
    bTime = t3.bTime
    timeL = bTime + np.arange(len(data))*t3.delta
    specFL= [t3.getSpec(i) for i in range(3)]
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(timeL,data[:,i],linewidth=0.5)


def plotStaRail(stations,mt,basemap):
    hsr = mt.readFault('data/hsr.shape')
    staLa,staLo=stations.loc()
    dLa = staLa.max()-staLa.min()
    dLo = staLo.max()-staLo.min()
    laL = [staLa.min()-dLa*2,staLa.max()+dLa*2]
    loL = [staLo.min()-dLo*2,staLo.max()+dLo*2]
    laL0 = [staLa.min()-dLa*90,staLa.max()+dLa*90]
    loL0 = [staLo.min()-dLo*90,staLo.max()+dLo*90]
    #fig.tight_layout()
    figureSet.init()
    plt.close()
    fig=plt.figure(figsize=[4,8])
    plt.subplot(2,2,1)
    m0 = basemap.Basemap(llcrnrlat=laL0[0],urcrnrlat=laL0[1],llcrnrlon=loL0[0],\
            urcrnrlon=loL0[1])
    pc=mt.plotTopo(m0,laL0+loL0,isColorbar=False,vmin=0,vmax=1500)
    squareLa = [laL[0],laL[0],laL[1],laL[1],laL[0]]
    squareLo = [loL[0],loL[1],loL[1],loL[0],loL[0]]
    squareX,squareY=m0(squareLo,squareLa)
    plt.plot(squareX,squareY,'b',linewidth=0.5)
    for fault in hsr:
        if fault.inR(laL0+loL0):
            f=fault.plot(m0,linewidth=2,cmd='-k')

    plt.xlim()
    plt.ylim()
    R=laL0+loL0
    dD=max(int((R[1]-R[0])*3)/8,int((R[3]-R[2])*3)/8)
    #m.arcgisimage(service='World_Physical_Map', xpixels = 1500, verbose= False)
    parallels = np.arange(int(R[0]),int(R[1]+1),dD)
    m0.drawparallels(parallels,labels=[1,0,0,1])
    meridians = np.arange(int(R[2]),int(R[3]+1),dD)
    m0.drawmeridians(meridians,labels=[1,0,1,0])
    figureSet.setABC('(a)',[0.01,0.98],c='k',m=m0)
    plt.subplot(2,2,3)
    figureSet.setColorbar(pc,label='$Topography$(m)',pos='ZGKX')
    plt.subplot(2,2,2)
    m = basemap.Basemap(llcrnrlat=laL[0],urcrnrlat=laL[1],llcrnrlon=loL[0],\
            urcrnrlon=loL[1])
    staX,staY=m(np.array(staLo),np.array(staLa))
    R=laL+loL
    st=m.plot(staX,staY,'b^',markersize=2)
    for fault in hsr:
        if fault.inR(laL+loL):
            f=fault.plot(m,linewidth=2,cmd='-k')

    figureSet.setABC('(b)',[0.01,0.98],c='k',m=m)
    dD=max(int((R[1]-R[0])*150)/400,int((R[3]-R[2])*150)/400)
    #m.arcgisimage(service='World_Physical_Map', xpixels = 1500, verbose= False)
    parallels = np.arange(int(R[0]),int(R[1]+1),dD)
    m.drawparallels(parallels,labels=[1,0,0,1])
    meridians = np.arange(int(R[2]),int(R[3]+1),dD)
    m.drawmeridians(meridians,labels=[1,0,1,0])
    fig.tight_layout()
    plt.savefig('./hsr_sta.eps',dpi=600)

def plotStaRail2(mt,basemap):
    hsr = mt.readFault('data/China_HSR_2016_lines.shape',maxD=2)
    #staLa,staLo=stations.loc()
    #dLa = staLa.max()-staLa.min()
    #dLo = staLo.max()-staLo.min()
    laL0 = [20,43]
    loL0 = [110,125]
    #laL0 = [staLa.min()-dLa*90,staLa.max()+dLa*90]
    #loL0 = [staLo.min()-dLo*90,staLo.max()+dLo*90]

    plt.close()
    fig=plt.figure(figsize=[5,4])
    #fig.tight_layout()
    figureSet.init()
    #plt.subplot(1,2,1)
    m0 = basemap.Basemap(llcrnrlat=laL0[0],urcrnrlat=laL0[1],llcrnrlon=loL0[0],\
            urcrnrlon=loL0[1])
    #pc=mt.plotTopo(m0,laL0+loL0,isColorbar=False,vmin=-10000,vmax=10000,topo='/media/jiangyr/MSSD/ETOPO1_Ice_g_gmt4_new3.grd',cptfile='etopo1.cpt')
    #m0.drawcoastlines(linewidth=0.1, linestyle='solid', color='k', antialiased=0.1, ax=None, zorder=0)
    #m0.etopo()
    #squareLa = [laL[0],laL[0],laL[1],laL[1],laL[0]]
    #squareLo = [loL[0],loL[1],loL[1],loL[0],loL[0]]
    #squareX,squareY=m0(squareLo,squareLa)
    #plt.plot(squareX,squareY,'b',linewidth=0.5)
    for fault in hsr[:]:
        if fault.inR(laL0+loL0):
            fault.plot(m0,linewidth=0.5,cmd='-r')
    plt.xlim()
    plt.ylim()
    R=laL0+loL0
    dD=5
    #m.arcgisimage(service='World_Physical_Map', xpixels = 1500, verbose= False)
    parallels = np.arange(0,90,dD)
    m0.drawparallels(parallels,labels=[1,0,0,1])
    meridians = np.arange(0,180,dD)
    m0.drawmeridians(meridians,labels=[1,0,1,0])
    figureSet.setABC('(a)',[0.01,0.98],c='k',m=m0)
    #figureSet.setColorbar(pc,label='$Topography$(m)',pos='ZGKX')
    '''
    plt.subplot(1,2,2)
    m = basemap.Basemap(llcrnrlat=laL[0],urcrnrlat=laL[1],llcrnrlon=loL[0],\
            urcrnrlon=loL[1])
    staX,staY=m(np.array(staLo),np.array(staLa))
    R=laL+loL
    st=m.plot(staX,staY,'b^',markersize=2)
    for fault in hsr:
        if fault.inR(laL+loL):
            f=fault.plot(m,linewidth=2,cmd='-k')

    figureSet.setABC('(b)',[0.01,0.98],c='k',m=m)
    dD=max(int((R[1]-R[0])*150)/400,int((R[3]-R[2])*150)/400)
    #m.arcgisimage(service='World_Physical_Map', xpixels = 1500, verbose= False)
    parallels = np.arange(int(R[0]),int(R[1]+1),dD)
    m.drawparallels(parallels,labels=[1,0,0,1])
    meridians = np.arange(int(R[2]),int(R[3]+1),dD)
    m.drawmeridians(meridians,labels=[1,0,1,0])
    '''
    fig.tight_layout()
    plt.savefig('../hsrRes/hsr_staEast.jpg',dpi=600)