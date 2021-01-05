from ..io import seism
import numpy as np
from obspy import UTCDateTime
from glob import glob
import os
from matplotlib import pyplot as plt
from multiprocessing import pool
import matplotlib
matplotlib.rcParams['font.family']='Simhei'
class hsr:
    def __init__(self,f0=3.2,fmax=50,fL0=np.arange(0,50,0.02)):
        self.f0=f0
        self.fmax= fmax
        self.fL0=fL0
    def findFF(self,T3,comp=2):
        spec,fL=T3.getSpec(comp=comp)
        return self.FindFF(spec,fL)
    def FindFF(self,spec,fL,minF=2.7,maxF=3.6,avoidF=-1,fmax=49,reS=False,mul=1):
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
            FL=np.arange(FF*mul,fmax,FF)
            indexL=((FL-fL[0])/df0).astype(np.int)
            s = np.abs(spec)[indexL].sum()
            if s>=S:
                S=s
                FF0=FF
        if  reS:
            return FF0,S
        return FF0*mul
    def adjustSpec_old(self,T3,f,f0=3.2,comp=2):
        spec,fL=T3.getSpec(isNorm=True,comp=comp)
        fL=fL/f*f0
        df0=(fL[1]-fL[0])
        indexL=((self.fL0-fL[0])/df0).astype(np.int)
        return spec[indexL]
    def adjustSpec(self,T3,f,f0=3.2,comp=2):
        data = T3.Data()
        data/=data.std()
        data=data[:,comp].reshape([1,-1])
        delta=T3.delta
        timeL=(np.arange(data.size)*delta).reshape([1,-1])
        fL = (self.fL0*f/f0).reshape([-1,1])
        M  = data*np.exp(-1j*fL*timeL*np.pi*2)
        return M.sum(axis=1)
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
            if np.abs(v0-v1)>0.1:
                continue
            t3=T3.slice(time-20,time+20)
            time = self.MeanTime(t3)
            if t3.bTime>0 and time>0:
                print('handle',time)
                t3At=T3.slice(time-5,time+5)
                t3Be=T3.slice(time-eSec,time-bSec)#-25--5
                t3Af=T3.slice(time+bSec,time+eSec)
                if t3At.bTime<0 or t3Be.bTime<0 or t3Af.bTime<0:
                    continue
                FFAt.append(self.findFF(t3At,comp=comp))
                specAt.append(self.adjustSpec(t3At,FFAt[-1],comp=comp))
                FFBe.append(self.findFF(t3Be,comp=comp))
                specBe.append(self.adjustSpec(t3Be,FFBe[-1],comp=comp))
                FFAf.append(self.findFF(t3Af,comp=comp))
                specAf.append(np.abs(self.adjustSpec(t3Af,FFAf[-1],comp=comp)))
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
    def handleDay(self,stations,day,workDir='../hsrRes/',compL=[2],rotate=0,bSecL=[5,10,15,20],eSecL=[15,20,25,30],T3L=[]):
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

    def loadStationsSpec(self,stations,workDir,comp=2,isAdd=False,minDT=-100,maxDT=100,dF=0.1,bandStr=''):
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
                fL0L[-1]= fL0L[-1][0:1]
                specAtL[-1]=self.add(fL0,specAtL[-1],FFAtL[-1],dTL[-1],\
                    minDT=minDT,maxDT=maxDT,minFF=3.2-dF,maxFF=3.2+dF,F0=FFAtL[-1],dF0=0.08)
                specBeL[-1]=self.add(fL0,specBeL[-1],FFBeL[-1],dTL[-1],\
                    minDT=minDT,maxDT=maxDT,minFF=3.2-dF,maxFF=3.2+dF,F0=FFAtL[-1],dF0=0.08)
                specAfL[-1]=self.add(fL0,specAfL[-1],FFAfL[-1],dTL[-1],\
                    minDT=minDT,maxDT=maxDT,minFF=3.2-dF,maxFF=3.2+dF,F0=FFAtL[-1],dF0=0.08)
        return fL0L,specAtL,specBeL,specAfL,FFAtL,FFBeL,FFAfL,dTL
    def add(self,fL0,spec,FF,dT,minDT=-100,maxDT=100,minFF=3.0,maxFF=3.3,F0=[],dF0=-1):
        res = fL0[0:1]*0
        count=0
        maxSum=0
        for i in range(len(spec)):
            if FF[i]<minFF or FF[i]>maxFF:
                continue
            if dT[i]<minDT or dT[i]>maxDT:
                continue
            if len(F0)>0 and np.abs(FF[i]-F0[i])>dF0:
                continue
            res += spec[i:i+1]
            maxSum = spec[i:i+1].max()
            count+=1
        print('add',count,'maxSum',maxSum)
        return res/count
    def showSpec(self,fL0L,specAtL,specBeL,specAfL,stations,head='',workDir='',maxF=20,v=3000,isPlot=True):
        n= np.arange(1,10)
        fBe = v/(v-80)*3.2*n*25/32
        fAf = v/(v+80)*3.2 *n*25/32
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
            filename = workDir+'/'+ staName+'.jpg'
            plt.close()
            FBe,SBe= self.FindFF(specBe[0],fL0L[0][0,:],minF=4.6,maxF=5.4,avoidF=3.2,fmax=9,reS=True,mul=1)
            FAf,SAf = self.FindFF(specAf[0],fL0L[0][0,:],minF=4.6,maxF=5.4,avoidF=3.2,fmax=9,reS=True,mul=1)
            fBeL = FBe*n
            fAfL = FAf*n
            V = (FBe+FAf)/(FBe-FAf)*80
            FBeL.append(FBe)
            FAfL.append(FAf)
            SBeL.append(SBe)
            SAfL.append(SAf)
            vL.append(V)
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
            for i in range(len(specAt)):
                hAt=plt.plot(fL0L[i][i,:],specAt[i],'k',linewidth=0.5)
                hBe=plt.plot(fL0L[i][i,:],specBe[i],'b',linewidth=0.5)
                hAf=plt.plot(fL0L[i][i,:],specAf[i],'r',linewidth=0.5)
            plt.legend((hAt,hBe,hAf),['at','before','after'])
            plt.xlim([0,maxF])
            #plt.ylim([-1,800])
            plt.title(staName+' V: %.2f'%V)
            fileDir = os.path.dirname(filename)
            if not os.path.exists(fileDir):
                os.makedirs(fileDir)
            plt.savefig(filename,dpi=300)
            plt.close()
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
    def plotFV(self,FBeLNM,FAfLNM,vLNM,FBeLSM,FAfLSM,vLSM,bSecL,eSecL,dTimeL,workDir,head='',marker='.'):
        for comp in range(3):
            FBeLNL = np.array(FBeLNM[comp])
            FAfLNL = np.array(FAfLNM[comp])
            vLNL = np.array(vLNM[comp])
            FBeLSL = np.array(FBeLSM[comp])
            FAfLSL = np.array(FAfLSM[comp])
            vLSL = np.array(vLSM[comp])
            midSecL = (eSecL+bSecL)/2
            plt.close()
            plt.figure(figsize=[3,4])
            plt.subplot(2,1,1)
            plt.ylim([4,6])
            plt.xlim([-5000,5000])
            for i in range(len(dTimeL)):
                dTime=dTimeL[i]
                plt.plot((midSecL-dTime)*80,FBeLNL[:,i],marker+'b')
                plt.plot((midSecL-dTime)*80,FAfLNL[:,i],marker+'r')
                plt.plot((-midSecL+dTime)*80,FBeLSL[:,i],marker+'b')
                plt.plot((-midSecL+dTime)*80,FAfLSL[:,i],marker+'r')
            #plt.xlim([-60,60])
            plt.ylabel('f/Hz')
            plt.subplot(2,1,2)
            plt.ylim([0,5000])
            for i in range(len(dTimeL)):
                dTime=dTimeL[i]
                plt.plot((midSecL-dTime)*80,vLNL[:,i],marker+'k')
                plt.plot((-midSecL+dTime)*80,vLSL[:,i],marker+'k')
            plt.xlim([-5000,5000])
            plt.xlabel('dis/m')
            plt.ylabel('v/(m/s)')
            if not os.path.exists(workDir):
                os.makedirs(workDir)
            plt.savefig(workDir+head+'%d.pdf'%comp)
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
            plt.hist2d(beNT+beST+afNT+afST,beNS+beSS+afNS+afSS,range=[[-50,50],[0,1]],cmap='bwr')
            plt.ylabel(compL[comp]+'/All')
            if comp==2:
                plt.xlabel('t/s')
        if not os.path.exists(workDir):
            os.makedirs(workDir)
        plt.savefig(workDir+head+'.pdf')
    def calE(self,N=16,L=25,v=80,f=np.arange(0,100,0.01)):
        E = (f*0).astype(np.complex128)
        for i in range(N):
            E+=np.exp(-1j*f*np.pi*2*i*L/v)
        return E
    def calr(self,N=16,v=80,U=3000,L=np.arange(100)*32+16,A=1,timeL=np.arange(-40,40,0.01)):
        r = timeL*0
        for l in L:
            t0 = l/v
            dt = +np.abs(l)/U
            t=t0+dt
            index = np.abs(timeL-t).argmin()
            r[index]+= A/np.abs(l)
            #r[index+1]-= A/np.abs(l)
        return r
    def calR(self,N=16,v=80,U=3000,L=np.arange(100)*32+16,A=1,timeL=np.arange(-50,50,0.01)):
        r = self.calr(N=N,v=v,U=U,L=L,A=1,timeL=timeL)
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
    def plotr(self,N=16,v=80,U=2300,L=np.arange(100)*32+16,A=1,timeL=np.arange(-50,50,0.01),head=''):
        rAf= self.calr(N=N,v=v,U=U,L=L,A=A,timeL=timeL)
        rBe= self.calr(N=N,v=v,U=U,L=-L,A=A,timeL=timeL)
        plt.close()
        plt.figure(figsize=[8,4])
        plt.subplot(2,1,1)
        plt.plot(timeL,rAf,'r',linewidth=0.5)
        plt.plot(timeL,rBe,'b',linewidth=0.5)
        plt.xlabel('t/s')
        plt.ylabel('Disp')
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
        

def handleDay(l):
    stations,day,workDir,compL,rotate,bSecL,eSecL,T3L=l
    h = hsr()
    h.handleDay(stations,day,workDir,compL,rotate,bSecL,eSecL,T3L)
def SFFT(data,dt,dN,dn):
    N0 = len(data)-dn+1
    N1 = int(N0/dN)
    print(N1)
    data1 = np.zeros([dn,N1],np.complex128)
    tL= (np.arange(N1)*dN-dN/2)*dt
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


        

        


