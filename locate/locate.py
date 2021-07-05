from ..io.seism import QuakeCC,RecordCC
from ..io.tool import quickTaupModel
import numpy as np
from ..mathTool.distaz import DistAz
import os
from numba import jit

class locator:
    def __init__(self,staInfos,modelFile=os.path.dirname(__file__)+'/../data/iaspTaupMat',maxDT=30):
        self.staInfos=staInfos
        self.timeM=quickTaupModel(modelFile)
        self.maxErr=-1
        self.maxDT=30
    def locate(self, quake,r=1,e=0.1,maxDT=-1,isDel=False,maxErr=-1):
        mulL=[40]*3+[20]*2+[10]*2+[8   ]*3+[5  ]*10+[3  ]*10
        adL =[1 ]*3+[1 ]*2+[1 ]*2+[0.75]*3+[0.5]*10+[0.1]*10
        #mulL=[40]*2+[20]*2+[10]*2+[8   ]*2+[5  ]*3+[3  ]*5
        #adL =[1 ]*2+[1 ]*2+[1 ]*2+[0.75]*2+[0.5]*3+[0.25]*5
        time=365*86400*100
        quake['dep']=10+10*np.random.rand(1)
        quake['time']=float(quake['time'])
        self.maxErr=maxErr
        if maxDT<0:
            maxDT=self.maxDT
        for i in range(len(quake)):
            record=quake.records[i]
            staInfo=self.staInfos[record['staIndex']]
            if record['pTime']<time:
                time=record['pTime']
                dLa=float((np.random.rand()-0.5)*0.3)
                dLo=float((np.random.rand()-0.5)*0.3)
                dLz=float((np.random.rand()-0.5)*5)
                quake['la'],quake['lo'],quake['dep']=\
                [staInfo['la']+dLa,staInfo['lo']+dLo,10+dLz]
        for i in range(len(mulL)):
            quake,res=self.__locate__(quake,mul=mulL[i],r=r,ad=adL[i],e=e,maxDT=maxDT,\
                isDel=(isDel and i+1==len(mulL)))
        return quake,res
    def __locate__(self,quake,mul=10,r=1,ad=1,e=0.1,maxDT=30,isDel=False):
        phaseL=[]
        timeL=[]
        staIndexL=[]
        rIndexL=[]
        wL=[]
        for i in range(len(quake.records)):
            record=quake.records[i]
            if record['pTime']>0 and record['pTime']-quake['time']<maxDT:
                timeL.append(record['pTime']-quake['time'])
                phaseL.append('p')
                staIndexL.append(record['staIndex'])
                rIndexL.append(i)
                r0=1
                if isinstance(quake,QuakeCC):
                    r0=max(record['pCC'],0.01)
                wL.append(1*r0)
            if record['sTime']>0 and record['sTime']-quake['time']<maxDT*1.7:
                timeL.append(record['sTime']-quake['time'])
                phaseL.append('s')
                staIndexL.append(record['staIndex'])
                rIndexL.append(i)
                r0=1
                if isinstance(quake,QuakeCC):
                    r0=max(record['pCC'],0.01)
                wL.append(r*r0)
        timeL=np.array(timeL)
        wL=np.array(wL).transpose()
        #print(timeL,phaseL)
        gM=self.__timeG__(quake,phaseL,staIndexL)
        dTime=timeL-gM[:,0]
        #print(gM[-1,:],timeL[-1])
        validL=np.where(np.abs(dTime-dTime.mean())<mul*max(0.2,dTime.std()))[0]  
        if len(validL)<5:
            return quake, 999
        if isDel:
            if self.maxErr == -1:
                maxErr=mul*max(0.2,dTime.std())
            else:
                maxErr=self.maxErr
            nvalidL=np.where(np.abs(dTime-dTime.mean())>maxErr)[0]
            for i in  nvalidL:
                rIndex=rIndexL[i]
                if phaseL[i]=='p':
                    quake.records[rIndex]['pTime']=0
                if phaseL[i]=='s':
                    quake.records[rIndex]['sTime']=0
            quake.removeZeros() 
        dTime=dTime[validL]
        gM=gM[validL,1:5]
        wL=wL[validL]
        gM=np.mat(wL.reshape((-1,1))*gM)
        dTime=np.mat(wL*dTime)
        gMT=gM.transpose()
        dM=gMT*dTime.transpose()
        gMTgM=gMT*gM
        gMTgM[2,2]=gMTgM[2,2]+e
        gMTgM[1,1]=gMTgM[1,1]+e
        gMTgM[0,0]=gMTgM[0,0]+e
        gMTgM[3,3]=gMTgM[3,3]+e/100
        MM=np.linalg.pinv(gMTgM)
        dd=MM*dM
        #print(dd)
        quake['la']+=float(dd[0,0])*ad
        quake['lo']+=float(dd[1,0])*ad
        quake['dep']=float(max(-4,quake['dep']+float(dd[2,0])*ad))
        quake['time']+=float(dd[3,0])
        return quake,dTime.std()
    @jit
    def __timeG__(self,quake,phaseL,staIndexL):
        gM=np.zeros((len(phaseL),5))
        loc=quake.loc()
        for i in range(len(phaseL)):
            staLa=self.staInfos[staIndexL[i]]['la']
            staLo=self.staInfos[staIndexL[i]]['lo']
            dep=self.staInfos[staIndexL[i]]['dep']/1000
            dd=0.0001
            ddz=1
            delta=DistAz(loc[0],loc[1],\
                    staLa,staLo).delta
            time=self.timeM.get_travel_times(\
                np.abs(loc[2]+dep),delta,phaseL[i])[0].time
            ddLa=(DistAz(loc[0]+dd,loc[1],\
                    staLa,staLo).delta-delta)/dd
            ddLo=(DistAz(loc[0],loc[1]+dd,\
                    staLa,staLo).delta-delta)/dd
            dTime=(self.timeM.get_travel_times(\
                np.abs(loc[2]+dep),delta+dd,phaseL[i])[0].time-\
                    time)/dd
            ddLaTime=dTime*ddLa
            ddLoTime=dTime*ddLo
            ddZTime=(self.timeM.get_travel_times(np.abs(loc[2]+dep+ddz),delta,phaseL[i])[0].time-\
                    time)/ddz
            gM[i,0]=time
            gM[i,1]=ddLaTime
            gM[i,2]=ddLoTime
            gM[i,3]=ddZTime
            gM[i,4]=1
        #print(gM)
        #return

        return gM

    def getG(self,quake,quakeRef=None,minCC=0.5,minMul=0):
        staIndexL=[]
        phaseL=[]
        if quakeRef != None:
            return self.getGRef(quake,quakeRef,minCC=minCC,minMul=minMul)
        for record in quake:
            if record['pTime']>0:
                if isinstance(record,RecordCC):
                    if record['pCC']<minCC or record.getPMul()<minMul:
                        continue
                staIndexL.append(record['staIndex'])
                phaseL.append('p')
            if record.sTime()>0:
                if isinstance(record,RecordCC):
                    if record['sCC']<minCC or record.getSMul()<minMul:
                        continue
                staIndexL.append(record['staIndex'])
                phaseL.append('s')
        return np.mat(self.__timeG__(quake,phaseL,staIndexL))

    def getGRef(self,quake,quakeRef,minCC=0.5,minMul=0):
        staIndexL=[]
        phaseL=[]
        indexLRef=quakeRef.staIndex()
        for record in quake:
            if record['pTime']>0 and record['staIndex'] in indexLRef>0:
                if isinstance(record,RecordCC):
                    if record['pCC']<minCC or record.getPMul()<minMul\
                     or quakeRef.records\
                     [indexLRef.index(record['staIndex'])]['pTime']<0:
                        continue
                staIndexL.append(record['staIndex'])
                phaseL.append('p')
            if record['sTime']>0 and record['staIndex'] in indexLRef>0:
                if isinstance(record,RecordCC):
                    if record['SCC']<minCC or record.getSMul()<minMul \
                     or quakeRef.records\
                     [indexLRef.index(record['staIndex'])]['sTime']<0:
                        continue
                staIndexL.append(record['staIndex'])
                phaseL.append('s')
        return np.mat(self.__timeGRef__(quake,phaseL,staIndexL,quakeRef))

    def getGM(self,quake,quakeRef=None,minCC=0.5,minMul=0):
        timeG=self.getG(quake,quakeRef=quakeRef,minCC=minCC,minMul=minMul)[:,1:5]
        if timeG.shape[0]>0:
            timeG[:,-1]*=10
            timeG[:,-2]*=111
            G=timeG.transpose()*timeG
            i=np.arange(G.shape[0])
            G/=G[i,i].sum()
            V,v=np.linalg.eig(G)
        else:
            G=np.zeros((3,3))
            V=[-9,-9,-9]
            v=np.zeros((3,3))
        return G,V,v,quake.calCover(self.staInfos,minCC=minCC)

    def locateRef(self, quake,quakeRef,r=1,e=0.00001,maxDT=-1,minCC=0.4):
        mulL=[40]
        adL =[1]
        time=365*86400*100
        if maxDT<0:
            maxDT=self.maxDT
        #quake.loc[2]=10+10*np.random.rand(1)
        for i in range(3):
            quake['la'],quake['lo'],quake['dep']=\
            quakeRef['la'],quakeRef['lo'],quakeRef['dep']
        for i in range(len(mulL)):
            quake,res=self.__locateRef__(quake,quakeRef,mul=mulL[i],\
                r=r,ad=adL[i],e=e,maxDT=maxDT,minCC=minCC)
        return quake,res
    def __locateRef__(self,quake,quakeRef,mul=10,r=1,ad=1,\
        e=0.00001,maxDT=35,minCC=0.4):
        phaseL=[]
        timeL=[]
        staIndexL=[]
        wL=[]
        indexLRef=quakeRef.staIndexs()
        for record in quake.records:
            index = record['staIndex']
            if index in indexLRef:
                indexRef = indexLRef.index(index)
            else:
                continue
            if record['pTime']>0 and quakeRef.records[indexRef]['pTime']>0 \
            and record['pTime']-quake['time']<maxDT:
                timeL.append(record['pTime']-quake['time'])
                #print(record.pTime(),timeL[-1])
                phaseL.append('p')
                staIndexL.append(index)
                r0=1
                if isinstance(quake,QuakeCC):
                    r0=max(record['pCC']**2,0.01)
                    if record['pCC']<minCC:
                        r0=1e-4
                wL.append(1*r0)
            if record['sTime']>0 and quakeRef.records[indexRef]['sTime']>0 \
            and record['sTime']-quake['time']<maxDT*1.7:
                timeL.append(record['sTime']-quake['time'])
                #print(record.pTime(),timeL[-1])
                phaseL.append('s')
                staIndexL.append(index)
                r0=1
                if isinstance(quake,QuakeCC):
                    r0=max(record['sCC']**2,0.01)
                    if record['sCC']<minCC:
                        r0=1e-4
                wL.append(1*r0)
        timeL=np.array(timeL)
        wL=np.array(wL).transpose()
        gM=self.__timeGRef__(quake,phaseL,staIndexL,quakeRef,minCC=minCC)
        dTime=timeL-gM[:,0]
        #print(quake.time,quakeRef.time)
        #print(timeL)
        #print(gM[:,0])
        validL=np.where(np.abs(dTime-dTime.mean())<mul*max(0.2,dTime.std()))[0]
        if len(validL)<5:
            return quake, 999
        dTime=dTime[validL]
        gM=gM[validL,1:5]
        wL=wL[validL]
        #quake.time=quake.time+dTime.mean()
        #dTime=dTime-dTime.mean()
        gM=np.mat(wL.reshape((-1,1))*gM)
        dTime=np.mat(wL*dTime)
        gMT=gM.transpose()
        dM=gMT*dTime.transpose()
        gMTgM=gMT*gM
        #e=0.000001
        gMTgM[2,2]=gMTgM[2,2]+e
        gMTgM[1,1]=gMTgM[1,1]+e
        gMTgM[0,0]=gMTgM[0,0]+e
        gMTgM[3,3]=gMTgM[3,3]+e/100
        MM=np.linalg.pinv(gMTgM)
        dd=MM*dM
        quake['la']+=float(dd[0,0])*ad
        quake['lo']+=float(dd[1,0])*ad
        quake['dep']=float(max(-10,quake['dep']+float(dd[2,0])*ad))
        quake['time']+=float(dd[3,0])
        dTime=dTime.transpose()-gM*dd
        return quake,dTime.std()
    def __timeGRef__(self,quake,phaseL,staIndexL,quakeRef,minCC=0.5):
        gM=np.zeros((len(phaseL),5))
        indexLRef=quakeRef.staIndexs()
        indexL=quake.staIndexs()
        loc=quake.loc()
        for i in range(len(phaseL)):
            index=indexL.index(staIndexL[i])
            indexRef=indexLRef.index(staIndexL[i])
            staLa=self.staInfos[staIndexL[i]]['la']
            staLo=self.staInfos[staIndexL[i]]['lo']
            dep=self.staInfos[staIndexL[i]]['dep']/1000
            dd=0.0001
            ddz=0.1
            delta=DistAz(quake['la'],quake['lo'],\
                staLa,staLo).delta
            time=self.timeM.get_travel_times(quake['dep']+dep,delta,phaseL[i])[0].time
            ddLa=(DistAz(quake['la']+dd,quake['lo'],\
                staLa,staLo).delta-delta)/dd
            ddLo=(DistAz(quake['la'],quake['lo']+dd,\
                staLa,staLo).delta-delta)/dd
            dTime=(self.timeM.get_travel_times(loc[2]+dep,delta+dd,phaseL[i])[0].time-\
                time)/dd
            ddLaTime=dTime*ddLa
            ddLoTime=dTime*ddLo
            ddZTime=(self.timeM.get_travel_times(loc[2]+dep+ddz,delta,phaseL[i])[0].time-\
                time)/ddz
            gM[i,0]=time
            if phaseL[i]=='p':
                if quakeRef.records[indexRef]['pTime']>10:
                    gM[i,0]=quakeRef.records[indexRef]['pTime']-quakeRef['time']
                    #print(staIndexL[i],gM[i,0],p[staIndexL[i]]-quake.time)
            else:
                if quakeRef.records[indexRef]['sTime']>10:
                    gM[i,0]=quakeRef.records[indexRef]['sTime']-quakeRef['time']
                    #print(gM[i,0])
            gM[i,1]=ddLaTime
            gM[i,2]=ddLoTime
            gM[i,3]=ddZTime
            gM[i,4]=1
        return gM

def getRefM(quakeRefs,staInfos):
    staN=len(staInfos)
    qN=len(quakeRefs)
    timeM=np.zeros((qN,staN))
    laloM=np.zeros((qN,2))
    for i in range(qN):
        timeM[i,:]=np.sign(quakeRefs[i].getPTimeL(staInfos)).reshape((1,-1))
        laloM[i,:]=np.array(quakeRefs[i].loc()[:2]).reshape((1,-1))
    return timeM,laloM

def findNearQuake(quake,timeM,laloM,staInfos,maxDis=0.2,minSta=5):
    sameStaN=(timeM*quake.getPTimeL(staInfos)).sum(axis=1)
    dis=np.linalg.norm(laloM-np.array(quake.loc()[:2]).reshape((1,-1)),axis=1)
    index=(dis/(1+sameStaN/100*(dis<maxDis))+100*(sameStaN<minSta)).argmin()
    if dis[index]>maxDis or sameStaN[index] < minSta:
        return None
    return index
def getTmpD(quakeTmpL):
    tmpD={}
    count=0
    for quake in quakeTmpL:
        tmpD[quake.filename]=count
        count+=1
    return tmpD
def relocQuakeByTmp(quakeLs,quakeTmpL,staInfos):
    loc=locator(staInfos)
    tmpD=getTmpD(quakeTmpL)
    for quakeL in quakeLs:
        for quake in quakeL:
            loc.locateRef(quake,quakeTmpL[tmpD[quake.filename]])
    return quakeLs


def relocQuakeLs(quakeLs,quakeRefs,staInfos):
    timeM,laloM=getRefM(quakeRefs,staInfos)
    quakeRelocL=[]
    loc=locator(staInfos)
    count0=0
    count=0
    for quakeL in quakeLs:
        for quake in quakeL:
            #quake,res=loc.locate(quake)
            index=findNearQuake(quake,timeM,laloM,staInfos,maxDis=0.2)
            count0+=1
            #
            if index!=None:
                count+=1
                #print(index)
                print(count0,count,'###',quake.loc(),quakeRefs[index].loc())
                quake,res=loc.locateRef(quake,quakeRefs[index])
                print(count0,count,'***',quake.loc(),quakeRefs[index].loc(),res)
                quakeRelocL.append(quake)
    return quakeRelocL

