import numpy as np
from numba import jit, float32, int64
from obspy import UTCDateTime
from tool import  QuakeCC, RecordCC
from mathFunc import getDetec, cmax,corrNP
import time as Time
from .. import mathTool
from mathTool.distaz import DistAz
from multiprocessing import Process, Manager
import cudaFunc
import torch
'''
in this part we do WMFT in GPU
to run fast and reduce the GPU memory, we would like to 
'''
defaultSecL=[-2,2]#[-1,3]
tentype=cudaFunc.tentype
nptype=cudaFunc.nptype
dtype=cudaFunc.dtype
nptypeO=cudaFunc.nptypeO
minF=cudaFunc.minF
maxF=cudaFunc.maxF
torch.set_default_tensor_type(tentype)
convert=cudaFunc.convert
maxStaN=20
isNum=False
corrTorch=cudaFunc.torchcorrnn

def getTimeLim(staL):
    n = len(staL)
    bTime = UTCDateTime(1970,1,1).timestamp
    eTime = UTCDateTime(2200,12,31).timestamp
    for i in range(n):
        bTime = max([staL[i].bTime, bTime])
        eTime = min([staL[i].eTime, eTime])
    return bTime, eTime

def doMFT(staL,T3PSL,bTime, n, wM=np.zeros((2*maxStaN,86700*50)\
    ,dtype=nptype),delta=0.02,minMul=3,MINMUL=8, winTime=0.4,\
    minDelta=20*50, locator=None,tmpName=None,quakeRef=None,\
    maxCC=1,R=[-90,90,-180,180],staInfos=None,maxDis=200,\
    deviceL=['cuda:0'],minChannel=8,mincc=0.4,secL=defaultSecL):
    time_start = Time.time()
    winL=int(winTime/delta)
    staSortL = np.argsort(quakeRef.records)
    #tmpTimeL = np.arange(defaultSecL[0],defaultSecL[1],delta).astype(nptype)
    #tmpRefTime=waveform['indexL'][0][0]*delta
    #tmpIndexL=((tmpTimeL-tmpRefTime)/delta).astype(np.int64)
    aM=torch.zeros(n,device=deviceL[0],dtype=dtype)
    rIndexL=[]
    index=0
    staIndexL=[]
    phaseTypeL=[]
    mL=[]
    sL=[]
    oTimeL=[]
    if isinstance(quakeRef,list):
        oTime=quakeRef.time
    rIndex=0
    pCount=0
    for rIndex in staSortL:
        staIndex=quakeRef.records[rIndex]['staIndex']
        record = quakeRef.records[rIndex]
        if staIndex>=len(staL):
            continue
        if staInfos!=None:
            staInfo=staInfos[staIndex]
            if staInfo['la']<R[0] or \
                staInfo['la']>R[1] or \
                staInfo['lo']<R[2] or \
                staInfo['lo']>R[3]:
                continue
        if quakeRef !=None and staInfos !=None:
            staInfo=staInfos[staIndex]
            if staInfos[staIndex].dist(quakeRef)>maxDis:
                continue
        
        if record['pTime']>0 and staL[staIndex].data.data.shape[-1]>1000:
            if (staL[staIndex].data.data==0).sum()>60*3/staL[staIndex].data.delta:
                continue
            dTime=(record['pTime']-oTime+bTime-staL[staIndex].data.bTime.timestamp)
            dIndex=int(dTime/delta)
            if dIndex<0:
                continue
            if len(T3PSL[0][rIndex].data)==0:
                continue
            c,m,s=corrTorch(staL[staIndex].data.data[2,:],T3PSL[0][rIndex].data[2])
            if torch.isnan(c).sum()>0:
                print(staIndex,rIndex)
            if s==1:
                continue
            rIndexL.append(rIndex)
            staIndexL.append(staIndex)
            phaseTypeL.append(1)
            oTimeL.append(dTime+staL[staIndex].data.bTime.timestamp\
                -secL[0])
            mL.append(m)
            sL.append(s)
            wM[index]=torch.zeros(n+50*100,device=c.device)
            wM[index][0:c.shape[0]-dIndex]=c[dIndex:]
            threshold=m+minMul*s
            if threshold>mincc:
                threshold=mincc
            cudaFunc.torchMax(c[dIndex:],threshold,winL, aM)
            index+=1
            pCount+=1
            if pCount>=maxStaN:
                break
        if record['sTime']>0 and staL[staIndex].data.data.shape[-1]>1000:
            dTime=(record['sTime']-oTime+bTime\
                -staL[staIndex].data.bTime.timestamp)
            dIndex=int(dTime/delta)
            if dIndex<0:
                continue
            chanelIndex=0
            if len(T3PSL[1][rIndex].data)==0:
                continue
            if T3PSL[1][rIndex][1].max()>T3PSL[1][rIndex][0].max():
                chanelIndex=1
            c,m,s=corrTorch(staL[staIndex].data.data[chanelIndex,:],\
                T3PSL[1][rIndex][chanelIndex])
            if s==1:
                continue
            rIndexL.append(rIndex)
            staIndexL.append(staIndex)
            phaseTypeL.append(2)
            oTimeL.append(dTime+staL[staIndex].data.bTime.timestamp\
                -secL[0])
            mL.append(m)
            sL.append(s)
            wM[index]=torch.zeros(n+50*100,device=c.device,dtype=dtype)
            wM[index][0:c.shape[0]-dIndex]=c[dIndex:]
            threshold=m+minMul*s
            if threshold>mincc:
                threshold=mincc
            cudaFunc.torchMax(c[dIndex:],threshold,winL,aM)
            index+=1
    if index<minChannel:
        return []
    aM/=index
    aMNew=aM[aM>-2]
    aMNew=aMNew[aMNew!=0]
    M=aMNew[10000:-10000].mean().cpu().numpy().astype(nptypeO)
    S=aMNew[10000:-10000].std().cpu().numpy().astype(nptypeO)
    if S<5e-3:
        return []
    threshold=min(maxCC,M+MINMUL*S)
    indexL, vL= getDetec(aM.cpu().numpy().astype(nptypeO),\
     minValue=threshold, minDelta=minDelta)
    print("M: %.5f S: %.5f thres: %.3f peakNum:%d num:%d"%\
        (M,S,threshold,len(indexL),index))
    print('corr',Time.time()-time_start)
    wLL=np.arange(-10,winL)
    quakeL=[]
    for i in range(len(indexL)):
        cc=vL[i]
        index = indexL[i]
        if index+wLL[0]<0:
            print('too close to the beginning')
            continue
        time= index*delta+bTime
        staD={}
        quakeCC = QuakeCC(cc=cc,M=M,S=S,la=quakeRef['la'],lo=quakeRef['lo'],\
            dep=quakeRef['dep'], time=time,\
         tmpName=tmpName)
        phaseCount=0
        for j in range(len(staIndexL)):
            staIndex=staIndexL[j]
            dIndex=wM[j][index+wLL].argmax().cpu().numpy()
            phaseTime=float(oTimeL[j]+(wLL[dIndex]+index)*delta)
            if phaseTypeL[j]==1:
                quakeCC.append(RecordCC(staIndex=staIndex, pTime=phaseTime,sTime=0,\
                    pCC=wM[j][index+wLL[dIndex]].cpu().numpy().astype(nptypeO)\
                    , sCC=0, pM=mL[j].astype(nptypeO), pS=sL[j].astype(nptypeO), sM=0, sS=0))
                staD[staIndex]=phaseCount
                phaseCount+=1
            if phaseTypeL[j]==2:
                j0=staD[staIndex]
                quakeCC[j0]['sTime']=phaseTime
                quakeCC[j0]['sCC']=wM[j][index+wLL[dIndex]].cpu().numpy().astype(nptypeO)
                quakeCC[j0]['sM']=mL[j].astype(nptypeO)
                quakeCC[j0]['sS']=sL[j].astype(nptypeO)
        if locator != None and len(quakeCC)>=3:
            if quakeRef==None:
                quakeCC,res=locator.locate(quakeCC)
            else:
                try:
                    quakeCC,res=locator.locateRef\
                    (quakeCC,quakeRef,minCC=0.2)
                except:
                    print('wrong in locate')
                else:
                    print(quakeCC['time'],quakeCC.loc(),res,quakeCC['cc'])
                    pass
            
            if False:
                try:
                    if quakeRef==None:
                        quakeCC,res=locator.locate(quakeCC)
                    else:
                        quakeCC,res=locator.locateRef(quakeCC,quakeRef)
                    print(quakeCC.time,quakeCC.loc,res,quakeCC.cc)
                except:
                    print('wrong in locate')
                else:
                    pass
        quakeL.append(quakeCC)
    time_end=Time.time()
    print(time_end-time_start)
    return quakeL

def doMFTAll(staL,T3PSLL,bTime,n=86400*50,delta=0.02\
        ,minMul=4,MINMUL=8, winTime=0.4,minDelta=20*50, \
        locator=None,tmpNameL=None, isParallel=False,\
        NP=2,quakeRefL=None,maxCC=1,R=[-90,90,-180,180],\
        maxDis=200,isUnique=True,isTorch=True,deviceL=['cuda:0'],\
        minChannel=8,mincc=0.4,secL=defaultSecL):
    for sta in staL:
        sta.data=sta.Data()
    for i in range(T3PSLL):
        T3PSL = T3PSLL[i]
        quakeRef = quakeRefL[i]
        for j in range(len(T3PSL[0])):
            T3=T3PSL[0][j]
            record = quakeRefL[i].records[j]
            if T3.pTime>0:
                T3.data = T3.Data(secL[0]+T3.pTime,T3.pTime+secL[-1]).transpose()
                if len(T3.data)>0:
                    T3.data=torch.tensor(T3.data,\
                            device=deviceL[record['staIndex']%len(deviceL)],dtype=dtype)
            T3=T3PSL[1][j]
            if T3.sTime>0:
                T3.data = T3.Data(secL[0]+T3.sTime,T3.sTime+secL[-1]).transpose()
                if len(T3.data)>0:
                    T3.data=torch.tensor(T3.data,\
                            device=deviceL[record['staIndex']%len(deviceL)],dtype=dtype)
    if not isParallel:
        quakeL=[]
        wM=[None for i in range(maxStaN*2)]
        count=0
        for sta in staL:
            count+=1
            sta.data.data = sta.data.Data()
            if sta.data.data.shape[0]>1/delta or sta.data.data.shape[-1]>1/delta:
                if sta.data.data.shape[0]>sta.data.data.shape[-1]:
                    sta.data.data=sta.data.data.transpose()
                if not isinstance(sta.data.data,torch.Tensor):
                    sta.data.data=(sta.data.data*convert).astype(nptype)
                    if  isTorch :
                        sta.data.data=torch.tensor(sta.data.data,\
                            device=deviceL[(count)%len(deviceL)],dtype=dtype)
            if sta.data.data.shape[-1]>11*3600/delta:
                bTime=max(bTime, sta.data.bTime.timestamp+1)
        for i in range(len(T3PSLL)):
            print('doing on %d find %d'%(i,len(quakeL)))
            if tmpNameL!=None:
                tmpName=tmpNameL[i]
            else:
                tmpName=None
            quakeRef=None
            if quakeRefL!=None:
                quakeRef=quakeRefL[i]
            quakeL=quakeL+doMFT(staL,T3PSLL[i],bTime,n,wM=wM,\
                delta=delta,minMul=minMul,MINMUL=MINMUL,\
                winTime=winTime, minDelta=minDelta,locator=locator,\
                tmpName=tmpName, quakeRef=quakeRef,\
                maxCC=maxCC,R=R,maxDis=maxDis,deviceL=deviceL,\
                minChannel=minChannel,mincc=mincc)
            if i%20==0 and isUnique:
                quakeL=uniqueQuake(quakeL)

        if isUnique:
            quakeL=uniqueQuake(quakeL)
        return quakeL


def __doMFTAll(staLP,waveformLP,bTime,quakeLP,n=86400*50,delta=0.02\
        ,minMul=4,MINMUL=8, winTime=0.4,minDelta=20*50, \
        locator=None,tmpNameL=None,NP=2,IP=0):
    staL=staLP[0]
    waveformL=waveformLP[0]
    quakeL=[]
    wM=np.zeros((2*maxStaN,n+50*100),dtype=nptype)
    for i in range(IP,len(waveformL),NP):
        print('doing on %d'%i)
        if tmpNameL!=None:
            tmpName=tmpNameL[i]
        else:
            tmpName=None
        quakeL=quakeL+doMFT(staL,waveformL[i],bTime,n,wM=wM,delta=delta,minMul=minMul,MINMUL=MINMUL,\
             winTime=winTime, minDelta=minDelta,locator=locator,tmpName=tmpName)
    quakeLP.append(quakeL)

def uniqueQuake(quakeL,minDelta=5, minD=0.2):
    PS=np.zeros((len(quakeL),7))
    for i in range(len(quakeL)):
        PS[i,0]=i
        PS[i,1]=quakeL[i]['time']
        PS[i,2:4]=quakeL[i].loc()[0:2]
        PS[i,4]=quakeL[i].getMul(isNum)
        PS[i,5]=quakeL[i]['cc']
        PS[i,6]=quakeL[i]['M']
    L=np.argsort(PS[:,1])
    PS=PS[L,:]
    L=uniquePS(PS,minDelta=minDelta,minD=minD)
    quakeLTmp=[]
    for i in L:
        quakeLTmp.append(quakeL[i])
    return quakeLTmp


@jit
def uniquePS(PS,minDelta=20, minD=0.5):
    L=[]
    N=len(PS[:,0])
    for i in range(N) :
        isMax=1
        if np.isnan(PS[i,5]) or np.isnan(PS[i,6]):
            continue
        for j in range(i-1,0,-1):
            if np.isnan(PS[j,5]):
                continue
            if PS[j,1]<PS[i,1]-minDelta:
                break
            if np.linalg.norm(PS[j,2:3]-PS[i,2:3])>minD:
                continue
            if PS[j,4]>PS[i,4]:
                isMax=0
                break
        for j in range(i+1,N):
            if np.isnan(PS[i,5]) or np.isnan(PS[i,6]):
                continue
            if PS[j,1]>PS[i,1]+minDelta:
                break
            if np.linalg.norm(PS[j,2:3]-PS[i,2:3])>minD:
                continue
            if PS[j,4]>PS[i,4]:
                isMax=0
                break
        if isMax==1:
            L.append(int(PS[i,0]))
    return L



