import obspy
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
import threading
import time
from time import ctime
import math
from numba import jit
from ..mathTool.mathFunc_bak import getDetec, prob2color
from ..io import tool
from ..io.seism import getTrace3ByFileName,Quake,QuakeL,Record,QuakeCC,RecordCC,t0,t1
from ..io.sacTool import staTimeMat

from ..mapTool.mapTool import readFault,plotTopo,faultL
import mpl_toolkits.basemap as basemap
import torch
from obspy import taup
taupModel=taup.TauPyModel(model='iasp91')
maxA=2e19
os.environ["MKL_NUM_THREADS"] = "32"
@jit
def isZeros(a):
    new = a.reshape([-1,10,a.shape[-1]])
    if (new.std(axis=(1))==0).sum()>5:
        return True
    return False


'''
@jit
def predictLongData(model, x, N=2000, indexL=range(750, 1250)):
    if len(x) == 0:
        return np.zeros(0)
    N = x.shape[0]
    Y = np.zeros(N)
    perN = len(indexL)
    loopN = int(math.ceil(N/perN))
    perLoop = int(1000)
    inMat = np.zeros((perLoop, 2000, 1, 3))
    #print(len(x))
    zeroCount=0
    for loop0 in range(0, int(loopN), int(perLoop)):
        loop1 = min(loop0+perLoop, loopN)
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                inMat[loop-loop0, :, :, :] = processX(x[sIndex: sIndex+2000, :])\
                .reshape([2000, 1, 3])
        outMat = (model.predict(inMat)[:,:,:,:1]).reshape([-1, 2000])
        for loop in range(loop0, loop1):
            i = loop*perN
            if isZeros(inMat[loop-loop0, :, :, :]):
                zeroCount+=1
                continue
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                Y[indexL0[0]+sIndex: indexL0[-1]+1+sIndex] = \
                np.append(Y[indexL0[0]+sIndex: indexL0[-1]+1+sIndex],\
                    outMat[loop-loop0, indexL0].reshape([-1])\
                    ).reshape([2,-1]).max(axis=0)
    if zeroCount>0:
        print('zeros: %d'%zeroCount)
                
    return Y
'''
indexL0=range(275, 1500)
@jit
def predictLongData(model, x, N=2000, indexL=range(750, 1250),dIndex=2000,dec=1):
    L = x.shape[0]
    if L <=dIndex*3:
        return np.zeros(L)
    validD = len(indexL)
    loop = math.ceil(dIndex/validD)
    zerosCount=0
    out = np.zeros([loop,L],np.float32)
    for i in range(loop):
        i0=validD*i
        i1=int((L-i0)/dIndex)*dIndex+i0
        X = x[i0:i1].reshape([-1,dIndex,1,3])
        XSTD = X.reshape([X.shape[0],-1,10,3]).std(axis=2,keepdims=True)
        sum0 = (XSTD==0).sum(axis=(1,2,3))
        X/=X.std(axis=(1,2,3),keepdims=True)+np.finfo(x.dtype).eps
        Y = model.predict(X)
        zerosCount+=(sum0>5).sum()
        Y[sum0>5]*=0
        Y[:, :indexL0[ 0] ]*=0
        Y[:,  indexL0[-1]:]*=0
        out[i,i0:i1]=Y.reshape([-1])
    if zerosCount>0:
        print('zeros: %d'%zerosCount)          
    return out.max(axis=0) 
'''
@jit
def processX(X, rmean=True, normlize=True, reshape=True):
    if reshape:
        X = X.reshape(-1, 2000, 1, 3)
    if rmean:
        X = X - X.mean(axis=(1, 2)).reshape([-1, 1, 1, 3])
    if normlize:
        X = X/(X.std(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))
    return X
'''
@jit
def processX(X, rmean=True, normlize=False, reshape=True,isNoise=False,num=2000):
    if reshape:
        X = X.reshape(-1, num, 1, 3)
    #print(X.shape)
    if rmean:
        X-= X.mean(axis=1,keepdims=True)
    if normlize:
        X /=(X.std(axis=(1, 2, 3),keepdims=True))
    if isNoise:
        X+=(np.random.rand(X.shape[0],num,1,3)-0.5)*np.random.rand(X.shape[0],1,1,3)*X.max(axis=(1,2,3),keepdims=True)*0.15*(np.random.rand(X.shape[0],1,1,1)<0.1)
    return X



def originFileName(net, station, comp, YmdHMSJ, dirL=['data/']):
    #dir='tmpSacFile/'
    sacFileNames = list()
    Y = YmdHMSJ
    for dir in dirL:
        sacFileNamesStr = dir+net+'.'+station+'.'+Y['Y']+Y['j']+\
            '*'+comp
        for file in glob(sacFileNamesStr):
            sacFileNames.append(file)
    return sacFileNames

class sta(object):
    def __init__(self, station, day,freq=[-1, -1], \
        taupM=tool.quickTaupModel(),delta0=0.02,\
        R=[-91,91,-181,181]):
        self.net = station['net']
        self.loc = station.loc()
        self.station = station['sta']
        self.sta = station
        self.day = day
        self.taupM=taupM
        if self.loc[0]<R[0] or self.loc[0]>R[1] \
        or self.loc[1]<R[2] or self.loc[1]>R[3]:
            self.data=getTrace3ByFileName([[],[],[]], freq=freq,delta0=delta0,\
                maxA=maxA,isData=False)
            print('skip')
        else:
            self.data = getTrace3ByFileName(station.getFileNames(self.day),\
             freq=freq,delta0=delta0,\
                maxA=maxA,isData=False)
        print(self.station,self.data.bTime,self.data.eTime)
    def __del__(self):
        try:
            del(self.data)
            torch.cuda.empty_cache()
        except:
            pass
        else:
            pass
    def predict(self,modelL=None, staTimeM=None,\
     mode='mid', isClearData=False,maxD=80,decPre=1,maxDTime=2):
        self.timeL = list()
        self.vL = list()
        self.mode = mode
        indexLL = [range(275, 775), range(275, 775)]
        if mode=='norm':
            minValueL=[0.5,0.5]
        if mode=='high':
            minValueL=[0.4, 0.4]
        if mode=='mid':
            minValueL=[0.25, 0.25]
        if mode=='low':
            minValueL=[0.2, 0.2]
        if mode=='higher':
            minValueL=[0.6, 0.6]
        minDeltaL=[500, 750]
        for i in range(len(modelL)):
            y = predictLongData(modelL[i], self.data.Data(),\
             indexL=indexLL[i],dec=decPre)
            #print(y.max(),y.std(),len(y))
            tmpL = getDetec(y, minValue=minValueL[i], minDelta =\
              minDeltaL[i])
            print(ctime(),'find',len(tmpL[0]))
            self.timeL.append(tmpL[0])
            self.vL.append(tmpL[1])
        self.pairD = self.getPSPair(maxD=maxD)
        self.isPick = np.zeros(len(self.pairD))
        self.orignM = self.convertPS2orignM(staTimeM,maxD=maxD,maxDTime=maxDTime)
        if isClearData:
            self.clearData()

    def __repr__(self):
        reprStr=self.net + ' '+self.station+\
        str(self.loc)
        return 'detec in station '+ reprStr


    def getSacFileNamesL(self, station):
        return station.getFileNames(self.day)

    def clearData(self):
        self.data.data = np.zeros((0, 3))

    def plotData(self):
        colorStr = ['.r', '.g']
        plt.plot(self.data.data[:,2]/self.data.data[:,2].max()\
            + np.array(0))
        for i in range(len(self.timeL)):
            plt.plot(self.timeL[i],self.vL[i], colorStr[i])
        plt.show()

    def calOrign(self, pTime, sTime):
        return self.taupM.get_orign_times(pTime, sTime, self.data.delta)

    def getPSPair(self, maxD=80):
        pairD = list()
        if len(self.timeL) == 0:
            return pairD
        if self.data.delta<=0:
            return pairD
        maxN = maxD/self.data.delta
        pN=len(self.timeL[0])
        sN=len(self.timeL[1])
        j0=0
        for i in range(pN):
            pTime = self.timeL[0][i]
            if i < pN-1 and self.mode != 'low':
                pTimeNext = self.timeL[0][i+1]
            else:
                pTimeNext= self.timeL[0][i]+maxN
            pTimeNext = min(pTime+maxN, pTimeNext)
            isS = 0
            for j in range(j0, sN):
                if isS==0:
                    j0=j
                if self.timeL[1][j] > pTime and self.timeL[1][j] < pTimeNext:
                    sTime=self.timeL[1][j]
                    #print(pTime, sTime)
                    pairD.append([pTime*self.data.delta, sTime*self.data.delta\
                        , self.calOrign(pTime, sTime)*self.data.delta, \
                        (sTime-pTime)*self.data.delta, i, j])
                    isS=1
                if self.timeL[1][j] >= pTimeNext:
                    break
        return pairD

    def convertPS2orignM(self, staTimeM, maxDTime=2,maxD=100):
        laN = staTimeM.minTimeD.shape[0]
        loN = staTimeM.minTimeD.shape[1]
        orignM = [[list() for j in range(loN)] for i in range(laN)]
        if len(self.pairD)==0:
            return orignM
        bSec = self.data.bTime.timestamp
        timeL = np.zeros(len(self.pairD))
        for i in range(len(self.pairD)):
            timeL[i] = self.pairD[i][2]+bSec
        sortL = np.argsort(timeL)
        for laIndex in range(laN):
            for loIndex in range(loN):
                minPairTime=staTimeM.minTimeD[laIndex][loIndex] - maxDTime
                if minPairTime>maxD:
                    continue
                maxPairTime=staTimeM.maxTimeD[laIndex][loIndex] + maxDTime
                for i in sortL:
                    if self.pairD[i][3] >= minPairTime \
                    and self.pairD[i][3] <= maxPairTime:
                        pTime = self.pairD[i][0]+bSec
                        sTime = self.pairD[i][1]+bSec
                        timeTmp = [pTime, sTime, timeL[i], i]
                        orignM[laIndex][loIndex].append(timeTmp)
        return orignM
    def filt(self,f=[-1,-1],filtOrder=2):
        self.data.filt(f,filtOrder)
        return self
    def resample(self,resampleN):
        self.data.resample(resampleN)
        return self
    def pickQuake(self,quake,modelL,bSec=-10,eSec=10,bCount=-3000,eCount=3000):
        deg = quake.dist(self.sta)/111.19
        dep = self.sta['dep']/1000+quake['dep']
        timeL=[0,0]
        proL=[-1,-1]
        pTime0= self.getEarliest(taupModel.get_travel_times(dep, deg, \
                ['p', 'P', 'PP', 'pP','Pn']))+quake['time']
        sTime0= self.getEarliest(taupModel.get_travel_times(dep, deg, \
                ['s', 'S', 'SS', 'sS','Sn']))+quake['time']
        time0L = [pTime0,sTime0]
        for i in range(2):
            time0= time0L[i]
            bTime=time0+bCount*self.data.Delta()
            eTime=time0+eCount*self.data.Delta()
            bIndex=int((time0+bSec-bTime)/self.data.Delta())
            eIndex=int((time0+eSec-bTime)/self.data.Delta())
            if self.data.bTime<bTime and self.data.eTime>eTime:
                data = self.data.Data(bTime=bTime,eTime=eTime)
                if len(data)>4000:
                    y = predictLongData(modelL[i], data,indexL =range(275, 775))
                    yMax = y[bIndex:eIndex].max()
                    proL[i]=yMax
                    if yMax>=0.5:
                        timeL[i]=(y[bIndex:eIndex].argmax()+bIndex)*self.data.Delta()+bTime
        return timeL+proL

    def getEarliest(self,arrivals):
        time=10000000
        if len(arrivals)==0:
            print('no phase')
            return 0
        for arrival in arrivals:
            time = min(time, arrival.time)
        return time
def argMax2D(M):
    maxValue = np.max(M)
    maxIndex = np.where(M==maxValue)
    return maxIndex[0][0], maxIndex[1][0]

def associateSta(staL, aMat, staTimeML, timeR=30, minSta=3, maxDTime=3, N=1, \
    isClearData=False, locator=None, maxD=80,taupM=tool.quickTaupModel()):
    timeN = int(timeR)*2
    startTime = obspy.UTCDateTime(2100, 1, 1)
    endTime = obspy.UTCDateTime(1970, 1, 1)
    staN = len(staL)
    for staIndex in range(staN):
        if isClearData:
            staL[staIndex].clearData()
        staL[staIndex].isPick = staL[staIndex].isPick*0
    for staTmp in staL:
        if staTmp.data.bTime<=t0+10:
            continue
        startTime = min(startTime, staTmp.data.bTime)
        endTime = max(endTime, staTmp.data.eTime)
    startSec = int(startTime.timestamp-3600)
    endSec = int(endTime.timestamp+3600)
    if N==1:
        quakeL=QuakeL()
        __associateSta(quakeL, staL, \
            aMat, staTimeML, startSec, \
            endSec, timeR=timeR, minSta=minSta,\
             maxDTime=maxDTime,locator=locator,maxD=maxD)
        return quakeL
    for i in range(len(staL)):
        staL[i].clearData()
    manager=Manager()
    quakeLL=[manager.list() for i in range(N)]
    perN = int(int((endSec-startSec)/N+1)/timeN+1)*timeN
    processes=[]
    for i in range(N):
        process=Process(target=__associateSta, args=(quakeLL[i], \
            staL, aMat, staTimeML, startSec+i*perN, \
            startSec+(i+1)*perN+1))
        #process.setDaemon(True)
        process.start()
        processes.append(process)

    for process in processes:
        print(process)
        process.join()

    for quakeLTmp in quakeLL:
        for quakeTmp in quakeLTmp:
            quakeL.append(quakeTmp)
    return quakeL
    

def __associateSta(quakeL, staL, aMat, staTimeML, startSec, endSec, \
    timeR=30, minSta=3, maxDTime=3, locator=None,maxD=80,\
    taupM=tool.quickTaupModel()):
    typeO = np.int16#in maxD determined Range, if the max station cound is small than 125,use np.int8 else
    #,Using np.int16

    print('start', startSec, endSec)
    laN = aMat.laN
    loN = aMat.loN
    staN = len(staL)
    timeN = int(timeR)*90
    stackM = np.zeros((timeN*3, laN, loN),typeO)
    tmpStackM=np.zeros((timeN*3+3*maxDTime, laN, loN),typeO)
    stackL = np.zeros(timeN*3)
    staMinTimeL=np.ones(staN)*0
    quakeCount=0
    dTimeL=np.arange(-maxDTime, maxDTime+1)
    for loop in range(2):
        staOrignMIndex = np.zeros((staN, laN, loN), dtype=int)
        staMinTimeL=np.ones(staN)*0
        count=0
        for sec0 in range(startSec-3*timeN, endSec+3*timeN, timeN):
            count=count+1
            if count%10==0:
                print(ctime(),'process:',(sec0-startSec)/(endSec-startSec)*100,'%  find:',len(quakeL))
            stackM[0:2*timeN, :, :] = stackM[timeN:, :, :]
            stackM[2*timeN:, :, :] = stackM[0:timeN, :, :]*0
            tmpStackM=tmpStackM*0
            st=sec0+2*timeN - maxDTime
            et=sec0+3*timeN + maxDTime
            for staIndex in range(staN):
                tmpStackM=tmpStackM*0
                for laIndex in range(laN):
                    for loIndex in range(loN):
                        if len(staL[staIndex].orignM[laIndex][loIndex])>0:
                            index0=staOrignMIndex[staIndex, laIndex, loIndex]
                            for index in range(index0, len(staL[staIndex].orignM[laIndex][loIndex])):
                                timeT = staL[staIndex].orignM[laIndex][loIndex][index][2]
                                pairIndex = staL[staIndex].orignM[laIndex][loIndex][index][3]
                                if timeT >et:
                                    staOrignMIndex[staIndex, laIndex, loIndex] = index
                                    break
                                if timeT > st and staL[staIndex].isPick[pairIndex]==0:
                                    pIndex = staL[staIndex].pairD[pairIndex][4]
                                    sIndex = staL[staIndex].pairD[pairIndex][5]
                                    pTime = staL[staIndex].timeL[0][pIndex]
                                    sTime = staL[staIndex].timeL[1][sIndex]
                                    staOrignMIndex[staIndex, laIndex, loIndex] = index
                                    if pTime * sTime ==0:
                                        continue
                                    tmpStackM[int(timeT-sec0)+dTimeL, laIndex, loIndex]=\
                                    tmpStackM[int(timeT-sec0)+dTimeL, laIndex, loIndex]*0+1
                                    '''
                                    for dt in range(-maxDTime, maxDTime+1):
                                        tmpStackM[int(timeT-sec0+dt), laIndex, loIndex]=1
                                    '''
                stackM[2*timeN: 3*timeN, :, :] += tmpStackM[2*timeN: 3*timeN, :, :]
            stackL = stackM.max(axis=(1,2))
            peakL, peakN = tool.getDetec(stackL, minValue=minSta, minDelta=timeR)

            for peak in peakL:
                if peak > timeN and peak <= 2*timeN:
                    time = peak + sec0
                    laIndex, loIndex = argMax2D(stackM[peak, :, :].reshape((laN, loN)))
                    quakeCount+=1
                    quake = Quake(la=aMat[laIndex][loIndex].midLa,lo=aMat[laIndex][loIndex].midLo,\
                        dep=10.0,\
                        time=time, randID=quakeCount)
                    for staIndex in range(staN):
                        isfind=0
                        if staTimeML[staIndex].minTimeS[laIndex,loIndex]\
                        -staTimeML[staIndex].minTimeP[laIndex,loIndex] > maxD:
                            continue
                        if len(staL[staIndex].orignM[laIndex][loIndex]) != 0:
                            for index in range(staOrignMIndex[staIndex, laIndex, loIndex], -1, -1):
                                if int(abs(staL[staIndex].orignM[laIndex][loIndex][index][2]-time))<=maxDTime:
                                    if staL[staIndex].isPick[staL[staIndex].\
                                            orignM[laIndex][loIndex][index][3]]==0:
                                        pairDIndex = staL[staIndex].orignM[laIndex][loIndex][index][3]
                                        pIndex = staL[staIndex].pairD[pairDIndex][4]
                                        sIndex = staL[staIndex].pairD[pairDIndex][5]
                                        if staL[staIndex].timeL[0][pIndex] > 0 and \
                                                staL[staIndex].timeL[1][sIndex] > 0:
                                            quake.Append(Record(staIndex=staIndex, \
                                                pTime=staL[staIndex].orignM[laIndex][loIndex][index][0], \
                                                sTime=staL[staIndex].orignM[laIndex][loIndex][index][1],\
                                                pProb=staL[staIndex].vL[0][pIndex],\
                                                sProb=staL[staIndex].vL[1][sIndex]))
                                            isfind=1
                                            staL[staIndex].timeL[0][pIndex] = 0
                                            staL[staIndex].timeL[1][sIndex] = 0
                                            staL[staIndex].isPick[pairDIndex] = 1
                                            break
                                if staL[staIndex].orignM[laIndex][loIndex][index][2] < time - maxDTime:
                                    break
                            if isfind==0:
                                pTime=0
                                sTime=0
                                pProb=-1
                                sProb=-1
                                pTimeL=staL[staIndex].timeL[0]*staL[staIndex].data.delta\
                                +staL[staIndex].data.bTime.timestamp
                                sTimeL=staL[staIndex].timeL[1]*staL[staIndex].data.delta\
                                +staL[staIndex].data.bTime.timestamp
                                pTimeMin=time+staTimeML[staIndex].minTimeP[laIndex,loIndex]-maxDTime
                                pTimeMax=time+staTimeML[staIndex].maxTimeP[laIndex,loIndex]+maxDTime
                                sTimeMin=time+staTimeML[staIndex].minTimeS[laIndex,loIndex]-maxDTime
                                sTimeMax=time+staTimeML[staIndex].maxTimeS[laIndex,loIndex]+maxDTime
                                validP=np.where((pTimeL/1e5-pTimeMin/1e5)*(pTimeL/1e5-pTimeMax/1e5)<=0)
                                if len(validP)>0:
                                    if len(validP[0])>0:
                                        if pTimeL[validP[0]][0]<=(time+maxD/0.7+maxDTime):
                                            pTime=pTimeL[validP[0]][0]
                                            pIndex=validP[0][0]
                                            pProb = staL[staIndex].vL[0][pIndex]
                                if pTime < 1:
                                    continue
                                validS=np.where((sTimeL-sTimeMin)*(sTimeL-sTimeMax) < 0)
                                if len(validS)>0:
                                    if len(validS[0])>0:
                                        if sTimeL[validS[0]][0]<=(time+maxD*1.7/0.7+maxDTime):
                                            sTime=sTimeL[validS[0]][0]
                                            sIndex=validS[0][0]
                                            sProb = staL[staIndex].vL[1][sIndex]
                                if pTime >1 and sTime>1:
                                    if np.abs(taupM.get_orign_times(pTime,sTime)-time)>=maxDTime:
                                        continue
                                if pTime > 1:
                                    staL[staIndex].timeL[0][pIndex]=0
                                    if sTime >1:
                                        staL[staIndex].timeL[1][sIndex]=0
                                    quake.Append(Record(staIndex=staIndex, pTime=pTime, sTime=sTime, pProb=pProb, sProb=sProb))
                    if locator != None and len(quake)>=3:
                        try:
                            quake,res=locator.locate(quake)
                            print(quake['time'],quake.loc(),res)
                        except:
                            print('wrong in locate')
                        else:
                            pass
                    quakeL.append(quake)
    return quakeL

def getStaTimeL(staInfos, aMat,taupM=tool.quickTaupModel()):
    #manager=Manager()
    #staTimeML=manager.list()
    staTimeML=list()
    for staInfo in staInfos:
        loc=staInfo.loc()[:2]
        staTimeML.append(staTimeMat(loc, aMat, taupM=taupM))
    return staTimeML

def getSta(staL,i, staInfo, date, modelL, staTimeM, loc, \
        freq,getFileName,taupM, mode,isPre=True,R=[-90,90,\
    -180,180],comp=['BHE','BHN','BHZ'],maxD=80,delta0=0.02,\
    bTime=None,eTime=None):
    staL[i] = sta(staInfo, date, modelL, staTimeM, loc, \
            freq=freq, getFileName=getFileName, taupM=taupM, \
            mode=mode,isPre=isPre,R=R,comp=comp,maxD=maxD,\
            delta0=delta0,bTime=bTime,eTime=eTime)
def preSta(staL,i, staInfo, date, modelL, staTimeM, loc, \
        freq,getFileName,taupM, mode,isPre=True,R=[-90,90,\
    -180,180],comp=['BHE','BHN','BHZ'],maxD=80,delta0=0.02,\
    bTime=None,eTime=None):
    staL[i].predict(staInfo, date, modelL, staTimeM, loc, \
            freq=freq, getFileName=getFileName, taupM=taupM, \
            mode=mode,isPre=isPre,R=R,comp=comp,maxD=maxD,\
            delta0=delta0,bTime=bTime,eTime=eTime)


'''
self, station, day,freq=[-1, -1], \
        taupM=tool.quickTaupModel(),delta0=0.02,\
        R=[-91,91,-181,181],bTime=None,eTime=None
self,modelL=None, staTimeM=None,\
     mode='mid', isClearData=False,maxD=80
     '''
def getStaL(staInfos, staTimeML=[], modelL=[],\
    date=obspy.UTCDateTime(0),taupM=tool.quickTaupModel(),\
     mode='mid',isPre=True,f=[2, 15],R=[-90,90,\
    -380,380],maxD=80,f_new=[-1,-1],delta0=0.02,resampleN=-1,\
    isClearData=False,decPre=1,maxDTime=2):
    staL=[None for i in range(len(staInfos))]
    threads = list()
    for i in range(len(staInfos)):  
        print(ctime(),'process on sta: ',date,i,staInfos[i])
        staL[i]=sta(staInfos[i], date,
            f, taupM,R=R,delta0=delta0)
        staL[i].filt(f_new)
        staL[i].resample(resampleN)
        #print(ctime(),'processed on sta: ',staL[i].data)
    if not isPre:
        return staL
    for i in range(len(staInfos)):
        if len(staTimeML)>0:
            staTimeM=staTimeML[i]
        else:
            staTimeM=None
        print(ctime(),'predict on sta: ',date,i)
        staL[i].predict(modelL, staTimeM, mode,\
            maxD=maxD,maxDTime=maxDTime,isClearData=isClearData,decPre=decPre)
    return staL
def getForQuake(staL,quakes,modelL,**kwargs):
    for quake in quakes:
        for count in range(len(staL)):
            sta = staL[count]
            if len(sta.data)==3:
                pTime,sTime,pProb,sProb = sta.pickQuake(quake,modelL,**kwargs)
                if pTime > 0 :
                    quake.records.append(Record(staIndex=count,pTime=pTime,sTime=sTime,pProb=pProb,sProb=sProb))
                    print(quake['time'],count,pTime,sTime,pProb,sProb)


def getStaQuick(staInfos,date,f,taupM,R,delta0,f_new,resampleN):
    for i in range(len(staInfos)):  
        print(ctime(),'process on sta: ',date,i)
        sta=sta(staInfos[i], date,
            f, taupM,R=R,delta0=delta0)
        sta.filt(f_new)
        sta.resample(resampleN)

from ..io.parRead import StaReader,DataLoader,collate_function
def getStaLQuick(staInfos, staTimeML=[], modelL=[],\
    date=obspy.UTCDateTime(0),taupM=tool.quickTaupModel(),\
     mode='mid',isPre=True,f=[2, 15],R=[-90,90,\
    -180,180],maxD=80,f_new=[-1,-1],delta0=0.02,resampleN=-1,\
    isClearData=False,num_workers=5):
    staReader = StaReader(staInfos,getStaLQuick,date,f,taupM,R,delta0,f_new,resampleN)
    parReader = DataLoader(staReader,batch_size=1,collate_fn=collate_function,num_workers=num_workers)
    staL=[]
    for tmp in parReader:
        for t in tmp:
            staL.append(t)
    if not isPre:
        return staL
    for i in range(len(staInfos)):
        if len(staTimeML)>0:
            staTimeM=staTimeML[i]
        else:
            staTimeM=None
        print(ctime(),'predict on sta: ',date,i)
        staL[i].predict(modelL, staTimeM, mode,\
            maxD=maxD,isClearData=isClearData)
    return staL


def showExample(filenameL,modelL,delta=0.02,t=[]):
    data=getTrace3ByFileName(filenameL,freq=[2,15])
    data=data.Data()[:2000*50]
    
    #i0=int(750/delta)
    #i1=int(870/delta)
    #plt.specgram(np.sign(data[i0:i1,1])*(np.abs(data[i0:i1,1])**0.5),NFFT=200,Fs=50,noverlap=190)
    data/=data.max()/2
    #plt.colorbar()
    #plt.show()
    plt.close()
    plt.figure(figsize=[4,4])
    yL=[predictLongData(modelL[i],data) for i in range(2)]
    timeL=np.arange(data.shape[0])*delta-720
    #print(data.shape,timeL.shape)
    for i in range(3):
        plt.plot(timeL,np.sign(data[:,i])*(np.abs(data[:,i]))+i,'k',linewidth=0.3)
    for i in range(2):
        plt.plot(timeL,yL[i]-i-1.5,'k',linewidth=0.5)
    if len(t)>0:
        plt.xlim(t)
    plt.yticks(np.arange(-2,3),['S','P','E','N','Z'])
    plt.ylim([-2.7,3])
    plt.xlabel('t/s')
    plt.savefig('fig/complexCondition.eps')
    plt.savefig('fig/complexCondition.tiff',dpi=300)
    plt.close()
    

def showExampleV2(filenameL,modelL,delta=0.02,t=[],staName='sta'):
    data=getTrace3ByFileName(filenameL,freq=[2,15],delta=delta)
    data=data.Data()[:3500*50]
    
    #i0=int(750/delta)
    #i1=int(870/delta)
    #plt.specgram(np.sign(data[i0:i1,1])*(np.abs(data[i0:i1,1])**0.5),NFFT=200,Fs=50,noverlap=190)
    data/=data.max()/2
    #plt.colorbar()
    #plt.show()
    plt.close()
    plt.figure(figsize=[4,4])
    yL=[predictLongData(model,data) for model in modelL]
    timeL=np.arange(data.shape[0])*delta-720
    #print(data.shape,timeL.shape)
    for i in range(3):
        plt.plot(timeL,np.sign(data[:,i])*(np.abs(data[:,i]))+i,'k',linewidth=0.3)
    for i in range(len(modelL)):
        plt.plot(timeL,yL[i]-i-1.5,'k',linewidth=0.5)
        #plt.plot(timeL,yL[i]*0+0.5-i-1.5,'--k',linewidth=0.5)
    if len(t)>0:
        plt.xlim(t)
    plt.yticks(np.arange(-4,3),['S1','S0','P1','P0','E','N','Z'])
    plt.ylim([-4.7,3])
    plt.xlabel('t/s')
    plt.savefig('fig/complexConditionV2_%s.eps'%staName)
    plt.savefig('fig/complexConditionV2_%s.tiff'%staName,dpi=300)
    plt.close()

def plotRes(staL, quake, filename=None):
    colorStr='br'
    for record in quake.records:
        color=0
        pTime=record['pTime']
        sTime=record['sTime']
        staIndex=record['staIndex']
        if staIndex>100:
            color=1
        #print(staIndex,pTime,sTime)
        st=quake['time']-10
        et=sTime+40
        if sTime==0:
            et=pTime+60
        pD=(pTime-quake['time'])
        if pTime ==0:
            pD = ((sTime-quake['time'])/1.73)
        if staL[staIndex].data.bTime<0:
            continue
        #print(st, et, staL[staIndex].data.delta)
        timeL=np.arange(st, et, staL[staIndex].data.delta)
        #data = staL[staIndex].data.getDataByTimeL(timeL)
        data=staL[staIndex].data.getDataByTimeLQuick(timeL)
        if timeL.shape[0] != data.shape[0]:
            print('not same length for plot')
            continue
        if timeL.size<1:
            print("no timeL for plot")
            continue
        indexL=np.arange(data.shape[0])
        if pTime>0:
            index0=max(int((pTime-5-st)/staL[staIndex].data.delta),0)
            index1=int((pTime+5-st)/staL[staIndex].data.delta)
            indexL=np.arange(index0,index1)
        #if record.pProb()>1 or record.pProb()<0:
        #    plt.plot(timeL, data[:, 2]/data[indexL,2].max()+pD,colorStr[color],linewidth=0.3)
        #else:
        if True:
            #color = prob2color(record['pProb'])
            if isinstance(quake,QuakeCC):
                color = prob2color(record['pCC'])
                pValue = record['pCC']
                sValue = record['sCC']
            else:
                color = prob2color(record['pProb'])
                pValue = record['pProb']
                sValue = record['sProb']
            plt.plot(timeL, data[:, 2]/data[indexL,2].max()+pD,color=color,linewidth=0.3)
        plt.text(timeL[0],pD+0.5,'%s %.2f %.2f'%(staL[staIndex].station,pValue,\
            sValue))
        if pTime>0:
            plt.plot([pTime, pTime], [pD+2, pD-2], 'g',linewidth=0.5)
        if sTime >0:
            plt.plot([sTime, sTime], [pD+2, pD-2], 'r',linewidth=0.5)
    if isinstance(quake,QuakeCC):
        plt.title('%s %.3f %.3f %.3f %.3f cc:%.3f' % (obspy.UTCDateTime(quake['time']).\
            ctime(), quake['la'], quake['lo'],quake['dep'],quake['ml'],quake['cc']))
    else:
        plt.title('%s %.3f %.3f %.3f %.3f' % (obspy.UTCDateTime(quake['time']).\
            ctime(), quake['la'], quake['lo'],quake['dep'],quake['ml']))
    if filename==None:
        plt.show()
    if filename!=None:
        dayDir=os.path.dirname(filename)
        if not os.path.exists(dayDir):
            os.mkdir(dayDir)
        plt.savefig(filename,dpi=200)
        plt.close()

def plotResS(staL,quakeL, outDir='output/'):
    for quake in quakeL:
        filename=outDir+'/'+quake['filename'][0:-3]+'png'
        #filename=outDir+'/'+str(quake.time)+'.jpg'
        #try:
        plotRes(staL,quake,filename=filename)
        #except:
        #    pass
        #else:
        #    pass

def plotQuakeL(staL,quakeL,laL,loL,outDir='output/',filename='',vModel=None,isPer=False):
    dayIndex = int(quakeL[-1]['time']/86400)
    Ymd = obspy.UTCDateTime(dayIndex*86400).strftime('%Y%m%d')
    if len(filename)==0:
        filename = '%s/%s_quake_loc.jpg'%(outDir,Ymd)
    dayDir=os.path.dirname(filename)
    if not os.path.exists(dayDir):
        os.mkdir(dayDir)
    m = basemap.Basemap(llcrnrlat=laL[0],urcrnrlat=laL[1],llcrnrlon=loL[0],\
        urcrnrlon=loL[1])
    staLa= []
    staLo=[]
    for sta in staL:
        if sta.data.bTime>0:
            staLa.append(sta.loc[0])
            staLo.append(sta.loc[1])
    #staLa,staLo = staL.loc()
    staX,staY=m(np.array(staLo)%360,np.array(staLa))
    m.plot(staX,staY,'b^',markersize=4,alpha=0.2)
    eLa= []
    eLo=[]
    for quake in quakeL:
        eLa.append(quake['la'])
        eLo.append(quake['lo'])
    eX,eY=m(np.array(eLo)%360,np.array(eLa))
    #m.etopo()
    for fault in faultL:
        if fault.inR(laL+loL):
            fault.plot(m,markersize=0.3)
    m.plot(eX,eY,'ro',markersize=0.5)
    parallels = np.arange(-90,90,3)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,3)
    plt.gca().yaxis.set_ticks_position('right')
    m.drawmeridians(meridians,labels=[True,False,False,True])
    plt.title(Ymd)
    plt.savefig(filename,dpi=300)
    plt.close()

def plotQuakeLDis(staInfos,quakeL,laL,loL,outDir='output/',filename='',isTopo=False,rL=[],R0=[]):
    dayIndex = int(quakeL[-1]['time']/86400)
    Ymd = obspy.UTCDateTime(dayIndex*86400).strftime('%Y%m%d')
    if len(filename)==0:
        filename = '%s/%s_quake_loc.jpg'%(outDir,Ymd)
    dayDir=os.path.dirname(filename)
    if not os.path.exists(dayDir):
        os.mkdir(dayDir)
    fig=plt.figure(figsize=[6.2,5])
    m = basemap.Basemap(llcrnrlat=laL[0],urcrnrlat=laL[-1],llcrnrlon=loL[0],\
        urcrnrlon=loL[-1])
    if len(staInfos)>0:
        req={'staInfos':staInfos,'minCover':0.5,'minMl':-5}
    else:
        req={'minMl':-5}
    req={}
    pL=quakeL.paraL(req=req)
    eX,eY,=m(np.array(pL['lo']),np.array(pL['la']))
    #m.etopo()
    ml,dep=[np.array(pL['ml']),np.array(pL['dep'])]
    for fault in faultL:
        if fault.inR(laL+loL):
            fh=fault.plot(m,markersize=0.3)
    
    #m.plot(eX,eY,'ro',markersize=2)
    #m.etopo()
    if isTopo:
        plotTopo(m,laL+loL)
    #sc=m.scatter(eX,eY,c=dep,s=((ml*0+1)**2)*0.3/3,vmin=-5,vmax=50,cmap='gist_rainbow')#Reds
    #sc=m.scatter(eX,eY,c=dep,s=((ml*0+1)**2)*0.3/3,vmin=-5,vmax=50,cmap='jet')
    eh=m.plot(eX,eY,'.r',markersize=0.01,alpha=1,linewidth=0.01)
    staLa= []
    staLo=[]
    for sta in staInfos:
        staLa.append(sta.loc()[0])
        staLo.append(sta.loc()[1])
    #staLa,staLo = staL.loc()
    staX,staY=m(np.array(staLo),np.array(staLa))
    sh=m.plot(staX,staY,'k^',markersize=3,alpha=1)
    #plt.legend([st,sc,f],['station','event','fault'])
    #m.arcgisimage()
    #plt.scatter()
    parallels = np.arange(0.,90,3)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,3)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    plt.gca().yaxis.set_ticks_position('left')
    #plt.title('Detection Results')
    #cbar=fig.colorbar(sc, orientation="horizontal", fraction=0.046, pad=0.04)
    #cbar.set_label('Depth')
    #plt.colorbar()
    #plt.legend((sh,eh,fh),['station','earthquake','fault'],loc=1)
    fig.tight_layout()
    for R in rL:
        x,y=m(R.xyL[:,1],R.xyL[:,0])
        plt.arrow(x[0],y[0],x[1]-x[0],y[1]-y[0],color='b')
        plt.text(x[0],y[0],R.name,ha='left',va='bottom',c='b',size=12,weight='bold')
        plt.text(x[1],y[1],R.name+'\'',ha='right',va='top',c='b',size=12,weight='bold')
    R=R0
    if len(R)>0:
        la = [R[0],R[0],R[1],R[1],R[0]]
        lo = [R[2],R[3],R[3],R[2],R[2]]
        x,y=m(lo,la)
        plt.plot(x,y,color='r',linewidth=1)
    plt.savefig(filename,dpi=300)

    plt.close()

def showStaCover(aMat,staTimeML,filename='cover.jpg'):
    fig=plt.figure(figsize=[5,5])
    aM = np.zeros([aMat.laN,aMat.loN])
    for staTimeM in staTimeML:
        aM[staTimeM.minTimeD<=21]+=1
    laL=[]
    loL=[]
    for a in aMat.subareas[0]:
        loL.append(a.midLo)
    for a in aMat.subareas:
        laL.append(a[0].midLa)
    m = basemap.Basemap(llcrnrlat=laL[0],urcrnrlat=laL[-1],llcrnrlon=loL[0],\
        urcrnrlon=loL[-1])
    aX,aY=m(np.array(loL),np.array(laL))
    setMap(m)
    #m.pcolor(aX,aY,(aM.transpose()>2)*np.log(aM.transpose()+1),cmap='jet')
    pc=m.pcolor(aX,aY,(aM>2)*aM,cmap='jet')
    for fault in faultL:
        if fault.inR(laL+loL):
            fault.plot(m,markersize=0.3)
    cbar=fig.colorbar(pc, orientation="horizontal",fraction=0.046, pad=0.04)
    cbar.set_label('Station Coverage')
    plt.savefig(filename,dpi=300)

def setMap(m):
    
    parallels = np.arange(0.,90,3)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,3)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    
    plt.gca().yaxis.set_ticks_position('left')



'''

def getStaLByQuake(staInfos, aMat, staTimeML, modelL,quake,\
    getFileName=originFileName,taupM=tool.quickTaupModel(), \
    mode='mid', N=5,isPre=False,bTime=-100,delta0=0.02):
    staL=[None for i in range(len(staInfos))]
    threads = list()
    for i in range(len(staInfos)):
        staInfo=staInfos[i]
        nt = staInfo['net']
        st = staInfo['sta']
        loc = [staInfo['la'],staInfo['lo']]
        print('process on sta: ',i)
        dis=DistAz(quake.loc[0],quake.loc[1],staInfos[i]['la'],\
            staInfos[i]['lo']).getDelta()
        date=obspy.UTCDateTime(quake.time+taupM.get_travel_times(quake.loc[2],dis)[0].time+bTime)
        getSta(staL, i, nt, st, date, modelL, staTimeML[i], loc, \
            [0.01, 15], getFileName, taupM, mode,isPre=isPre,delta0=delta0)
    return staL
'''
