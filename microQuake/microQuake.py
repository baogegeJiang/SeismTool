import numpy as np
from obspy import Trace
from ..io import seism 
import os
from ..io.seism import comp3,comp33,cI33

def crossByPair(quake,staInfos,staPairM,matDir='output',f=[-1,-1],filtOrder=2,resDir='',threshold=10):
    T3L=quake.loadSacs(staInfos,matDir=matDir,\
        f=f,filtOrder=filtOrder)
    quake.records.sort()
    quakeName = quake.name('.')
    for i in range(len(quake.records)):
        record0 = quake.records[i]
        staIndex0=record0['staIndex']
        sta0 = staInfos[staIndex0]
        if T3L[i].bTime<=0:
            continue
        az = sta0.az(quake)
        T3L[i] = T3L[i].rotate(az-90)
        T30=T3L[i]
        for t in T30:
            t.data /=t.data.std()
        for j in range(i):
            record1 = quake.records[j]
            if T3L[j].bTime<=0:
                continue
            T31=T3L[j]
            staIndex1=record1['staIndex']
            sta1 = staInfos[staIndex1]
            staPair = staPairM[staIndex0][staIndex1]
            if np.abs((staPair.az-az+threshold)%360)>threshold*2:
                continue
            fileDir=staPair.resDir(resDir)
            filenames= [ fileDir + quakeName+'.'+comp for comp in 'RTZ']
            t3=seism.Trace3([seism.corrTrace(T30[cI],T31[cI])for cI in range(3)])
            if not os.path.exists(os.path.dirname(filenames[0])):
                os.makedirs(os.path.dirname(filenames[0]))
            print(filenames)
            t3.write(filenames)

def crossByPairP(quake,staInfos,staPairM,matDir='output',f=[-1,-1],filtOrder=2,resDir='',threshold=10):
    T3L=quake.loadSacs(staInfos,matDir=matDir,\
        f=f,filtOrder=filtOrder)
    quake.records.sort()
    quakeName = quake.name('.')
    for i in range(len(quake.records)):
        record0 = quake.records[i]
        staIndex0=record0['staIndex']
        sta0 = staInfos[staIndex0]
        if T3L[i].bTime<=0:
            continue
        az = sta0.az(quake)
        T3L[i] = T3L[i].rotate(az-90)
        T30=T3L[i]#.slice(record0['pTime']-6,record0['pTime']+6)
        for t in T30:
            t.data /=t.data.std()
        for j in range(i):
            record1 = quake.records[j]
            if T3L[j].bTime<=0:
                continue
            staIndex1=record1['staIndex']
            sta1 = staInfos[staIndex1]
            staPair = staPairM[staIndex0][staIndex1]
            if np.abs((staPair.az-az+threshold)%360)>threshold*2:
                continue
            T31=T3L[j].slice(record1['pTime']-2,record1['pTime']+4)
            fileDir=staPair.resDir(resDir)
            filenames= [ fileDir + quakeName+'.'+comp for comp in 'RTZ']
            t3=seism.Trace3([seism.corrTrace(T30[cI],T31[cI])for cI in range(3)])
            if not os.path.exists(os.path.dirname(filenames[0])):
                os.makedirs(os.path.dirname(filenames[0]))
            print(filenames)
            t3.write(filenames)


def crossByPairP33(quake,staInfos,staPairM,matDir='output',f=[-1,-1],filtOrder=2,resDir='',threshold=10):
    T3L=quake.loadSacs(staInfos,matDir=matDir,\
        f=f,filtOrder=filtOrder)
    quake.records.sort()
    quakeName = quake.name('.')
    for i in range(len(quake.records)):
        record0 = quake.records[i]
        staIndex0=record0['staIndex']
        sta0 = staInfos[staIndex0]
        if T3L[i].bTime<=0:
            continue
        az = sta0.az(quake)
        dist=sta0.dist(quake)
        T3L[i] = T3L[i].rotate(az-90)
        T30=T3L[i]#.slice(record0['pTime']-6,record0['pTime']+6)
        for t in T30:
            t.data /=t.data.std()
        for j in range(i):
            record1 = quake.records[j]
            if T3L[j].bTime<=0:
                continue
            staIndex1=record1['staIndex']
            sta1 = staInfos[staIndex1]
            staPair = staPairM[staIndex0][staIndex1]
            if dist<staPair.dist+10:
                continue
            if np.abs((staPair.az-az+threshold)%360)>threshold*2:
                continue
            T31=T3L[j].slice(record1['pTime']-2,record1['pTime']+4)
            fileDir=staPair.resDir(resDir)
            filenames= [ fileDir + quakeName+'.'+comp for comp in comp33]
            t3=seism.Trace3([seism.corrTrace(T30[cI[0]],T31[cI[1]])for cI in cI33])
            if not os.path.exists(os.path.dirname(filenames[0])):
                os.makedirs(os.path.dirname(filenames[0]))
            print(filenames)
            t3.write(filenames)


