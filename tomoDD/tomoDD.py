import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from ..mathTool.mathFunc_bak import corrNP as xcorr, Model as Model0,outR
#from ..MFT.cudaFunc import torchcorrnnNorm as xcorr
from ..io.tool import getYmdHMSj
from ..mathTool.distaz import DistAz
from ..mapTool import mapTool as mt
import mpl_toolkits.basemap as basemap
import pycpt
import os
from scipy import interpolate,stats
import matplotlib.colors as colors
from SeismTool.plotTool import figureSet as fs


cmap = pycpt.load.gmtColormap(os.path.dirname(__file__)+'/../data/temperatureInv')
cmapRWB = pycpt.load.gmtColormap(os.path.dirname(__file__)+'/../data/rwb.cpt')
faultL = mt.readFault(os.path.dirname(__file__)+'/../data/Chinafault_fromcjw.dat')
def setMap(m):
    
    parallels = np.arange(0.,90,3)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,3)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    
    plt.gca().yaxis.set_ticks_position('left')

#from cudaFunc import  torchcorrnn as xcorr
SPRatio=0.5
wkdir='TOMODD'
def preEvent(quakeL,staInfos,filename='abc',R=[-90,90,-180,180]):
    if filename=='abc':
        filename='%s/input/event.dat'%wkdir
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            ml=0
            if not quake.inR(R):
                if quake['ml']!=None :
                    if quake['ml']>-2:
                        ml=quake['ml']
                Y=getYmdHMSj(UTCDateTime(quake['time']))
                f.write("%s  %s%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                    (Y['Y']+Y['m']+Y['d'],Y['H']+Y['M']+Y['S'],int(quake['time']*100)%100,\
                        (R[0]+R[1])/2+i/10e5,(R[2]+R[3])/2+i/10e5,max(0,quake['dep']),ml,1,i,0))
                continue
            if quake['ml']!=None :
                if quake['ml']>-2:
                    ml=quake['ml']
            Y=getYmdHMSj(UTCDateTime(quake['time']))
            f.write("%s  %s%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                (Y['Y']+Y['m']+Y['d'],Y['H']+Y['M']+Y['S'],int(quake['time']*100)%100,\
                    quake['la'],quake['lo'],max(0,quake['dep']),ml,1,i,0))

def preABS(quakeL,staInfos,filename='abc',isNick=True,notTomo=False,R=[-90,90,-180,180]):
    if filename=='abc':
        filename='%s/input/ABS.dat'%wkdir
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            if not quake.inR(R):
                continue
            ml=0
            if quake['ml']!=None :
                if quake['ml']>-2:
                    ml=quake['ml']
            if notTomo:
                Y=getYmdHMSj(UTCDateTime(quake.time))
                f.write("%s %s %s  %s %s %s.%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                    (Y['Y'],Y['m'],Y['d'],Y['H'],Y['M'],Y['S'],int(quake['time']*100)%100,\
                    quake['la'],quake['lo'],quake['dep'],ml,1,i,0))
            else:
                f.write('#  % 8d\n'%i)
            for record in quake.records:
                staIndex=record['staIndex']
                staInfo=staInfos[staIndex]
                if not isNick:
                    staName=staInfo['sta']
                else:
                    staName=staInfo['nickName']
                if record['pTime']>0:
                    if not isNick:
                        f.write('%8s    %7.2f   %5.3f   P\n'%\
                            (staName,record['pTime']-quake['time'],1.0))
                    else:
                        f.write('%s     %7.2f   %5.3f   P\n'%\
                            (staName,record['pTime']-quake['time'],1.0))
                if record['sTime']>0:
                    if not isNick:
                        f.write('%8s    %7.2f   %5.3f   S\n'%\
                            (staName,record['sTime']-quake['time'],SPRatio))
                    else:
                        f.write('%s     %7.2f   %5.3f   S\n'%\
                            (staName,record['sTime']-quake['time'],SPRatio))

def preSta(staInfos,filename='abc',isNick=True):
    if filename=='abc':
        filename='%s/input/station.dat'%wkdir
    with open(filename,'w+') as f:
        for staInfo in staInfos:
            if not isNick:
                staName=staInfo['sta']
            else:
                staName=staInfo['nickName']
            f.write('%s %7.4f %8.4f %.0f\n'\
                %(staName,staInfo['la'],staInfo['lo'],staInfo['dep']))

def preDTCC(quakeL,staInfos,dTM,maxD=0.5,minSameSta=5,minPCC=0.75,minSCC=0.75,\
    perCount=500,filename='abc',isNick=True,minDP=3/0.7,R=[-90,90,-180,180]):
    if filename=='abc':
        filename='%s/input/dt.cc'%wkdir
    N=len(quakeL)
    countL=np.zeros(N)
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            print(i)
            quake0 = quakeL[i]
            if not quake0.inR(R):
                continue
            indexL0=quake0.staIndexs()
            time0=quake0['time']
            if countL[i]>perCount:
                continue
            jL=np.arange(i+1,len(quakeL))
            np.random.shuffle(jL) 
            for j in jL:
                quake1=quakeL[j]
                if dTM[i][j]==None:
                    continue
                if countL[j]>perCount:
                    continue
                if not quake1.inR(R):
                    continue
                if quake0.distaz(quake1).getDelta()>maxD:
                    continue
                indexL1=quake1.staIndexs()
                time1=quakeL[j]['time']
                sameIndexL =[]
                for I in indexL0:
                    if I in indexL1:
                        sameIndexL.append(I)
                if len(sameIndexL)<minSameSta:
                    continue                  
                for dtD in dTM[i][j]:
                    dt,maxC,staIndex,phaseType=dtD
                    if maxC>2:
                        continue
                    index0 = indexL0.index(staIndex)
                    index1 = indexL1.index(staIndex)
                    if isNick:
                        staName=staInfos[staIndex]['nickName']
                    else:
                        staName=staInfos[staIndex]['sta']
                    if phaseType==1 and maxC>minPCC:
                        if np.abs(quake0.records[index0]['pTime']-time0)<minDP:
                            continue
                        if np.abs(quake1.records[index1]['pTime']-time1)<minDP:
                            continue
                        dt=quake0.records[index0]['pTime']-time0-(quake1.records[index1]['pTime']-time1+dt)
                        f.write("% 9d % 9d %s %8.3f %6.4f %s\n"%(i,j,\
                            staName,dt,maxC*maxC,'P'))
                        countL[i]+=1
                        countL[j]+=1
                    if phaseType==2 and maxC>minSCC:
                        if np.abs(quake0.records[index0]['sTime']-time0)<minDP*1.7:
                            continue
                        if np.abs(quake1.records[index1]['sTime']-time1)<minDP*1.7:
                            continue
                        dt=quake0.records[index0]['sTime']-time0-(quake1.records[index1]['sTime']-time1+dt)
                        f.write("% 9d % 9d %s %8.3f %6.4f %s\n"%(i,j,\
                            staName,dt,maxC*maxC*SPRatio,'S'))
                        countL[i]+=1
                        countL[j]+=1


def preMod(R,nx=8,ny=8,nz=12,filename='abc'):
    if filename=='abc':
        filename='%s/MOD'%wkdir
    with open(filename,'w+') as f:
        vp=np.array([2   , 3, 4.0, 5,  5.5, 5.6, 5.8,  5.88, 6.1, 6.2,  6.4,  6.45, 7.0,    7.75, 7.76,8.2])
        #vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00,6.1]
        z =         [-150, -5,-2.5, 0, 2.5,   5, 7.5,   10, 12.5,  15,   20,     30,  40,      50,  60, 200]
        vp=np.array([2   , 3,  5,    5.6,5.88,  6.2,    6.4,  6.45, 7.0,    7.75, 7.76,8.2])
        #vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00,6.1]
        z =         [-150, -5, 0,      5,  10,   15,   20,   30,   40,     50,   60,  200]
        vp=np.array([2   , 4,  5,    5.6,5.88,  6.2,    6.4, 6.42, 6.45, 6.8,7.0,    7.75, 7.76,8.2])
        #vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00,6.1]
        z =         [-150, -5, 0,      5,  10,   15,      20,  25,   30,  35, 40,     50,   60,  200]
        vp=np.array([2   , 4,  5,    5.88,  6.2,    6.4,  6.45,  7.0,    7.75, 7.76,8.2])
        #vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00,6.1]
        z =         [-150, -5, 2.5,    7.5,    12,      20,   30,   40,     50,   60,  200]
        vs=vp/1.71
        x=np.zeros(nx)
        y=np.zeros(ny)
        #z=[-150,-2, 0, 5,10, 15,25, 35, 50, 60,80, 500]
        #x lo nx loN
        #y la ny laN
        f.write('0.1 %d %d %d\n'%(nx,ny,nz))
        x[0]=R[2]-30
        x[-1]=R[3]+60
        y[0]=R[0]-30
        y[-1]=R[1]+20
        x[1]=(x[0]+R[2])/2
        x[-2]=(x[-1]+R[3])/2
        y[1]=(y[0]+R[0])/2
        y[-2]=(y[-1]+R[1])/2
        x[2:-2]=np.arange(R[2],R[3]+0.001,(R[3]-R[2])/(nx-5))
        y[2:-2]=np.arange(R[0],R[1]+0.001,(R[1]-R[0])/(ny-5))
        #f.write("\n")
        for i in range(nx):
            f.write('%.4f '%x[i])
        f.write('\n')
        for i in range(ny):
            f.write('%.4f '%y[i])
        f.write('\n')
        for i in range(nz):
            f.write('%.4f '%z[i])
        f.write('\n')

        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    f.write('%.2f '%vp[i])
                f.write('\n')

        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    f.write('%.2f '%(vp[i]/vs[i]))
                f.write('\n')

def sameSta(timeL1,timeL2):
    return np.where(np.sign(timeL1*timeL2)>0)[0]

def calDT(quake0,quake1,T3PSL0,T3PSL1,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minC=0.6,maxD=0.2,minSameSta=4):
    '''
    dT=[[dt,maxCC,staIndex,phaseType],....]
    dT is a list containing the dt times between quake0 and quake1
    dt is the travel time difference
    maxCC is the peak value of the normalized cross-correlation 
    staIndex is the index of the station
    phaseType: 1 for P; 2 for S
    '''
    indexL0 = quake0.staIndexs()
    indexL1 = quake1.staIndexs()
    sameIndex=[]
    if quake0.dist(quake1)/110>maxD:
        return None
    for I in indexL0:
        if I in indexL1:
            sameIndex.append(I)
    if len(sameIndex)<minSameSta:
        return None
    dT=[];
    #print(sameIndex)
    for staIndex in sameIndex:
        index0 = indexL0.index(staIndex)
        index1 = indexL1.index(staIndex)
        record0 = quake0.records[index0]
        record1 = quake1.records[index1]
        T3P0,T3S0=  [T3PSL0[0][index0],T3PSL0[1][index0]]
        T3P1,T3S1=  [T3PSL1[0][index1],T3PSL1[1][index1]]
        if record0['pTime']>0 and record1['pTime']>0:
            #pWave0=T3P0.Data(record0['pTime']+bSec0,record0['pTime']+eSec0)
            #pWave1=T3P1.Data(record1['pTime']+bSec1,record1['pTime']+eSec1)
            pWave0=T3P0.Data0#(record0['pTime']+bSec0,record0['pTime']+eSec0)
            pWave1=T3P1.Data1#(record1['pTime']+bSec1,record1['pTime']+eSec1)
            if len(pWave0)*len(pWave1)==0:
                continue
            #print(pWave0.shape,pWave1.shape)
            c=xcorr(pWave1,pWave0)
            #print(c)
            c=xcorr(pWave1,pWave0)[0].max(axis=1)#########
            maxC=c.max()
            if maxC>minC:
                maxIndex=c.argmax()
                dt=bSec1+maxIndex*T3P0.delta-bSec0
                dT.append([dt,maxC,staIndex,1])
        if record0['sTime']>0 and record1['sTime']>0:
            #sWave0=T3S0.Data(record0['sTime']+bSec0,record0['sTime']+eSec0)
            #sWave1=T3S1.Data(record1['sTime']+bSec1,record1['sTime']+eSec1)
            sWave0=T3S0.Data0#(record0['sTime']+bSec0,record0['sTime']+eSec0)
            sWave1=T3S1.Data1#(record1['sTime']+bSec1,record1['sTime']+eSec1)
            if len(sWave0)*len(sWave1)==0:
                continue
            c=xcorr(sWave1,sWave0)[0].max(axis=1)#########
            maxC=c.max()
            if maxC>minC:
                maxIndex=c.argmax()
                dt=bSec1+maxIndex*T3S0.delta-bSec0
                dT.append([dt,maxC,staIndex,2])
    return dT

def calDTM(quakeL,T3PSLL,staInfos,maxD=0.3,minC=0.6,minSameSta=3,\
    isFloat=False,bSec0=-2,eSec0=3,bSec1=-3,eSec1=4):
    '''
    dTM is 2-D list contianing the dT infos between each two quakes
    dTM[i][j] : dT in between quakeL[i] and quakeL[j]
    quakeL's waveform is contained by waveformL
    '''
    dTM=[[None for quake in quakeL]for quake in quakeL]
    #indexL0 = quake0.staIndexs()
    for i in range(len(quakeL)):
        print('cut waveform: ',i)
        quake = quakeL[i]
        T3PSL = T3PSLL[i]
        for index in  range(len(quake.records)):
            record = quake.records[index]
            T3P,T3S=  [T3PSL[0][index],T3PSL[1][index]]
            #record0 = quake0.records[index0]
            #T3P0,T3S0=  [T3PSL0[0][index0],T3PSL0[1][index0]]
            T3P.Data0=T3P.Data(record['pTime']+bSec0,record['pTime']+eSec0)
            T3P.Data1=T3P.Data(record['pTime']+bSec1,record['pTime']+eSec1)
            T3S.Data0=T3S.Data(record['sTime']+bSec0,record['sTime']+eSec0)
            T3S.Data1=T3S.Data(record['sTime']+bSec1,record['sTime']+eSec1)
    for i in range(len(quakeL)):
        #print(i)
        for j in range(i+1,len(quakeL)):
            if j%50==0:
                print(i,j)
            dTM[i][j]=calDT(quakeL[i],quakeL[j],T3PSLL[i],T3PSLL[j],\
                staInfos,maxD=maxD,minC=minC,minSameSta=minSameSta,\
                bSec0=bSec0,eSec0=eSec0,bSec1=bSec1,eSec1=eSec1)
    return dTM

from ..io.parRead import TomoCal,DataLoader,collate_function

def calDTMQuick(quakeL,T3PSLL,staInfos,maxD=0.3,minC=0.6,minSameSta=3,\
    isFloat=False,bSec0=-2,eSec0=3,bSec1=-3,eSec1=4,num_workers=5,doCut=True):
    '''
    dTM is 2-D list contianing the dT infos between each two quakes
    dTM[i][j] : dT in between quakeL[i] and quakeL[j]
    quakeL's waveform is contained by waveformL
    '''
    #dTM=[[None for quake in quakeL]for quake in quakeL]
    #indexL0 = quake0.staIndexs()
    if doCut:
        for i in range(len(quakeL)):
            print('cut waveform: ',i)
            quake = quakeL[i]
            T3PSL = T3PSLL[i]
            for index in range(len(quake.records)):
                record = quake.records[index]
                T3P,T3S=  [T3PSL[0][index],T3PSL[1][index]]
                #record0 = quake0.records[index0]
                #T3P0,T3S0=  [T3PSL0[0][index0],T3PSL0[1][index0]]
                T3P.Data0=T3P.Data(record['pTime']+bSec0,record['pTime']+eSec0)
                T3P.Data1=T3P.Data(record['pTime']+bSec1,record['pTime']+eSec1)
                T3S.Data0=T3S.Data(record['sTime']+bSec0,record['sTime']+eSec0)
                T3S.Data1=T3S.Data(record['sTime']+bSec1,record['sTime']+eSec1)
    '''
        staReader = StaReader(staInfos,getStaLQuick,date,f,taupM,R,delta0,f_new,resampleN)
        parReader = DataLoader(staReader,batch_size=1,collate_fn=collate_function,num_workers=num_workers)
    '''
    tomoCal = TomoCal(quakeL,T3PSLL,calDT,staInfos,maxD=maxD,\
        minC=minC,minSameSta=minSameSta,bSec0=bSec0,eSec0=eSec0,bSec1=bSec1,eSec1=eSec1)
    parCal = DataLoader(tomoCal,batch_size=10,collate_fn=collate_function,num_workers=num_workers)
    dTM=[]
    for tmp in parCal:
        for t in tmp:
            dTM.append(t)
    return dTM

def plotDT(T3PSLL,dTM,i,j,quake0,quake1,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minSameSta=5):
    plt.close()
    T3PL0=T3PSLL[i][0]
    T3PL1=T3PSLL[j][0]
    T3SL0=T3PSLL[i][1]
    T3SL1=T3PSLL[j][1]
    
    timeL0=np.arange(bSec0,eSec0,delta)
    timeL1=np.arange(bSec1,eSec1,delta)
    count=0
    indexL0 = quake0.staIndexs()
    indexL1 = quake1.staIndexs()
    for dT in dTM[i][j]:
        staIndex=dT[2]
        index0 = indexL0.index(staIndex)
        index1 = indexL0.index(staIndex)
        record0 = quake0.records[index0]
        record1 = quake1.records[index1]
        T3P0,T3S0=  [T3PL0[index0],T3SL0[index0]]
        T3P1,T3S1=  [T3PL1[index1],T3SL1[index1]]
        if dT[3]==1:
            w0=T3P0.getPTimeL(timeL0)
            w1=T3P1.getPTimeL(timeL1)
        else:
            w0=T3P0.getPTimeL(timeL0)
            w1=T3P1.getPTimeL(timeL1)
        if len(w0)*len(w1)==0:
                continue
        plt.plot(timeL0+dT[0],w0[:,2]/(w0[:,2].max())*0.5+count,'r')
        print(xcorr(w1,w0).max())
        plt.plot(timeL1,w1[:,2]/(w1[:,2].max())*0.5+count,'b')
        plt.plot(timeL0-dT[0],w0[:,2]/(w0[:,2].max())*0.5+count+2,'r')
        #print(w0.max())
        plt.plot(timeL1,w1[:,2]/(w1[:,2].max())*0.5+count+2,'b')
        #plt.plot(+count,'g')
        #print((w1/w1.max()).shape)
        plt.text(timeL1[0],count+0.5,'cc=%.2f dt=%.2f '%(dT[1],dT[0]))
        count+=1
        plt.savefig('fig/TOMODD/%d_%d_%d_dT.png'%(i,j,staIndex),dpi=300)
        plt.close()

def saveDTM(dTM,filename):
    N=len(dTM)
    with open(filename,'w+') as f:
        f.write("# %d\n"%N)
        for i in range(N):
            for j in range(N):
                if dTM[i][j]==None:
                    continue
                f.write("i %d %d\n"%(i,j))
                for dt in dTM[i][j]:
                    f.write("%f %f %d %d\n"%(dt[0],dt[1],dt[2],dt[3]))
def loadDTM(filename='dTM'):
    with open(filename) as f:
        for line in f.readlines():
            if line.split()[0]=='#':
                N=int(line.split()[1])
                dTM=[[None for i in range(N)]for i in range(N)]
                continue
            if line.split()[0]=='i':
                i=int(line.split()[1])
                j=int(line.split()[2])
                dTM[i][j]=[]
                continue
            staIndex=int(line.split()[2])
            dTM[i][j].append([float(line.split()[0]),float(line.split()[1]),\
            int(line.split()[2]),int(line.split()[3])])
    return dTM

def reportDTM(dTM):
    plt.close()
    N=len(dTM)
    sumN=np.zeros(N)
    quakeN=np.zeros(N)
    dTL=[]


    for i in range(N):
        for j in range(N):
            if dTM[i][j]!=None:
                if len(dTM)<=0:
                    continue
                sumN[i]+=len(dTM[i][j])
                sumN[j]+=len(dTM[i][j])
                quakeN[i]+=1
                quakeN[j]+=1
                for dT in dTM[i][j]:
                    dTL.append(dT[0])
    plt.subplot(3,1,1)
    plt.plot(sumN)
    plt.subplot(3,1,2)
    plt.plot(quakeN)
    plt.subplot(3,1,3)
    plt.hist(np.array(dTL))
    plt.savefig('fig/TOMODD/dTM_report.png',dpi=300)
    plt.close()

def analyDTM(dTM,resFile):
    #dt,maxC,staIndex,2
    plt.close()
    fig=plt.figure(figsize=[4,3])
    pL=[]
    pCCL=[]
    sL=[]
    sCCL=[]
    bins=[np.arange(-1,1,0.02),np.arange(0.6,1,0.03),]
    for event in dTM:
        for pair in event:
            if isinstance(pair,type(None)):
                continue
            for phase in pair:
                dt,maxC,staIndex,phaseType = phase
                if phaseType==1:
                    pL.append(dt)
                    pCCL.append(maxC)
                if phaseType==2:
                    sL.append(dt)
                    sCCL.append(maxC)
    plt.subplot(2,1,1)
    #plt.hist2d(pL,pCCL,bins)
    h,x,y=np.histogram2d(pL,pCCL,bins)
    h=h.transpose()
    h/=h.sum(axis=1,keepdims=True)
    plt.pcolor(x[:-1]*0.5+x[1:]*0.5,y[:-1]*0.5+y[1:]*0.5,h,norm=colors.LogNorm(vmin=h[h!=0].min()*2, vmax=h.max()))
    cbar=plt.colorbar()
    cbar.set_label('density')
    plt.xlabel('dTime/s')
    plt.ylabel('cc (P phase)')
    fs.setABC('(a)')
    plt.subplot(2,1,2)
    #plt.hist2d(sL,sCCL,bins)
    h,x,y=np.histogram2d(sL,sCCL,bins)
    h=h.transpose()
    h/=h.sum(axis=1,keepdims=True)
    plt.pcolor(x[:-1]*0.5+x[1:]*0.5,y[:-1]*0.5+y[1:]*0.5,h, norm=colors.LogNorm(vmin=h[h!=0].min()*2, vmax=h.max()))
    #plt.colorbar()
    cbar=plt.colorbar()
    cbar.set_label('density')
    plt.xlabel('dTime/s')
    plt.ylabel('cc (S phase)')
    fig.tight_layout()
    fs.setABC('(b)')
    plt.savefig(resFile,dpi=300)

def getReloc(quakeL,filename='abc'):
    if filename=='abc':
        filename='%s/tomoDD.reloc'%wkdir
    quakeRelocL=[]
    with open(filename) as f:
        for line in f.readlines():
            line=line.split()
            time=quakeL[0].tomoTime(line)
            index=int(line[0])
            print(quakeL[index]['time']-time)
            quakeRelocL.append(quakeL[index])
            quakeRelocL[-1].getReloc(line)
    return quakeRelocL

def getReloc(qL,filename):
    qLNew=[]
    with open(filename) as f:
        for line in f.readlines():
            tmp = line.split()
            index=int(tmp[0])
            quake = qL[index]#.copy()
            quake['la'] = float(tmp[1])
            quake['lo'] = float(tmp[2])
            quake['dep'] = float(tmp[3])
            timeL = [int(float(t)) for t in tmp[10:16]]
            sec  = float(tmp[15])%1
            if timeL[-1]>=60:
                timeL[-1]%=60
                sec+=60
            quake['time'] = (UTCDateTime(*timeL)+sec).timestamp
            qLNew.append(quake)
    return qLNew

def  analyReloc(filename,resFile):
    data = np.loadtxt(filename)
    erroL=np.arange(0,250,2)
    erroLa = data[:,7]
    erroLo = data[:,8]
    erroDep = data[:,9]
    plt.close()
    plt.figure(figsize=(3,3))
    plt.hist(erroLa,erroL,alpha=0.7)
    plt.hist(erroLo,erroL,alpha=0.7)
    plt.hist(erroDep,erroL,alpha=0.7)
    plt.legend(['latitude','longitude','depth'])
    plt.xlabel('uncertainty(m)')
    plt.ylabel('number')
    plt.savefig(resFile,dpi=300)
    plt.close()



class Model(Model0):
    def __init__(self,laN,loN,zN,laL,loL,zL,v,mode='v',quakeL=[],quakeL0=[],R=[-90,90,-180,180],vR=''):
        self.nxyz=[laN,loN,zN]
        self.laN =laN
        self.loN =loN
        self.zN  =zN
        self.la =laL
        self.lo =loL
        self.z  =zL
        self.v=v
        self.mode=mode
        self.quakeL=quakeL
        self.quakeL0=quakeL0
        self.R = R
        if len(vR)!=0:
            out  = outR(vR,self.la,self.lo)
            print(out)
            self.v[out]=np.nan
    def plot(self,resDir,doDense='True'):
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        for i in range(self.zN):
            name0='v/(km/s)'
            plt.close()
            z= self.z[i]
            if i <self.zN-1:
                z1 = self.z[i+1]
            else:
                z1 = self.z[i]+100
            req = {'minDep':z,'maxDep':z1}
            fig=plt.figure()
            m = basemap.Basemap(llcrnrlat=self.la[2],urcrnrlat=self.la[-3],llcrnrlon=self.lo[2],\
            urcrnrlon=self.lo[-3])
            la,lo,v=self.denseLaLoGrid(self.v[:,:,i].copy(),dIndex=2,doDense=doDense,N=500)
            #la,lo,v = [self.la,self.lo,self.v[:,:,i]] 
            if isinstance(v,type(None)):
                plt.close()
                print('no enough data in depth:',z)
                continue
            #x,y=m(*np.meshgrid(lo,la))
            x,y=m(lo,la)
            if self.mode in ['dVp','dVs','dVpr','dVsr']:
                cmapRWB.set_bad('#A9A9A9', 0)
                m.pcolormesh(x,y,v,vmin=-0.05,vmax=0.05,cmap=cmapRWB,shading='flat')
                name0='dv/v0'
            else:
                cmap.set_bad('#A9A9A9', 0)
                m.pcolormesh(x,y,v,cmap=cmap,shading='flat')
            R =self.R
            plt.gca().set(facecolor='#A9A9A9')
            for fault in faultL:
                if fault.inR(R):
                    fault.plot(m,markersize=0.3)
            if len(self.quakeL0)>0:
                pL=self.quakeL0.paraL(req =req)
                eX,eY,=m(np.array(pL['lo']),np.array(pL['la']))
                #ml,dep=[np.array(pL['ml']),np.array(pL['dep'])]
                plt.plot(eX,eY,'.k',markersize=0.3)
            if len(self.quakeL)>0:
                pL=self.quakeL.paraL(req =req)
                eX,eY,=m(np.array(pL['lo']),np.array(pL['la']))
                #ml,dep=[np.array(pL['ml']),np.array(pL['dep'])]
                plt.plot(eX,eY,'.r',markersize=0.3)
            setMap(m)
            cbar=plt.colorbar()
            cbar.set_label(name0)
            plt.title('%s depth: %.1f km'%(self.mode,z))
            fig.tight_layout()
            plt.savefig('%s/%s_%.1f.jpg'%(resDir,self.mode,z),dpi=300)

class model:
    def __init__(self,workDir,quakeL=[],quakeL0=[],isSyn=False,isDWS=False,minDWS=-1,R=[-90,90,-180,180],vR=''):
        initFile=workDir+'/MOD'
        with open(initFile) as f:
            loN,laN,zN=[int(tmp)for tmp in f.readline().split()[1:]]
            laL,loL,zL=[np.array([float(tmp)for tmp in f.readline().split()])for i in range(3)]
            vp0=np.array([[float(tmp)for tmp in f.readline().split()]for i in range(laN*zN)]\
                ).reshape(zN,laN,loN).transpose([1,2,0])
            vs0=np.array([[float(tmp)for tmp in f.readline().split()]for i in range(laN*zN)]\
                ).reshape(zN,laN,loN).transpose([1,2,0])
            vs0 = vp0/vs0
        vp = np.loadtxt(workDir+'/Vp_model.dat').reshape(zN,laN,loN).transpose([1,2,0])
        vs = np.loadtxt(workDir+'/Vs_model.dat').reshape(zN,laN,loN).transpose([1,2,0])
        self.vp0=Model(loN,laN,zN,loL,laL,zL,vp0,'vp0',quakeL=quakeL,quakeL0=quakeL0,R=R)
        self.vs0=Model(loN,laN,zN,loL,laL,zL,vs0,'vs0',quakeL=quakeL,quakeL0=quakeL0,R=R)
        self.vp =Model(loN,laN,zN,loL,laL,zL,vp ,'vp',quakeL=quakeL,quakeL0=quakeL0,R=R,vR=vR)
        self.vs =Model(loN,laN,zN,loL,laL,zL,vs ,'vs',quakeL=quakeL,quakeL0=quakeL0,R=R,vR=vR)
        self.modelL = [self.vp0,self.vs0,self.vp,self.vs]
        if isDWS:
            DWSFile = workDir + 'tomoDD.vel'
            with open(DWSFile) as f:
                lines = f.readlines()
            lN = len(lines)
            #print(lN)
            for i in range(lN-1,0,-1):
                if lines[i].split()[-1] == 'P-wave.':
                    index=i+1
                    pDWS=np.array([[float(tmp)for tmp in line.split()]for line in lines[index:index+laN*zN]]\
                    ).reshape(zN,laN,loN).transpose([1,2,0])
                    break
            for i in range(lN-1,0,-1):
                if lines[i].split()[-1] == 'S-wave.':
                    index=i+1
                    sDWS=np.array([[float(tmp)for tmp in line.split()]for line in lines[index:index+laN*zN]]\
                    ).reshape(zN,laN,loN).transpose([1,2,0])
                    break
            self.pDWS=Model(loN,laN,zN,loL,laL,zL,pDWS,'pDWS',quakeL=quakeL,quakeL0=quakeL0,R=R)
            self.sDWS=Model(loN,laN,zN,loL,laL,zL,sDWS,'sDWS',quakeL=quakeL,quakeL0=quakeL0,R=R)
            self.vp.v[self.pDWS.v<minDWS]=np.nan
            self.vs.v[self.sDWS.v<minDWS]=np.nan
            self.modelL+=[self.pDWS,self.sDWS]
        if isSyn:
            realFile = workDir + '../Syn/MOD'
            with open(realFile) as f:
                loN,laN,zN=[int(tmp)for tmp in f.readline().split()[1:]]
                laL,loL,zL=[np.array([float(tmp)for tmp in f.readline().split()])for i in range(3)]
                vpr=np.array([[float(tmp)for tmp in f.readline().split()]for i in range(laN*zN)]\
                    ).reshape(zN,laN,loN).transpose([1,2,0])
                vsr=np.array([[float(tmp)for tmp in f.readline().split()]for i in range(laN*zN)]\
                    ).reshape(zN,laN,loN).transpose([1,2,0])
                vsr = vpr/vsr
            self.vpr=Model(loN,laN,zN,loL,laL,zL,vpr,'vpReal',quakeL=quakeL,quakeL0=quakeL0,R=R)
            self.vsr=Model(loN,laN,zN,loL,laL,zL,vsr,'vsReal',quakeL=quakeL,quakeL0=quakeL0,R=R)
            dVp = self.vp.v/self.vp0.v-1
            dVs = self.vs.v/self.vs0.v-1
            self.dVp=Model(loN,laN,zN,loL,laL,zL,dVp,'dVp',quakeL=quakeL,quakeL0=quakeL0,R=R)
            self.dVs=Model(loN,laN,zN,loL,laL,zL,dVs,'dVs',quakeL=quakeL,quakeL0=quakeL0,R=R)
            dVpr = self.vpr.v/self.vp0.v-1
            dVsr = self.vsr.v/self.vs0.v-1
            self.dVpr=Model(loN,laN,zN,loL,laL,zL,dVpr,'dVpr',quakeL=quakeL,quakeL0=quakeL0,R=R)
            self.dVsr=Model(loN,laN,zN,loL,laL,zL,dVsr,'dVsr',quakeL=quakeL,quakeL0=quakeL0,R=R)
            self.modelL+=[self.vpr,self.vsr,self.dVp,self.dVs,self.dVpr,self.dVsr]

    def plot(self,resDir,nameL=[],doDense=True):
        for model in self.modelL:
            if len(nameL)!=0 and model.mode not in nameL:
                continue
            model.plot(resDir,doDense=doDense)
def denseLaLo(La,Lo,Per,N=200):
    dLa = (La[-1]-La[0])/N
    dLo = (Lo[-1]-Lo[0])/N
    la  = np.arange(La[0],La[-1],dLa)
    la.sort()
    lo  = np.arange(Lo[0],Lo[-1],dLo)
    per = interpolate.interp2d(Lo, La, Per,kind= 'linear')(lo,la)
    return la, lo, per

def diff(quakeL0,quakeL1,filename):
    filenameL =  quakeL0.paraL(keyL=['filename'])['filename']
    dkm =[]
    dz  =[]
    for quake1 in quakeL1:
        index0 = filenameL.index(quake1['filename'])
        quake0 = quakeL0[index0]
        dkm.append(quake0.dist(quake1))
        dz.append(np.abs(quake0['dep']-quake1['dep']))
    plt.close()
    plt.figure(figsize=(3,3))
    plt.plot(dkm,dz,'.',markersize=0.1)
    plt.xlabel('horizontal difference/km')
    plt.xlim([0,25])
    plt.gca().set_aspect(1)
    plt.ylabel('vertical difference/km')
    plt.savefig(filename,dpi=300)


'''
def calDT(quake0,quake1,waveform0,waveform1,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minC=0.6,maxD=0.3,minSameSta=5):
   
    #dT=[[dt,maxCC,staIndex,phaseType],....]
    #dT is a list containing the dt times between quake0 and quake1
    #dt is the travel time difference
    #maxCC is the peak value of the normalized cross-correlation 
    #staIndex is the index of the station
    #phaseType: 1 for P; 2 for S

    pTime0=quake0.getPTimeL(staInfos)
    sTime0=quake0.getSTimeL(staInfos)
    pTime1=quake1.getPTimeL(staInfos)
    sTime1=quake1.getSTimeL(staInfos)
    sameIndex=sameSta(pTime0,pTime1)
    if len(sameIndex)<minSameSta:
        return None
    if DistAz(quake0.loc[0],quake0.loc[1],quake1.loc[0],quake1.loc[1]).getDelta()>maxD:
        return None
    dT=[];
    timeL0=np.arange(bSec0,eSec0,delta)
    indexL0=(timeL0/delta).astype(np.int64)-waveform0['indexL'][0][0]
    timeL1=np.arange(bSec1,eSec1,delta)
    indexL1=(timeL1/delta).astype(np.int64)-waveform1['indexL'][0][0]
    for staIndex in sameIndex:
        if pTime0[staIndex]!=0 and pTime1[staIndex]!=0:
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            pWave0=waveform0['pWaveform'][index0,indexL0,2]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            #print(index1)
            pWave1=waveform1['pWaveform'][index1,indexL1,2]
            c=xcorr(pWave1,pWave0)#########
            maxC=c.max()
            if maxC>minC:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC,staIndex,1])
        if sTime0[staIndex]!=0 and sTime1[staIndex]!=0:
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            sWave0=waveform0['sWaveform'][index0,indexL0,0]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            sWave1=waveform1['sWaveform'][index1,indexL1,0]
            c=xcorr(sWave1,sWave0)##########
            maxC0=c.max()
            if maxC0>minC:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC0,staIndex,2])
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            sWave0=waveform0['sWaveform'][index0,indexL0,1]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            sWave1=waveform1['sWaveform'][index1,indexL1,1]
            c=xcorr(sWave1,sWave0)##########
            maxC1=c.max()
            if maxC1>minC and maxC1>maxC0:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC1,staIndex,2])
    return dT
'''