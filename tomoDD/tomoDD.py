import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from ..mathTool.mathFunc_bak import corrNP as xcorr
#from ..MFT.cudaFunc import torchcorrnnNorm as xcorr
from ..io.tool import getYmdHMSj
from ..mathTool.distaz import DistAz
from ..mapTool import mapTool as mt
import mpl_toolkits.basemap as basemap
import pycpt
import os
from scipy import interpolate,stats

cmap = pycpt.load.gmtColormap(os.path.dirname(__file__)+'/../data/temperatureInv')
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
def preEvent(quakeL,staInfos,filename='abc'):
    if filename=='abc':
        filename='%s/input/event.dat'%wkdir
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            ml=0
            if quake['ml']!=None :
                if quake['ml']>-2:
                    ml=quake['ml']
            Y=getYmdHMSj(UTCDateTime(quake['time']))
            f.write("%s  %s%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                (Y['Y']+Y['m']+Y['d'],Y['H']+Y['M']+Y['S'],int(quake['time']*100)%100,\
                    quake['la'],quake['lo'],quake['dep'],ml,1,i,0))

def preABS(quakeL,staInfos,filename='abc',isNick=True,notTomo=False):
    if filename=='abc':
        filename='%s/input/ABS.dat'%wkdir
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
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
    perCount=500,filename='abc',isNick=True,minDP=3/0.7):
    if filename=='abc':
        filename='%s/input/dt.cc'%wkdir
    N=len(quakeL)
    countL=np.zeros(N)
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            print(i)
            quake0 = quakeL[i]
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
        vp=np.array([4   ,5.0,5.5,5.8, 5.91, 6.1,  6.3, 6.50, 6.7,  8.0, 8.6, 9.0])
        #vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00,6.1]
        z =[-150, -2,  0,  5,   10,  15,   20,   30,  40,   50,  60, 200]
        vs=vp/1.71
        x=np.zeros(nx)
        y=np.zeros(ny)
        #z=[-150,-2, 0, 5,10, 15,25, 35, 50, 60,80, 500]
        #x lo nx loN
        #y la ny laN
        f.write('0.1 %d %d %d\n'%(nx,ny,nz))
        x[0]=R[2]-5
        x[-1]=R[3]+5
        y[0]=R[0]-5
        y[-1]=R[1]+5
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
        for index in []:# range(len(quake.records)):
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
            if timeL[-1]>=60:
                timeL[-1]%=60
                timeL[-2]+=1
            quake['time'] = (UTCDateTime(*timeL)+float(tmp[15])%1).timestamp
            qLNew.append(quake)
    return qLNew

class Model:
    def __init__(self,laN,loN,zN,laL,loL,zL,v,mode='v'):
        self.laN =laN
        self.loN =loN
        self.zN  =zN
        self.laL =laL
        self.loL =loL
        self.zL  =zL
        self.v=v
        self.mode=mode
    def plot(self,resDir):
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        for i in range(self.zN):
            plt.close()
            z= self.zL[i]
            m = basemap.Basemap(llcrnrlat=self.laL[2],urcrnrlat=self.laL[-3],llcrnrlon=self.loL[2],\
            urcrnrlon=self.loL[-3])
            la,lo,v=denseLaLo(self.laL[2:-2],self.loL[2:-2],self.v[2:-2,2:-2,i])
            x,y=m(lo,la)
            m.pcolor(x,y,v,cmap=cmap)
            R =[x[0],x[-1],y[0],y[-1]]
            for fault in faultL:
                if fault.inR(R):
                    fault.plot(m,markersize=0.3)
            plt.title('%s %d'%(self.mode,z))
            setMap(m)
            plt.colorbar()
            plt.savefig('%s/%s_%d.jpg'%(resDir,self.mode,z),dpi=300)

class model:
    def __init__(self,workDir):
        initFile=workDir+'/../inversion/MOD'
        with open(initFile) as f:
            loN,laN,zN=[int(tmp)for tmp in f.readline().split()[1:]]
            laL,loL,zL=[np.array([float(tmp)for tmp in f.readline().split()])for i in range(3)]
            vp0=np.array([[float(tmp)for tmp in f.readline().split()]for i in range(laN*zN)]\
                ).reshape(zN,laN,loN).transpose([1,2,0])
            vs0=np.array([[float(tmp)for tmp in f.readline().split()]for i in range(laN*zN)]\
                ).reshape(zN,laN,loN).transpose([1,2,0])
        vp = np.loadtxt(workDir+'/../inversion/Vp_model.dat').reshape(zN,laN,loN).transpose([1,2,0])
        vs = np.loadtxt(workDir+'/../inversion/Vs_model.dat').reshape(zN,laN,loN).transpose([1,2,0])
        self.vp0=Model(loN,laN,zN,loL,laL,zL,vp0,'vp0')
        self.vs0=Model(loN,laN,zN,loL,laL,zL,vs0,'vs0')
        self.vp =Model(loN,laN,zN,loL,laL,zL,vp ,'vp')
        self.vs =Model(loN,laN,zN,loL,laL,zL,vs ,'vs')
    def plot(self,resDir):
        for model in [self.vp0,self.vs0,self.vp,self.vs]:
            model.plot(resDir)

def denseLaLo(La,Lo,Per,N=200):
    dLa = (La[-1]-La[0])/N
    dLo = (Lo[-1]-Lo[0])/N
    la  = np.arange(La[0],La[-1],dLa)
    la.sort()
    lo  = np.arange(Lo[0],Lo[-1],dLo)
    per = interpolate.interp2d(Lo, La, Per)(lo,la)
    return la, lo, per
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