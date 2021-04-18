import os

from tensorflow.python.keras.backend import dtype
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from SeismTool.io import seism
from SeismTool.mapTool import mapTool
from SeismTool.mathTool import mathFunc_bak
from SeismTool.tomoDD import tomoDD
from SeismTool.detector import detecQuake
from numpy import random
from obspy import UTCDateTime,read
from glob import glob
from SeismTool.plotTool import figureSet as fs
import sys
from matplotlib import pyplot as plt
import matplotlib.font_manager
from scipy import signal
templateFile = 'phase/SCYNTomoReloc'
f_0 = [0.5,20]
f_1 = [2,10]
staLstFileL=['/home/jiangyr/Surface-Wave-Dispersion/stations/XU_sel.sta','/home/jiangyr/Surface-Wave-Dispersion/stations/SCYN_withComp_ac',]
templateDir = '/media/jiangyr/MSSD/output/'
ccDir       = '/HOME/jiangyr/detecQuake/output/outputV20210106CC8_2-10/'
figDir = '../accuratePickerV4/resDir/APP++/'
staInfos=seism.StationList(staLstFileL[0])
for staLstFile in staLstFileL[1:]:
    staInfos+=seism.StationList(staLstFile)
if len(sys.argv)>1:
    doL = [ arg for arg in sys.argv[1:]]
else:
    doL =['quake','h-ml'] 

if 'quake' in doL:
    templateL = seism.QuakeL(templateFile)
    for i in range(10):
        T3L = templateL[i].loadSacs(staInfos,matDir=templateDir)
        #templateL[i].plotSacs(T3L,figDir)
        seism.Quake.plotSacs(templateL[i],T3L,figDir+'quake/',key='HBDZKX')
if 'quakeCC' in doL:
    print('loading')
    templateL = seism.QuakeL(templateFile)
    qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)
    print('load done')
    filenameL=templateL.paraL(keyL=['filename'])['filename']
    for i in range(10):
        quakeCC = qCCLNew2[i]
        T3LCC = quakeCC.loadSacs(staInfos,matDir=ccDir)
        tmpName = quakeCC['tmpName']
        template= templateL[filenameL.index(tmpName)]
        T3L = template.loadSacs(staInfos,matDir=templateDir,f=f_1)
        #templateL[i].plotSacs(T3L,figDir)
        seism.Quake.plotSacs(quakeCC,T3LCC,figDir+'quakeCC/',key='HBDZKX',quakeRef=template,T3LRef=T3L,alpha=1)

laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]
R= laL+loL

styleKey = 'HBDZKX'
f_1 = [2,10]
fs.init(styleKey)

if 'ml' in doL:
    ###load results
    qLAll = seism.QuakeL('phase/phaseSCYN')
    #qL=seism.QuakeL('phase/SCYNTOMOSort')
    #qLNew=seism.QuakeL('phase/SCYNTomoRelocV5')
    qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)	
    qCCLNew = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust',Quake=seism.QuakeCC)
    ##adjust magnitude##
    #qLAll.adjustMl()
    #qL.adjustMl()
    #qLNew.adjustMl()
    #qCCLNew.adjustMl()
    #qLNew.adjustMl()

    ###plot###
    ##get quakes infromation
    pL = qLAll.paraL(req={})
    timeL  = np.array(pL['time'])
    mlL  = np.array(pL['ml'])
    laL = np.array(pL['la'])
    loL = np.array(pL['lo'])
    timeL  = np.array(pL['time'])+8*3600
    timeL -=UTCDateTime(2014,1,1).timestamp
    timeL %= 86400
    timeL /=3600
    '''
    ###plot num-t
    plt.close()
    fig=plt.figure(figsize=[5,2.5])
    plt.subplot(1,2,1)
    plt.hist(timeL,np.arange(0,24,0.3))
    plt.xlabel('hour')
    plt.xlim([0,23.5])
    plt.ylabel('number')
    fs.setABC('(a)',pos=[0.05,0.98],key=styleKey)
    plt.subplot(1,2,2)
    plt.hist(mlL,np.arange(0,7,0.1),cumulative=-1,log=True)
    plt.ylabel('number')
    plt.xlabel('magnitude')
    fig.tight_layout()
    fs.setABC('(b)',pos=[0.05,0.90],key=styleKey)
    plt.savefig(figDir+'hl+ml.eps',dpi=300)
    plt.close()
    '''
    pCCL = qCCLNew.paraL(keyL=['time','ml','dep','cc','la','lo','S','M'])
    timeCCL  = np.array(pCCL['time'])
    mlCCL  = np.array(pCCL['ml'])
    mulCCL   = (np.array(pCCL['cc'])-np.array(pCCL['M']))/np.array(pCCL['S'])
    pL1 = qLAll.paraL(req={'time0':timeCCL[0],'time1':timeCCL[-1]})
    timeL1  = np.array(pL1['time'])
    mlL1  = np.array(pL1['ml'])

    pCCL2= qCCLNew2.paraL(keyL=['time','ml','dep','cc','la','lo','S','M'])
    timeCCL2  = np.array(pCCL2['time'])
    mlCCL2  = np.array(pCCL2['ml'])
    mulCCL2   = (np.array(pCCL2['cc'])-np.array(pCCL2['M']))/np.array(pCCL2['S'])

    plt.close()
    plt.figure(figsize=[8,4])
    plt.subplot(1,2,1)
    plt.hist(mlCCL,np.arange(0,7,0.1),cumulative=-1,log=True)
    plt.hist(mlCCL2,np.arange(0,7,0.1),cumulative=-1,log=True)
    plt.hist(mlL1,np.arange(0,7,0.1),cumulative=-1,log=True,alpha=0.8)
    plt.ylabel('number')
    plt.xlabel('magnitude')
    fs.setABC('(a)')
    plt.legend(['WMFT','WMFT_sel','APP++'])
    #plt.savefig('resFig/ml_compare_select.jpg',dpi=300)
    plt.subplot(1,2,2)
    plt.hist(mulCCL,np.arange(6,15,0.02),cumulative=-1,log=True)
    plt.ylabel('number')
    plt.xlabel('Mul')
    fs.setABC('(b)')
    plt.savefig('resFig/mul_ml_cc.svg',dpi=300)
    plt.close()
    pL1 = qLAll.paraL(req={'time0':timeCCL[0],'time1':timeCCL[-1],'R':[27.5,28.5,104,105.5]})
    pCCL2= qCCLNew2.paraL(keyL=['time','ml','dep','cc','la','lo','S','M'],req={'time0':timeCCL[0],'time1':timeCCL[-1],'R':[27.5,28.5,104,105.5]})
    timeDay = (np.array(pL1['time'])-UTCDateTime(2019,6,17).timestamp)/86400
    timeDayCC = (np.array(pCCL2['time'])-UTCDateTime(2019,6,17).timestamp)/86400
    plt.close()
    plt.figure(figsize=[5,5])
    plt.subplot(2,1,1)
    plt.plot(timeDay,pL1['ml'],'.k',markersize=0.3)
    plt.ylabel('magnitude')
    plt.ylim([0,7])
    plt.xlim([-30,30])
    #plt.xlabel('Days from 2019:06:17')
    plt.legend(['APP++'])
    plt.subplot(2,1,2)
    plt.plot(timeDayCC,pCCL2['ml'],'.r',markersize=0.3)
    plt.ylabel('magnitude')
    plt.ylim([0,7])
    plt.xlim([-30,30])
    plt.xlabel('Days from 2019-06-17')
    plt.legend(['WMFT_sel'])
    plt.savefig('resFig/ml_day.svg',dpi=300)
    plt.close()
    


if 'b' in doL:

    if 'h-ml' not in doL:
        qLAll = seism.QuakeL('phase/phaseSCYN')
        pL = qLAll.paraL(req={})
        timeL  = np.array(pL['time'])
        mlL  = np.array(pL['ml'])
        laL = np.array(pL['la'])
        loL = np.array(pL['lo'])
    loLR = np.arange(98,108,0.2)
    laLR = np.arange(21,35,0.2)
    rMax=75
    pointL=np.array([laL.tolist(),loL.tolist()]).transpose()
    rM=[[mathFunc_bak.Round([la,lo],rMax) for lo in loLR]for la in laLR]
    mlLM=mathFunc_bak.devide(rM,pointL,mlL)
    #min_mag=2.25,max_mag=4.2
    bM=np.array([[mathFunc_bak.calc_B(mlL,min_num=150,min_mag=0.8,max_mag=5)for mlL in mlLL]for mlLL in mlLM])
    mapTool.showInMap(bM[:,:,0],laLR,loLR,R,resFile=figDir+'bMap.eps',name='b',abc='(a)')
    mapTool.showInMap(bM[:,:,1],laLR,loLR,R,resFile=figDir+'cMap.eps',name='mc',abc='(b)') 

if 'allQuake' in doL:
    ###load results
    fs.init(styleKey)
    qLAll = seism.QuakeL('phase/phaseSCYN')
    detecQuake.plotQuakeLDis(staInfos,qLAll,laL,loL,filename\
          =figDir+'./allQuake.eps',isTopo=True)
tomoDir = '/HOME/jiangyr/detecQuake/output/outputVSCYNdoV40/tomoDD_bak/input/'
if 'tomoSetting' in doL:
    tomoDD.plot1D(figDir+'./velocity.eps')
if 'tomoQuake' in doL:
    ###load results
    qL=seism.QuakeL('phase_bak/SCYNTOMOSort')
    qLNew=seism.QuakeL('phase_bak/SCYNTomoRelocV5')
    '''
    tomoDD.plotDistPSFreq(qL,staInfos,figDir+'distTime.eps')
    tomoDD.plotDistPSFreq(qLNew,staInfos,figDir+'distTimeNew.eps')
    tomoDD.analyReloc(tomoDir+'/../inversion/tomoDD.reloc',figDir+'analyReloc.eps')
    dTM = tomoDD.loadDTM(tomoDir+'../../dTM')
    tomoDD.analyDTM(dTM,figDir+'dTM.eps')
    '''
    tomoDD.diff(qL,qLNew,figDir+'diff.eps')
    detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          =figDir+'./tomoQuake.eps',isTopo=True)
    detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          =figDir+'./tomoQuakeReloc.eps',isTopo=True)
    with open(figDir+'./tomoQuake.info','w+') as f:
        f.write('input: event:%d p: %d s:%d\n'%qL.analy())
        f.write('reloc: event:%d p: %d s:%d\n'%qLNew.analy())

if 'line' in doL:
    qL=seism.QuakeL('phase_bak/SCYNTOMOSort')
    qLNew=seism.QuakeL('phase_bak/SCYNTomoRelocV5')
    qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)	
    #qCCLNew = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust',Quake=seism.QuakeCC)
    laL=[21,35]#area: [min latitude, max latitude]
    loL=[97,109]
    R = laL+loL
    NL=np.array([[30,102],[33,102],[33,106],[30,106],[27,104.5],[24,103],[23,100],[24,99],[27,100],[30,102]])
    
    #mL=tomoDD.model(tomoDir+'/../inversion/',qLNew,[],isDWS=True,minDWS=2,R=R,vR=NL)
    #mL.plot(figDir+'/real/',['vp','vs'])
    mLSyn=tomoDD.model(tomoDir+'/../Syn/Vel/',isSyn=True,isDWS=True,minDWS=3)
    mLSyn.plot(figDir+'/syn/',nameL=['dVp','dVs','dVpr','dVsr'],doDense=False)
    '''
    mLSyn.plot(figDir+'/syn/',nameL=['dVp','dVs','dVpr','dVsr'],doDense=False)
    RL=[]
    pL0 = np.array([[33,103],[30,106]])
    RL.append(mathFunc_bak.Line(pL0,20,name='A'))
    pL0 = np.array([[32,102],[29,105]])
    RL.append(mathFunc_bak.Line(pL0,20,name='B'))
    pL0 = np.array([[30,102],[33,106]])
    RL.append(mathFunc_bak.Line(pL0,20,name='C'))
    #pL0 = np.array([[30,100],[25.5,104.5]])
    pL0 = np.array([[30-2.3,100],[25.5-2.3,104.5]])
    RL.append(mathFunc_bak.Line(pL0,20,name='D'))
    pL0 = np.array([[26.5,102],[26.5001,104.5]])
    RL.append(mathFunc_bak.Line(pL0,20,name='E'))
    pL0 = np.array([[28,102.7],[25,102.71]])
    RL.append(mathFunc_bak.Line(pL0,20,name='F'))
    detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename=figDir+'resFig/tomoQuakeWithLine.eps',isTopo=True,rL=RL)
    detecQuake.plotQuakeLDis(staInfos,qCCLNew2,laL,loL,filename        =figDir+'resFig/ccQuakeWithLine.eps',isTopo=True,rL=RL)
    for r in  RL:
        mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaP.eps'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
        mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaS.eps'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
        mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sP.eps'%r.name,vModel=mL.vp,isPer=False,isTopo=True)
        mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sS.eps'%r.name,vModel=mL.vs,isPer=False,isTopo=True)
        #mapTool.plotDepV2(qCCLNew2,r,figDir+'resFig/dep%sRelaPCC.eps'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
        #mapTool.plotDepV2(qCCLNew2,r,figDir+'resFig/dep%sRelaSCC.eps'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
    '''
if 'lineCC' in doL:
    #qL=seism.QuakeL('phase_bak/SCYNTOMOSort')
    qLNew=seism.QuakeL('phase_bak/SCYNTomoRelocV5')
    qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)	
    #qCCLNew = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust',Quake=seism.QuakeCC)
    laL=[21,35]#area: [min latitude, max latitude]
    loL=[97,109]
    R = laL+loL
    NL=np.array([[30,102],[33,102],[33,106],[30,106],[27,104.5],[24,103],[23,100],[24,99],[27,100],[30,102]])
    #mL=tomoDD.model(tomoDir+'/../inversion/',qLNew,[],isDWS=True,minDWS=2,R=R,vR=NL)
    #mL.plot(figDir+'/real/',['vp','vs'])
    #mLSyn=tomoDD.model(tomoDir+'/../Syn/Vel/',isSyn=True,isDWS=True,minDWS=3)
    #mLSyn.plot(figDir+'/syn/',nameL=['dVp','dVs','dVpr','dVsr'],doDense=False)
    RL=[]
    pL0 = np.array([[30-0.25,104.5+0.25],[27-0.25,104.5+0.25]])
    RL.append(mathFunc_bak.Line(pL0,15,name='A'))
    pL0 = np.array([[28.5-0.25,103+0.25],[28.5-0.25,106+0.25]])
    RL.append(mathFunc_bak.Line(pL0,15,name='B'))
    pL0 = np.array([[30-0.25,106+0.25],[27-0.25,103+0.25]])
    RL.append(mathFunc_bak.Line(pL0,15,name='C'))
    #pL0 = np.array([[30,100],[25.5,104.5]])
    pL0 = np.array([[30-0.25,103+0.25],[27-0.25,106+0.25]])
    RL.append(mathFunc_bak.Line(pL0,15,name='D'))
    qCCLNew2.set('sort','time')
    qCCLNew2.sort()
    qLNew.select(req={'time0':qCCLNew2[0]['time'],'time1':qCCLNew2[-1]['time']})
    #detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename=figDir+'resFig/tomoQuakeWithLine.eps',isTopo=True,rL=RL)
    detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename        =figDir+'resFig/sameTimeQuake.eps',isTopo=True)
    #detecQuake.plotQuakeLDis(staInfos,qCCLNew2,laL,loL,filename        =figDir+'resFig/ccQuakeWithLine.eps',isTopo=True,rL=RL,R0=[27.5,28.5,104,105.5])
    for r in []:#  RL:
        #mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaP.eps'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
        #mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaS.eps'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
        #mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sP.eps'%r.name,vModel=mL.vp,isPer=False,isTopo=True)
        #mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sS.eps'%r.name,vModel=mL.vs,isPer=False,isTopo=True)
        mapTool.plotDepV2(qCCLNew2,r,figDir+'resFig/dep%sRelaPCC.eps'%r.name,isPer=True,isTopo=True)
        mapTool.plotDepV2(qCCLNew2,r,figDir+'resFig/dep%sRelaSCC.eps'%r.name,isPer=True,isTopo=True)
if 'sampleTest' in doL:
    from SeismTool.MFT import cudaFunc
    N = 20000
    freq = 1000
    mid  =  N/2
    Nfre  = freq/2
    resample = 20
    bandpassL =[[0.5/Nfre,20/Nfre],[0.5/Nfre,12/Nfre],[0.5/Nfre,8/Nfre]]
    strL='ab'
    for i in range(2):
        width = (1+np.random.rand())*freq
        G    = np.exp(-((np.arange(N+int(resample/2))-mid)/width)**2)
        plt.close()
        xL = []
        plt.figure(figsize=[4.5,4])
        wave = np.random.randn(N+int(resample/2))*G
        wave/= np.abs(wave).max()
        wave0= wave[:-int(resample/2)]
        wave1= wave[int(resample/2):]
        plt.plot(np.arange(-N/2,N/2)/freq,wave0+3,'k',linewidth=0.3)
        #plt.plot(np.arange(-N/2+resample/2,N/2+resample/2)/freq,wave1-1,'k',linewidth=0.3)
        count=-3.5
        for bandpass in bandpassL:
            b,a = signal.butter(2,bandpass,'bandpass')
            tmp0=signal.resample(signal.filtfilt(b,a,wave0),int(N/resample))
            tmp0-=tmp0.mean()
            tmp0/=(tmp0**2).sum()**0.5
            tmp1=signal.resample(signal.filtfilt(b,a,wave1),int(N/resample))
            tmp1-=tmp1.mean()
            tmp1/=(tmp1**2).sum()**0.5
            xx00 = signal.correlate(tmp0,tmp0,mode='full')
            xx01 = signal.correlate(tmp0,tmp1,mode='full')
            print(xx00.max(),xx01.max(),cudaFunc.torchcorrnn(tmp0,tmp1)[0].cpu())
            plt.plot(np.arange(-N/resample/2,N/resample/2)/freq*resample,tmp0+count+5,'k',linewidth=0.3)
            plt.plot(np.arange(-N/resample/2,N/resample/2)/freq*resample,tmp1+count+4.5,'gray',linewidth=0.3)
            plt.plot(np.arange(-N/resample,N/resample-1)/freq*resample,xx00*0+1+count+3,'b',linewidth=0.3)
            plt.plot(np.arange(-N/resample,N/resample-1)/freq*resample,xx01+count+3,'r',linewidth=0.3)
            count-=3.5
        plt.xlim([-N/2/freq*0.5,N/2/freq*0.5])
        #plt.ylim([])
        #plt.yticks([3,1.5,-0.5,-1.5,-2.5,-3,-4,-4.5,-5.5,-6],['wave','0.5-20','0.5-12','0.5-8','1','cc:0.5-20','1','cc:0.5-12','1','cc:0.5-8'])
        plt.yticks([3,1.5,0.5,-0.5,1.5-3.5,0.5-3.5,-0.5-3.5,1.5-3.5*2,0.5-3.5*2,-0.5-3.5*2],['wave','0.5-20 Hz','1','cc','0.5-12 Hz','1','cc','0.5- 8 Hz','1','cc'])
        fs.setABC('(%s)'%strL[i])
        plt.savefig('resFig/xx_%d.eps'%i,dpi=300)
    ccL=[]
    for bandpass in bandpassL:
        ccL.append([])
        b,a = signal.butter(2,bandpass,'bandpass')
        for i in range(1000):
            wave = np.random.randn(N+int(resample/2))*G
            wave/= np.abs(wave).max()
            wave0= wave[:-int(resample/2)]
            wave1= wave[int(resample/2):]
            tmp0=signal.resample(signal.filtfilt(b,a,wave0),int(N/resample))
            tmp0-=tmp0.mean()
            tmp0/=(tmp0**2).sum()**0.5
            tmp1=signal.resample(signal.filtfilt(b,a,wave1),int(N/resample))
            tmp1-=tmp1.mean()
            tmp1/=(tmp1**2).sum()**0.5
            #xx00 = signal.correlate(tmp0,tmp0,mode='full')
            xx01 = signal.correlate(tmp0,tmp1,mode='full')
            ccL[-1].append(xx01.max())
    plt.close()
    plt.figure(figsize=[4,4])
    for cc in ccL:
        plt.hist(ccL,bins=np.arange(0,1,0.03),alpha=0.3)
    plt.xlim([0.3,1])
    plt.xlabel('cc')
    plt.legend(['0.5-20 Hz','0.5-12 Hz','0.5-8 Hz'])
    plt.savefig('resFig/hist_bp.pdf',dpi=300)

if 'cpu-gpu' in doL:
    from SeismTool.MFT import cudaFunc
    from time import time
    import torch
    torch.set_num_threads(1)
    dtype = torch.float32
    freq = 50
    nL = (np.exp(np.arange(0,np.log(20)+0.001,np.log(20)/10))*freq).astype(np.int)
    N =86400*freq
    cpuTimeL = []
    gpuTimeL = []
    longData = np.random.rand(N)
    loopN=10
    with open('resFig/cpu-gpu.log','a') as f:
        f.write('n gpu cpu')
        for n in nL:
            shortData = np.random.rand(n)
            long  = torch.tensor(longData+1,dtype=dtype,device='cuda:0')
            short = torch.tensor(shortData+1,dtype=dtype,device='cuda:0')
            cudaFunc.torchcorrnn(long,short)
            sTime = time()
            for i in range(loopN):
                cudaFunc.torchcorrnn(long,short)
            print('done')
            eTime = time()
            gpuTimeL.append((eTime-sTime)/loopN)
            print('gpu:',eTime-sTime)
            long  = torch.tensor(longData+1,dtype=dtype,device='cpu')
            short = torch.tensor(shortData+1,dtype=dtype,device='cpu')
            cudaFunc.torchcorrnn(long,short)
            sTime = time()
            for i in range(loopN):
                cudaFunc.torchcorrnn(long,short)
            print('done')
            eTime = time()
            cpuTimeL.append((eTime-sTime)/loopN)
            print('cpu:',eTime-sTime)
            f.write('%d,%f %f'%(n,gpuTimeL[-1],cpuTimeL[-1]))
    plt.close()
    plt.figure(figsize=[3,3])
    plt.plot(nL,cpuTimeL,'o-k')
    plt.plot(nL,gpuTimeL,'d-k')
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('length')
    plt.ylabel('time/s')
    plt.savefig('resFig/cpu-gpu.eps')

if 'cpu-gpu2' in doL:
    from SeismTool.MFT import cudaFunc
    from time import time
    import torch
    torch.set_num_threads(30)
    dtype = torch.float32
    dtype2 = torch.float64
    freq = 50
    nL = (np.exp(np.arange(0,np.log(20)+0.001,np.log(20)/10))*freq).astype(np.int)
    N =86400*freq
    cpuTimeL = []
    gpuTimeL = []
    longData = np.random.rand(N).astype(np.float32)
    loopN=10
    with open('resFig/cpu-gpu.log','a') as f:
        f.write('n cpu\n')
        for n in []:#nL:
            shortData = np.random.rand(n).astype(np.float32)
            long  = torch.tensor(longData+1,dtype=dtype,device='cpu')
            short = torch.tensor(shortData+1,dtype=dtype,device='cpu')
            cudaFunc.torchcorrnn(long,short)
            sTime = time()
            for i in range(loopN):
                cudaFunc.torchcorrnn(long,short)
                #cudaFunc.torchcorrnp(longData,shortData)
            print('done')
            eTime = time()
            cpuTimeL.append((eTime-sTime)/loopN)
            print('cpu:',eTime-sTime)
            f.write('%d %f\n'%(n,cpuTimeL[-1]))
        f.write('n gpu\n')
        for n in nL:
            shortData = np.random.rand(n)
            long  = torch.tensor(longData+1,dtype=dtype,device='cuda:0')
            short = torch.tensor(shortData+1,dtype=dtype,device='cuda:0')
            cudaFunc.torchcorrnn(long,short)
            sTime = time()
            for i in range(loopN):
                cudaFunc.torchcorrnn(long,short)
            print('done')
            eTime = time()
            gpuTimeL.append((eTime-sTime)/loopN)
            print('gpu:',eTime-sTime)
            f.write('%d %f\n'%(n,gpuTimeL[-1]))
    
    plt.close()
    plt.figure(figsize=[3,3])
    plt.plot(nL,cpuTimeL,'o-k')
    plt.plot(nL,gpuTimeL,'d-k')
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('length')
    plt.ylabel('time/s')
    plt.savefig('resFig/cpu-gpu.eps')
        
            
