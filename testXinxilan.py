from obspy import read,UTCDateTime
from glob import glob
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from SeismTool.io import seism,sacTool,tool,dataLib
from SeismTool.detector import detecQuake
from SeismTool.io.seism import StationList,QuakeL
from SeismTool.locate.locate import locator
from matplotlib import pyplot as plt
import matplotlib.animation as animation
doL = sys.argv[1:]
if 'readStaInfo' in doL:
    la0 = -37.5628
    lo0 = -179.4443
    #center_lon = -110  # west US 
    maxDist = 40*110
    wkDir   = '../xinxilan2/'
    pattern = '../xinxilan2/*/*SAC'
    sacFileL = glob(pattern)
    staD = {}
    stationL = seism.StationList()
    for sacFileO in sacFileL:
        sacFile = os.path.basename(sacFileO)
        staKeys  = sacFile.split('.')
        staKey   = staKeys[0]+'.'+staKeys[1]+'.'+staKeys[3][:2]
        if staKey in staD:
            continue
        staD[staKey]=1
        sac      = read(sacFileO,headonly=True)[0]
        if  sac.stats['sac']['delta']>0.1:
            print(sac.stats['sac']['delta'],'too low f')
            continue
        net      = sac.stats['sac']['knetwk']
        sta      = sac.stats['sac']['kstnm']
        la       = sac.stats['sac']['stla']
        lo       = sac.stats['sac']['stlo']
        dep      = sac.stats['sac']['stel']
        compBase = sac.stats['sac']['kcmpnm'][:2]
        station  = seism.Station(net=net,sta=sta,la=la,lo=lo,dep=dep,compBase=compBase,nameMode='xinxilan')
        print(station.dist([la0,lo0]))
        if station.dist([la0,lo0])<maxDist:
            stationL.append(station)
            print(stationL[-1])

    stationL.write(wkDir+'staLst','net sta la lo dep compBase nameMode')
    stationL.plot(wkDir+'staLst.pdf')
if 'detecQuake' in doL:
    from SeismTool.deepLearning import fcn
    from tensorflow.keras.models import load_model
    workDir = '../xinxilan_wk/'
    stationFile='../xinxilan2/staLst'

    inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
    decN=4
    laL=[-44.79,-20]#area: [min latitude, max latitude]
    loL=[165.66,-172.74+360]#area: [min longitude, max longitude]
    laN=40 #subareas in latitude
    loN=40 #subareas in longitude
    maxD=40*decN
    f=[max(1/5,0.5/decN),20/decN]
    bSec=UTCDateTime(2021,2,27).timestamp
    bSec=UTCDateTime(2021,3,4).timestamp
    eSec=UTCDateTime(2021,3,9).timestamp


    if not os.path.exists(workDir):
        os.makedirs(workDir)

    if not os.path.exists(workDir+'output/'):
        os.makedirs(workDir+'output/')

    if not os.path.exists(workDir+'/phaseDir/'):
        os.makedirs(workDir+'/phaseDir/')

    #fcn.defProcess()

    taupM=tool.quickTaupModel(modelFile=inputDir+'/include/iaspTaupMat')
    modelL = [load_model(file,compile=False) for file in[inputDir+'model/norm_p_2000000_400000',inputDir+'model/norm_s_2000000_400000'] ]
    staInfos=StationList(stationFile)
    aMat=sacTool.areaMat(laL,loL,laN,loN)
    staTimeML= detecQuake.getStaTimeL(staInfos, aMat)
    quakeL=QuakeL()
    v_i='3'
    p_i='3'
    cudaI='1'
    if not os.path.exists(workDir+'output/outputV%s/'%v_i):
        os.makedirs(workDir+'output/outputV%s/'%v_i)
    staInfos.plot(workDir+'output/outputV%s/staDist.jpg'%v_i)
    detecQuake.maxA=1e15
    for date in range(int(bSec),int(eSec), 86400):
        print('doing:',v_i,p_i,cudaI,bSec,eSec)
        dayNum=int(date/86400)
        dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
        if os.path.exists(dayDir):
            print('done')
            continue
        date=UTCDateTime(float(date))
        print('pick on ',date)
        staL = detecQuake.getStaL(staInfos,staTimeML,\
        modelL, date, mode='norm',f=f,maxD=maxD,delta0=0.02*decN,maxDTime=2*decN)
        tmpQuakeL=detecQuake.associateSta(staL, aMat, \
            staTimeML, timeR=15*decN, maxDTime=2*decN, N=1,locator=\
            locator(staInfos,maxDT=maxD/0.7),maxD=maxD,taupM=taupM)
        #save:
        #result's in  workDir+'phaseDir/phaseLstVp_i'
        #result's waveform in  workDir+'output/outputVv_i/'
        #result's plot picture in  workDir+'output/outputVv_i/'
        if len(tmpQuakeL)>0:
            seism.saveSacs(staL, tmpQuakeL, staInfos,\
                matDir=workDir+'output/outputV%s/'%v_i,\
                    bSec=-20*decN,eSec=20*decN)
            detecQuake.plotResS(staL,tmpQuakeL,outDir\
                =workDir+'output/outputV%s/'%v_i)
            detecQuake.plotQuakeL(staL,tmpQuakeL,laL,loL,outDir\
                =workDir+'output/outputV%s/'%v_i)
        quakeL+=tmpQuakeL
        quakeL.write(workDir+'phaseDir/phaseLstV%s'%p_i)
        staL=[]# clear data  to save memory
if 'pickOnQuake' in doL:
    from SeismTool.deepLearning import fcn
    from tensorflow.keras.models import load_model
    workDir = '../xinxilan_wk/'
    stationFile='../xinxilan2/staLst'
    quakeFile0 = '../xinxilan2/catalog'

    inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
    decN=5
    laL=[-44.79,-10]#area: [min latitude, max latitude]
    loL=[155.66,-162.74+360]#area: [min longitude, max longitude]
    maxD=40*decN
    f=[max(1/4,0.5/decN),20/decN]
    bSec=UTCDateTime(2021,2,27).timestamp
    #bSec=UTCDateTime(2021,3,4).timestamp
    eSec=UTCDateTime(2021,3,9).timestamp

    if not os.path.exists(workDir):
        os.makedirs(workDir)
    if not os.path.exists(workDir+'output/'):
        os.makedirs(workDir+'output/')
    if not os.path.exists(workDir+'/phaseDir/'):
        os.makedirs(workDir+'/phaseDir/')

    #fcn.defProcess()

    modelL = [load_model(file,compile=False) for file in[inputDir+'model/norm_p_2000000_400000',inputDir+'model/norm_s_2000000_400000'] ]
    staInfos=StationList(StationList(stationFile))
    quakeL = QuakeL(QuakeL(quakeFile0))
    v_i='QuakeV5'
    p_i='QuakeV5'
    cudaI='1'
    if not os.path.exists(workDir+'output/outputV%s/'%v_i):
        os.makedirs(workDir+'output/outputV%s/'%v_i)
    staInfos.plot(workDir+'output/outputV%s/staDist.jpg'%v_i)
    detecQuake.maxA=1e15
    for date in range(int(bSec),int(eSec), 86400):
        print('doing:',v_i,p_i,cudaI,bSec,eSec)
        dayNum=int(date/86400)
        date=UTCDateTime(float(date))
        tmpQuakeL = quakeL.Slice(date,date+86400)
        print('quake num:',len(tmpQuakeL))
        dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
        if os.path.exists(dayDir):
            print('done')
            continue
        print('pick on ',date)
        staL = detecQuake.getStaL(staInfos,modelL= modelL, date=date, f=f,delta0=0.02*decN,isPre=False)
        detecQuake.getForQuake(staL,tmpQuakeL,modelL)
        
        #save:
        #result's in  workDir+'phaseDir/phaseLstVp_i'
        #result's waveform in  workDir+'output/outputVv_i/'
        #result's plot picture in  workDir+'output/outputVv_i/'
        if len(tmpQuakeL)>0:
            seism.saveSacs(staL, tmpQuakeL, staInfos,\
                matDir=workDir+'output/outputV%s/'%v_i,\
                    bSec=-20*decN,eSec=20*decN)
            detecQuake.plotResS(staL,tmpQuakeL,outDir\
                =workDir+'output/outputV%s/'%v_i)
            detecQuake.plotQuakeL(staL,tmpQuakeL,laL,loL,outDir\
                =workDir+'output/outputV%s/'%v_i)
        quakeL.write(workDir+'phaseDir/phaseLstV%s'%p_i,quakeKeysIn='type     la       lo          time index   randID filename ml   dep')
        staL=[]# clear data  to save memory

if 'MFT' in doL:
    from SeismTool.MFT import pyMFTCuda
    from SeismTool.detector import detecQuake
    decN=5
    laL=[-44.79,-10]#area: [min latitude, max latitude]
    loL=[160.66,-162.74+360]
    f_0 = [max(1/4,0.5/decN),20/decN]
    f_1 = [max(1/4,2/decN),10/decN]
    date0 = UTCDateTime(2021,3,4)
    dateL = np.arange(-5,5)
    workDir='../xinxilan_wk/'
    stationFile='../xinxilan2/staLst'
    inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
    templateFile = workDir+'phaseDir/phaseLstV%s'%'QuakeV5'
    templateDir = workDir+('output/outputV%s/'%'QuakeV5')
    v_i = '0328MFTV2'
    p_i = '0328MFTV2'
    deviceL=['cuda:0']
    minMul=4
    MINMUL=6
    minChannel=10
    winTime=0.4*decN
    maxThreshold=0.45
    maxCC=1
    staInfos=seism.StationList(stationFile)
    templateL = seism.QuakeL(templateFile)
    for tmp in templateL:
        tmp.nearest(staInfos,N=30,near=20,delta=0.01)
    templateL.select(req={'minN':6})
    templateL.write(templateFile+v_i+'select')
    #templateL=seism.QuakeL(templateL[:20])
    #T3PSLL =seism.getT3LLQuick(templateL,staInfos,matDir=templateDir,f=f_1,batch_size=50,num_workers=2)
    T3PSLL = [ tmp.loadPSSacs(staInfos,matDir=templateDir,f=f_1,delta0=0.02*decN)for tmp in templateL]
    pyMFTCuda.preT3PSLL(T3PSLL,templateL,secL=[-3*decN,4*decN])
    quakeCCL=seism.QuakeL()
    count=0
    for date in date0.timestamp + dateL*86400:
        dayNum=int(date/86400)
        dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
        if os.path.exists(dayDir):
            print('done')
            continue
        if count >=0:
            staL = detecQuake.getStaL(staInfos,date=date,f=f_0,isPre=False,f_new=f_1,delta0=0.02*decN)
            bTime=pyMFTCuda.preStaL(staL,date,deviceL=deviceL,delta=0.02*decN)
        count+=1
        dayQuakeL=pyMFTCuda.doMFTAll(staL,T3PSLL,bTime,n=int(86400*50/decN),quakeRefL=templateL,staInfos=staInfos,minChannel=minChannel,minMul=minMul,MINMUL=MINMUL,winTime=winTime,locator=locate.locator(staInfos,maxDT=80*decN),maxThreshold=maxThreshold,maxCC=maxCC,secL=[-3*decN,4*decN],maxDis=99999,deviceL=deviceL,delta=0.02*decN)
        quakeCCL+=dayQuakeL
        if len(dayQuakeL)>0:
            seism.saveSacs(staL, dayQuakeL, staInfos,\
                matDir=workDir+'output/outputV%s/'%v_i,\
                bSec=-10*decN,eSec=40*decN)
            detecQuake.plotResS(staL,dayQuakeL,outDir=workDir+'output/outputV%s/'%v_i)
            detecQuake.plotQuakeL(staL,dayQuakeL,laL,loL,outDir=workDir+'output/outputV%s/'%v_i)
            quakeCCL.write(workDir+'phaseDir/phaseLstV%s'%p_i)

if 'anMFT1' in doL:
    workDir='../xinxilan_wk/'
    stationFile='../xinxilan2/staLst'
    v_i = '0328MFTV2'
    p_i = '0328MFTV2'
    quakeCCFile = workDir+'phaseDir/phaseLstV%s'%p_i
    matDir=workDir+'output/outputV%s/'%v_i
    staInfos=seism.StationList(stationFile)
    templateFile = workDir+'phaseDir/phaseLstV%s'%'QuakeV5'+v_i+'select'
    templateL = seism.QuakeL(templateFile)
    quakeCCL  = seism.QuakeL(quakeCCFile,mode='CC')
    pCCL = quakeCCL.paraL(keyL=['time','ml','dep','cc','la','lo','S','M','tmpName'])
    timeCCL  = np.array(pCCL['time'])
    mlCCL  = np.array(pCCL['ml'])+seism.dm
    mulCCL   = (np.array(pCCL['cc'])-np.array(pCCL['M']))/np.array(pCCL['S'])
    pL = templateL.paraL()
    timeL1  = np.array(pL['time'])
    resDir = workDir+'/resFig/'
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    plt.close()
    fig=plt.figure(figsize=[4,3])
    plt.hist(mulCCL,np.arange(6,15,0.01),cumulative=-1,log=True)
    plt.ylabel('number')
    plt.xlabel('mul')
    fig.tight_layout()
    plt.savefig(resDir+'mul.jpg',dpi=300)
    plt.close()

    timeDay = (np.array(pL['time'])-UTCDateTime(2021,3,4).timestamp)/86400
    timeDayCC = (np.array(pCCL['time'])-UTCDateTime(2021,3,4).timestamp)/86400
    plt.close()
    plt.figure(figsize=[5,5])
    plt.subplot(2,1,1)
    plt.plot(timeDay,pL['ml'],'.k',markersize=0.1)
    plt.ylabel('magnitude')
    #plt.ylim([0,7])
    #plt.xlim([-30,30])
    plt.xlabel('Days from 2021:03:03')
    plt.legend(['APP++'])
    plt.subplot(2,1,2)
    plt.plot(timeDayCC,pCCL['ml'],'.r',markersize=0.1)
    plt.ylabel('magnitude')
    #plt.ylim([0,7])
    #plt.xlim([-30,30])
    plt.xlabel('Days from 2021:03:04')
    plt.legend(['WMFT_sel'])
    plt.savefig(resDir+'ml_day.jpg',dpi=300)
    plt.close()

    quakeFile0 = '../xinxilan2/catalog'
    qLo = seism.QuakeL(quakeFile0)
    mlL  = []
    dmlL = []
    dmlD = {}
    for tmp in templateL:
        q = qLo[qLo.find(tmp)]
        dmlL.append(q['ml'] - tmp['ml'])
        mlL.append(q['ml'])
        dmlD[tmp['filename']] = q['ml'] - tmp['ml']
        tmp['ml'] = q['ml']
    for quakeCC in quakeCCL:
        dml=dmlD[quakeCC['tmpName']]
        quakeCC['ml']+=dml
    plt.plot(mlL,dmlL,'.k')
    plt.savefig(resDir+'dml_ml.jpg',dpi=300)
    quakeCCL.write(quakeCCFile+'_adjustMl')
    templateL.write(templateFile+'_adjustMl')
    pCCL2 = quakeCCL.paraL(keyL=['time','ml','dep','cc','la','lo','S','M','tmpName'])
    pL2 = templateL.paraL()
    timeDay = (np.array(pL['time'])-UTCDateTime(2021,3,4).timestamp)/86400
    timeDayCC = (np.array(pCCL2['time'])-UTCDateTime(2021,3,4).timestamp)/86400
    plt.close()
    plt.figure(figsize=[5,5])
    plt.subplot(2,1,1)
    plt.plot(timeDay,pL2['ml'],'.k',markersize=0.1)
    plt.ylabel('magnitude')
    #plt.ylim([0,7])
    #plt.xlim([-30,30])
    plt.xlabel('Days from 2021:03:03')
    plt.legend(['APP++'])
    plt.subplot(2,1,2)
    plt.plot(timeDayCC,pCCL2['ml'],'.r',markersize=0.1)
    plt.ylabel('magnitude')
    #plt.ylim([0,7])
    #plt.xlim([-30,30])
    plt.xlabel('Days from 2021:03:04')
    plt.legend(['WMFT_sel'])
    plt.savefig(resDir+'ml_day_adjust.jpg',dpi=300)
    plt.close()

if 'anMFT2' in doL:
    if  not 'anMFT1' in doL:
        workDir='../xinxilan_wk/'
        stationFile='../xinxilan2/staLst'
        v_i = '0328MFTV2'
        p_i = '0328MFTV2'
        quakeCCFile = workDir+'phaseDir/phaseLstV%s'%p_i+'_adjustMl'
        matDir=workDir+'output/outputV%s/'%v_i
        quakeCCL  = seism.QuakeL(quakeCCFile,mode='CC')
        resDir = workDir+'/resFig/'
        if not os.path.exists(resDir):
            os.makedirs(resDir)
    staInfos=seism.StationList(stationFile)
    fig, ax = plt.subplots()
    hourDir = resDir+'hour/'
    if not os.path.exists(hourDir):
        os.makedirs(hourDir)
    #ln, = plt.scatter(xdata, ydata,mlL,depL, 'ro',animated=True)
    dayPer = 24
    plt.close()
    plt.figure(figsize=[7,5])
    #plt.set_aspect(1)
    plt.axis('equal')
    staInfos.plot(isSave=False)
    count=0
    for time in np.arange(-5,5,1/dayPer)*86400+UTCDateTime(2021,3,4).timestamp:
        time0 = time
        time1 = time+86400/dayPer
        pCCLIn= quakeCCL.paraL(req={'time0':time0,'time1':time1})
        #plt.scatter(np.array(pCCLIn['lo'])%180,np.array(pCCLIn['la']),np.array(pCCLIn['ml'])/3,pCCLIn['dep'])
        xdata, ydata,mlL,depL = [],[],[],[]
        xdata += (np.array(pCCLIn['lo'])%360).tolist()
        ydata += (np.array(pCCLIn['la'])).tolist()
        mlL   += (np.array(pCCLIn['ml'])).tolist()
        depL  += pCCLIn['dep']
        s = plt.scatter(xdata, ydata,mlL,depL,vmin=0,vmax=800)
        if count ==0:
            plt.colorbar()
            count+=1
        plt.xlim(163,(-164)%360)
        plt.ylim(-40, -12)
        t=plt.title(UTCDateTime(time1).strftime('untill %Y-%m+%d+%H-%M-%S'))
        plt.savefig(hourDir+UTCDateTime(time1).strftime('%Y-%m-%d+%H-%M-%S.jpg'),dpi=300)
    #plt.savefig(resDir+'locChanges.gif')