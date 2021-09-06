import os
from imp import reload
from SeismTool.io import seism,sacTool,tool
from SeismTool.detector import detecQuake
from SeismTool.io.seism import QuakeL
from SeismTool.locate.locate import locator
from SeismTool.io import seism
from obspy import UTCDateTime
from SeismTool.deepLearning import fcn
from tensorflow.keras.models import load_model

detecQuake.maxA=1e4
inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
staInfos = seism.StationList(inputDir+'../stations/SCYN_withComp_ac')
fcn.defProcess()
v_i='SCYNV2_2021-0608'
p_i='SCYNV2_2021-0608V4'
bSec=UTCDateTime(2014,1,1).timestamp
eSec=UTCDateTime(2020,1,1).timestamp
laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]#area: [min longitude, max longitude]
laN=50 #subareas in latitude
loN=50 #subareas in longitude
maxD=40#max ts-tp
maxDA=35
f=[0.5,20]
workDir='/HOME/jiangyr/detecQuake/'
if not os.path.exists(workDir+'output/'):
    os.makedirs(workDir+'output/')
if not os.path.exists(workDir+'/phaseDir/'):
    os.makedirs(workDir+'/phaseDir/')

inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
taupM=tool.quickTaupModel(modelFile=inputDir+'/include/iaspTaupMat')
modelL = [load_model(file,compile=False) for file in[inputDir+'model/None_p_4000000_800000',inputDir+'model/None_s_4000000_800000'] ]
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat,MAXDIS=maxD/0.7*9)
quakeL=QuakeL()
for date in range(int(bSec),int(eSec), 86400):
    print('doing:',v_i,p_i,bSec,eSec)
    dayNum=int(date/86400)
    dayDir=workDir+('output/output%s/'%v_i)+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    staL = detecQuake.getStaL(staInfos,staTimeML,\
     modelL, date, mode='norm',f=f,maxD=maxD,taupM=taupM)
    tmpQuakeL=detecQuake.associateSta(staL, aMat, \
        staTimeML, timeR=60, maxDTime=2, N=1,locator=\
        locator(staInfos),maxD=maxD,maxDA=maxDA,taupM=taupM,loopN=3,halfMaxDTime=1)
    '''
    save:
    result's in  workDir+'phaseDir/phaseLstp_i'
    result's waveform in  workDir+'output/outputv_i/'
    result's plot picture in  workDir+'output/outputv_i/'
    '''
    if len(tmpQuakeL)>0:
        seism.saveSacs(staL, tmpQuakeL, staInfos,\
            matDir=workDir+'output/output%s/'%v_i,\
                bSec=-10,eSec=40)
        detecQuake.plotResS(staL,tmpQuakeL,outDir\
            =workDir+'output/output%s/'%v_i)
        detecQuake.plotQuakeL(staL,tmpQuakeL,laL,loL,outDir\
            =workDir+'output/output%s/'%v_i)
    quakeL+=tmpQuakeL
    quakeL.write(workDir+'phaseDir/phaseLst%s'%p_i)
    staL=[]#