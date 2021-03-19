from obspy import read,UTCDateTime
from glob import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from SeismTool.io import seism,sacTool,tool,dataLib
from SeismTool.detector import detecQuake
from SeismTool.io.seism import StationList,QuakeL
from SeismTool.locate.locate import locator

#'''
la0 = -37.5628
lo0 = -179.4443
#center_lon = -110  # west US 
maxDist = 10*110
wkDir   = '../xinxilan2/'
pattern = '../xinxilan2/*/*SAC'
sacFileL = glob(pattern)
staD = {}
stationL = seism.StationList()
for sacFileO in sacFileL:
    sacFile = os.path.basename(sacFileO)
    staKeys  = sacFile.split('.')
    staKey   = staKeys[0]+'.'+staKeys[1]
    if staKey in staD:
        continue
    staD[staKey]=1
    sac      = read(sacFileO,headonly=True)[0]
    net      = sac.stats['sac']['knetwk']
    sta      = sac.stats['sac']['kstnm']
    la       = sac.stats['sac']['stla']
    lo       = sac.stats['sac']['stlo']
    dep      = 0
    compBase = sac.stats['sac']['kcmpnm'][:2]
    station  = seism.Station(net=net,sta=sta,la=la,lo=lo,dep=dep,compBase=compBase,nameMode='xinxilan')
    print(station.dist([la0,lo0]))
    if station.dist([la0,lo0])<maxDist:
        stationL.append(station)
        print(stationL[-1])

stationL.write(wkDir+'staLst','net sta la lo dep compBase nameMode')
stationL.plot(wkDir+'staLst.pdf')
'''
from SeismTool.deepLearning import fcn
from tensorflow.keras.models import load_model
workDir = '../xinxilan_wk/'
stationFile='../xinxilan2/staLst'

inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
laL=[-40.79,-7.12]#area: [min latitude, max latitude]
loL=[165.66,-166.74+360]#area: [min longitude, max longitude]
laN=40 #subareas in latitude
loN=40 #subareas in longitude
maxD=21*10
f=[0.5/10,20/10]
bSec=UTCDateTime(2021,2,27).timestamp
eSec=UTCDateTime(2021,3,9).timestamp


if not os.path.exists(workDir):
    os.makedirs(workDir)

if not os.path.exists(workDir+'output/'):
    os.makedirs(workDir+'output/')

if not os.path.exists(workDir+'/phaseDir/'):
    os.makedirs(workDir+'/phaseDir/')

#fcn.defProcess()

taupM=tool.quickTaupModel(modelFile=inputDir+'/include/iaspTaupMat')
modelL = [load_model(file,compile=False) for file in[inputDir+'model/norm_p_1000000_200000',inputDir+'model/norm_s_1000000_200000'] ]
staInfos=StationList(stationFile)
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat)
quakeL=QuakeL()
v_i='0'
p_i='0'
cudaI='1'
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
     modelL, date, mode='norm',f=f,maxD=maxD,delta0=0.02*10)
    tmpQuakeL=detecQuake.associateSta(staL, aMat, \
        staTimeML, timeR=60, maxDTime=20, N=1,locator=\
        locator(staInfos),maxD=maxD,taupM=taupM)
    #save:
    #result's in  workDir+'phaseDir/phaseLstVp_i'
    #result's waveform in  workDir+'output/outputVv_i/'
    #result's plot picture in  workDir+'output/outputVv_i/'
    if len(tmpQuakeL)>0:
        seism.saveSacs(staL, tmpQuakeL, staInfos,\
            matDir=workDir+'output/outputV%s/'%v_i,\
                bSec=-100,eSec=400)
        detecQuake.plotResS(staL,tmpQuakeL,outDir\
            =workDir+'output/outputV%s/'%v_i)
        detecQuake.plotQuakeL(staL,tmpQuakeL,laL,loL,outDir\
            =workDir+'output/outputV%s/'%v_i)
    quakeL+=tmpQuakeL
    quakeL.write(workDir+'phaseDir/phaseLstV%s'%p_i)
    staL=[]# clear data  to save memory
'''