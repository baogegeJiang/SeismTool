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

detecQuake.maxA=1e10
sacDir = '/HOME/jiangyr/hima3_sac'
staDir = '/home/jiangyr/Surface-Wave-Dispersion/stations/'
'''
staFileHima3 = sacDir+'/staLst_hima3*'
staFileHima3New = staDir+'/STALST_HIMA3'
staFileHima2 = staDir+'/STALST_HIMA'
staFileHima23 = staDir+'/STALST_HIMA23'
staFileHima23WithNameMode = staFileHima23+'WithNameMode'
'''
staFileHima23Ordos = staDir+'/STALST_ORDOS'
'''
staLst=seism.StationList(staFileHima3,isUnique=True)

staLst.write(staFileHima3New)

staLst.read(staFileHima2,isUnique=True)
staLst.write(staFileHima23) 
staLst.set('nameMode','hima23')
staLst.write(staFileHima23WithNameMode,'net sta compBase lo la dep nameMode')
staLst.plot('resFig/Hima23.jpg')
R= [33,43,104,116]
staLst.inR(R)
staLst.plot('resFig/Hima23Ordos.jpg')
staLst.write(staFileHima23Ordos,'net sta compBase lo la dep nameMode')
'''
staInfos = seism.StationList(staFileHima23Ordos)
fcn.defProcess()
v_i='Ordos2021-0606'
p_i='Ordos2021-0606V1'
bSec=UTCDateTime(2014,2,1).timestamp
bSec=UTCDateTime(2014,1,1).timestamp
eSec=UTCDateTime(2015,1,1).timestamp
laL=[32,45]#area: [min latitude, max latitude]
loL=[102,118]#area: [min longitude, max longitude]
laN=40 #subareas in latitude
loN=40 #subareas in longitude
maxD=35#max ts-tp
maxDA=15
f=[0.5,20]
workDir='/HOME/jiangyr/detecQuake/'
workDir='/home/jiangyr/detecQuake/'
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
        staTimeML, timeR=30, maxDTime=2, N=1,locator=\
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