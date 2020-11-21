from obspy import UTCDateTime
import os
import sys
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from imp import reload
from SeismTool.io import seism,sacTool,tool
from SeismTool.detector import detecQuake
from SeismTool.io.seism import StationList,QuakeL
from SeismTool.locate.locate import locator
from multiprocessing import Pool
#from SeismTool.deepLearning import fcn
from SeismTool.locate import locate

detecQuake.maxA=1e15#这个是否有道理


#2019-06-17
#250~1750衡量
#bSec=UTCDateTime(2018,7,1).timestamp#begain date#
#eSec=UTCDateTime(2020,1,1).timestamp# end date
#p_i='SCYN_allV4'
#v_i='SCYN_all'#phaseSaveFile

#os.environ["CUDA_VISIBLE_DEVICES"] = cudaI
inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'

argL=[['SCYN1122','SCYN1122V0','0',UTCDateTime(2014,1,1).timestamp,UTCDateTime(2015,7,1).timestamp],\
      ['SCYN1122','SCYN1122V1','0',UTCDateTime(2015,7,1).timestamp,UTCDateTime(2017,1,1).timestamp],\
      ['SCYN1122','SCYN1122V2','1',UTCDateTime(2017,1,1).timestamp,UTCDateTime(2018,7,1).timestamp],\
      ['SCYN1122','SCYN1122V3','1',UTCDateTime(2018,7,1).timestamp,UTCDateTime(2020,1,1).timestamp],]
l=argL[int(sys.argv[1])]
print(l)
v_i,p_i,cudaI,bSec,eSec =l 
os.environ["CUDA_VISIBLE_DEVICES"] = cudaI
#def do(l):
from SeismTool.deepLearning import fcn

workDir='/HOME/jiangyr/detecQuake/'# workDir: the dir to save the results
staLstFileL=[inputDir+'../stations/XU_sel.sta',inputDir+'../stations/SCYN_withComp_ac',]
#station list file
#bSec=UTCDateTime(2018,7,1).timestamp#begain date
#eSec=UTCDateTime(2020,1,1).timestamp# end date
#p_i='SCYN_allV4'
#v_i='SCYN_all'#phaseSaveFile
fcn.defProcess()
#output dir

laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]#area: [min longitude, max longitude]
laN=40 #subareas in latitude
loN=40 #subareas in longitude
maxD=21#max ts-tp
f=[0.5,20]

if not os.path.exists(workDir+'output/'):
    os.makedirs(workDir+'output/')
if not os.path.exists(workDir+'/phaseDir/'):
    os.makedirs(workDir+'/phaseDir/')
#####no need to change########
taupM=tool.quickTaupModel(modelFile=inputDir+'/include/iaspTaupMat')
modelL = [fcn.genModel0('norm','p')[0],fcn.genModel0('norm','s')[0]]
modelL[0].load_weights(inputDir+'model/norm_p_400000_80000')
modelL[1].load_weights(inputDir+'model/norm_s_400000_80000')
staInfos=StationList(staLstFileL[0])+StationList(staLstFileL[1])
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeL=QuakeL()
#############################

for date in range(int(bSec),int(eSec), 86400):
    print('doing:',v_i,p_i,cudaI,bSec,eSec)
    dayNum=int(date/86400)
    dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML,\
     modelL, date, mode='norm',f=f,maxD=maxD)
    tmpQuakeL=detecQuake.associateSta(staL, aMat, \
        staTimeML, timeR=30, maxDTime=2, N=1,locator=\
        locator(staInfos),maxD=maxD,taupM=taupM)
    '''
    save:
    result's in  workDir+'phaseDir/phaseLstVp_i'
    result's waveform in  workDir+'output/outputVv_i/'
    result's plot picture in  workDir+'output/outputVv_i/'
    '''
    if len(tmpQuakeL)>0:
        seism.saveSacs(staL, tmpQuakeL, staInfos,\
            matDir=workDir+'output/outputV%s/'%v_i,\
                bSec=-10,eSec=40)
        detecQuake.plotResS(staL,tmpQuakeL,outDir\
            =workDir+'output/outputV%s/'%v_i)
        detecQuake.plotQuakeL(staL,tmpQuakeL,laL,loL,outDir\
            =workDir+'output/outputV%s/'%v_i)
    quakeL+=tmpQuakeL
    quakeL.write(workDir+'phaseDir/phaseLstV%s'%p_i)
    staL=[]# clear data  to save memory
'''
detecQuake.plotQuakeL(staL,tql,laL,loL,outDir\
            =workDir+'output/outputV%s/'%v_i)
for sta in staL:
    sta.data = seism.Trace3(sta.data)
detecQuake.plotResS(staL,tql,outDir\
            =workDir+'output/outputV%s/'%v_i)
        seism.saveSacs(staL, tql, staInfos,\
            matDir=workDir+'output/outputV%s/'%v_i,\
                bSec=-10,eSec=40)
seism.saveSacs(staL, qL, staInfos,\
            matDir=workDir+'output/outputV%s/'%v_i,\
                index0=-1500,index1=1500)
R=[35,45,96,105]
staInfos=sacTool.readStaInfos('../staLstNMV2')
staInfos=tool.getStaInArea(staInfos,'staLstNMV2Select',R)
staInfos=sacTool.readStaInfos('staLstNMV2Select')
quakeLs=tool.readQuakeLs('../NM/phaseLstNMALLReloc_removeBadSta',staInfos)
detecQuake.plotQuakeDis(quakeLs[:1],R=R,staInfos=staInfos,topo=None,minCover=0)
plt.savefig('quakeLsDis.png',dpi=500)
'''



