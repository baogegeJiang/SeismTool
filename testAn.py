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
import numpy as np
from matplotlib import pyplot as plt
laN=60 #subareas in latitude
loN=60
laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]
detecQuake.maxA=1e15#这个是否有道理

inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'

argL=[['SCYNdoV40', 'SCYNdoV40V0', '0', UTCDateTime(2014,1,1).timestamp, UTCDateTime(2015,7,1).timestamp],\
      ['SCYNdoV10', 'SCYNdoV10V1', '0', UTCDateTime(2015,7,1).timestamp, UTCDateTime(2017,1,1).timestamp],\
      ['SCYNdoV10', 'SCYNdoV10V2', '1', UTCDateTime(2017,1,1).timestamp, UTCDateTime(2018,7,1).timestamp],\
      ['SCYNdoV10', 'SCYNdoV10V3', '1', UTCDateTime(2018,7,1).timestamp, UTCDateTime(2020,1,1).timestamp],]

v_i,p_i=argL[0][:2]


workDir='/HOME/jiangyr/detecQuake/'# workDir: the dir to save the results
staLstFileL=[inputDir+'../stations/XU_sel.sta',inputDir+'../stations/SCYN_withComp_ac',]
phaseFile=workDir+'phaseDir/phaseLstV%s'%p_i[:-1]+'?'
workDir+'output/outputV%s/'%v_i
staInfos=StationList(staLstFileL[0])+StationList(staLstFileL[1])
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat)

qL = QuakeL(phaseFile)
pL = qL.paraL()

timeL  = np.array(pL['time'])
mlL  = np.array(pL['ml'])

plt.plot(timeL[mlL>-10]%86400,mlL[mlL>-10],'.',markersize=0.5)
plt.savefig('%s/time_ml.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)


timeL  = np.array(pL['time'])+8*3600
timeL -=UTCDateTime(2014,1,1).timestamp
timeL %= 86400
timeL /=3600
mlL  = np.array(pL['ml'])
plt.close()
plt.hist(timeL,np.arange(0,24,1))
plt.savefig('%s/hl_ml.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()
seism.getQuakeNoise(qL,staInfos,workDir+'output/outputV%s/'%v_i,workDir+'output/outputV%s/noiseStd'%v_i)
coverL=[]
for q in qL:
    coverL.append(q.calCover(staInfos))





plt.close()
plt.hist()
plt.savefig('%s/ml.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()

pL = qL.paraL(req={'loc0':[27,103,0],'maxDist':30})
pL = qL.paraL(req={'minN':10,'minCover':0.8})

timeL  = np.array(pL['time'])+8*3600
timeL -=UTCDateTime(2014,1,1).timestamp
timeL %= 86400
timeL /=3600
mlL =  np.array(pL['ml'])

plt.close()
plt.hist(timeL,np.arange(0,24,1))
plt.savefig('%s/hl_ml.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()


mlL0 = mlL[(timeL-1)*(timeL-6)<0]
mlL1 = mlL[(timeL-12)*(timeL-20)<0]
plt.close()
plt.hist(mlL0[(mlL0+5)*(mlL0-5)<0],np.arange(-2,5,0.1),alpha=0.3,density=True,cumulative=-1,log=True)
plt.hist(mlL1[(mlL1+5)*(mlL1-5)<0],np.arange(-2,5,0.1),alpha=0.3,density=True,cumulative=-1,log=True)
plt.savefig('%s/ml_cmp.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()

plt.close()
plt.hist2d(timeL[(mlL+2)*(mlL-5)<0],mlL[(mlL+2)*(mlL-5)<0],bins=[8,40],cmap='jet')
#plt.hist(mlL1[(mlL1+5)*(mlL1-5)<0],np.arange(-2,5,0.5))
plt.colorbar()
plt.savefig('%s/ml_2d.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()

plt.close()
plt.hist(mlL[(mlL+5)*(mlL-5)<0],np.arange(-2,5,0.1),alpha=0.3,density=True,cumulative=-1,log=True)
#plt.hist(mlL1[(mlL1+5)*(mlL1-5)<0],np.arange(-2,5,0.1),alpha=0.3,density=True,cumulative=-1,log=True)
plt.savefig('%s/ml_all.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()


detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          =workDir+'output/outputV%s/allQuake.jpg'%v_i)
detecQuake.showStaCover(aMat,staTimeML,filename=workDir+'output/outputV%s/cover.jpg'%v_i)


seism.getQuakeNoise(qL,staInfos,workDir+'output/outputV%s/'%v_i,workDir+'output/outputV%s/noiseStd'%v_i)
staIndexL,timeL,noiseL=detecQuake.loadNoiseRes(resFile=workDir+'output/outputV%s/noiseStd'%v_i)
detecQuake.plotStaNoiseDay(staIndexL,timeL,noiseL,resDir=workDir+'output/outputV%s/staNoise/'%v_i)

#按 record  load
qL.select(req={'minN':10,'minCover':0.8})
T3PSLL = [q.loadPSSacs(staInfos,matDir=workDir+'output/outputV%s/'%v_i,f=[2,8]) for q in qL]
from SeismTool.tomoDD import tomoDD