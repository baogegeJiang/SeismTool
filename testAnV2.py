from numpy.core.arrayprint import _make_options_dict
from obspy import UTCDateTime
import os
import sys
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
from SeismTool.tomoDD import tomoDD
from microQuake import microQuake
import SeismTool.mapTool as mt 
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

qLAll = seism.QuakeL(phaseFile)
qLAll.adjustMl()
pL = qLAll.paraL(req={'minN':5,'minCover':0.8,'maxDep':50})

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
plt.hist(timeL[mlL>2],np.arange(0,24,1))
plt.savefig('%s/hl_ml.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()


mlL0 = mlL[(timeL-2)*(timeL-4)<0]
mlL1 = mlL[(timeL-9)*(timeL-11)<0]
plt.close()
plt.hist(mlL0[(mlL0+5)*(mlL0-5)<0],np.arange(-2,5,0.1),alpha=0.3,cumulative=-1,log=True)
plt.hist(mlL1[(mlL1+5)*(mlL1-5)<0],np.arange(-2,5,0.1),alpha=0.3,cumulative=-1,log=True)
plt.savefig('%s/ml_cmp.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()

plt.close()
plt.hist2d(timeL[(mlL+2)*(mlL-5)<0],mlL[(mlL+2)*(mlL-5)<0],bins=[8,40],cmap='jet')
#plt.hist(mlL1[(mlL1+5)*(mlL1-5)<0],np.arange(-2,5,0.5))
plt.colorbar()
plt.savefig('%s/ml_2d.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()

plt.close()
plt.figure(figsize=[4,4])
plt.hist(mlL[(mlL+5)*(mlL-5)<0],np.arange(-2,5,0.1),alpha=1,density=True,cumulative=-1,log=True)
plt.xlabel('Magnitude')
plt.ylabel('Density')
#plt.hist(mlL1[(mlL1+5)*(mlL1-5)<0],np.arange(-2,5,0.1),alpha=0.3,density=True,cumulative=-1,log=True)
plt.savefig('%s/ml_all.jpg'%(workDir+'output/outputV%s/'%v_i),dpi=300)
plt.close()


detecQuake.plotQuakeLDis(staInfos,qLAll,laL,loL,filename\
          =workDir+'output/outputV%s/allQuake.jpg'%v_i)
detecQuake.showStaCover(aMat,staTimeML,filename=workDir+'output/outputV%s/cover.jpg'%v_i)


seism.getQuakeNoise(qL,staInfos,workDir+'output/outputV%s/'%v_i,workDir+'output/outputV%s/noiseStd'%v_i)
staIndexL,timeL,noiseL=detecQuake.loadNoiseRes(resFile=workDir+'output/outputV%s/noiseStd'%v_i)
detecQuake.plotStaNoiseDay(staIndexL,timeL,noiseL,resDir=workDir+'output/outputV%s/staNoise/'%v_i)

#按 record  load
qL.select(req={'minN':10,'minCover':0.8,'locator':locator(staInfos),'maxRes':1.5})
T3PSLL = [q.loadPSSacs(staInfos,matDir=workDir+'output/outputV%s/'%v_i,f=[2,8]) for q in qL]
#from SeismTool.tomoDD import tomoDD
dTM = tomoDD.calDTM(qL,T3PSLL,staInfos)
tomoDD.saveDTM(dTM,workDir+'output/outputV%s/dTM'%v_i)
tomoDir = workDir+'output/outputV%s/tomoDD/input/'%v_i
if not os.path.exists(tomoDir):
	os.makedirs(tomoDir)
tomoDD.preEvent(qL,staInfos,tomoDir+'event.dat')
tomoDD.preABS(qL,staInfos,tomoDir+'ABS.dat',isNick=False)
tomoDD.preSta(staInfos,tomoDir+'station.dat',isNick=False)
tomoDD.preDTCC(qL,staInfos,dTM,maxD=0.2,minSameSta=2,minPCC=0.7,minSCC=0.6,filename=tomoDir+'dt.cc',isNick=False)
tomoDD.preMod(laL+loL,nx=16,ny=18,nz=12,filename=tomoDir+'../inversion/MOD')
qL.write(tomoDir+'../tomoQuake')
detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          =tomoDir+'../tomoQuake.jpg')
qLNew = QuakeL(tomoDD.getReloc(qL,tomoDir+'../inversion/tomoDD.reloc'))
detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          =tomoDir+'../tomoQuakeReloc.jpg')
mL=tomoDD.modeL(tomoDir)
mL.plot(tomoDir+'../resFig/')

qLAll = seism.QuakeL(phaseFile)
aLAll.adjustMl()
qLAll.select(req={'minN':6,'minCover':0.8,'maxDep':50,'locator':locator(staInfos),'maxRes':1.5})

stationPairM= seism.StationPairM(staInfos)


for q in qLAll:
	microQuake.crossByPairP33(q,staInfos,stationPairM,\
    	matDir=workDir+'output/outputV%s/'%v_i,f=[2,10],filtOrder=2,resDir=workDir+'/cross10P/',threshold=5)



from glob import glob
#tsL=stationPairM[86][66].loadTraces(workDir+'/cross/')
#stationPairM[86][66].average()
for i in range(len(staInfos)):
	for j in range(len(staInfos)):
		stationPair = stationPairM[i][j]
		if stationPair.getNum(workDir+'/cross10P/')<5:
			continue
		print(stationPair.getNum(workDir+'/cross10P/'))
		stationPair.getAverage(workDir+'/cross10P/')
for stationPairL in stationPairM:
	seism.plotStationPairL(stationPairL,resDir=workDir+'/cross10P/',parentDir=workDir+'/cross10P/',mul=5)

'''
hsr = mt.readFault('data/hsr.shape')
faults = mt.readFault('data/Chinafault_fromcjw.dat')

staLa= []
staLo=[]
for sta in staInfos:
	staLa.append(sta.loc()[0])
	staLo.append(sta.loc()[1])
#staLa,staLo = staL.loc()
plt.close()
m = basemap.Basemap(llcrnrlat=laL[0],urcrnrlat=laL[1],llcrnrlon=loL[0],\
        urcrnrlon=loL[1])
staX,staY=m(np.array(staLo),np.array(staLa))

R=laL+loL
st=m.plot(staX,staY,'b^',markersize=4,alpha=0.3)
for fault in hsr:
	if fault.inR(laL+loL):
		f=fault.plot(m,markersize=0.3,cmd='-r')

for fault in faults:
	if fault.inR(laL+loL):
		f=fault.plot(m,markersize=0.2,cmd='-k')

dD=max(int((R[1]-R[0])*10)/40,int((R[3]-R[2])*10)/40)
parallels = np.arange(int(R[0]),int(R[1]+1),dD)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(int(R[2]),int(R[3]+1),dD)
m.drawmeridians(meridians,labels=[True,False,False,True])
plt.savefig('hsr_sta.jpg',dpi=600)
'''
import os
os.environ['OPENBLAS_NUM_THREADS']='5'
import mpl_toolkits.basemap as basemap
from obspy import UTCDateTime
import os
import sys
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from SeismTool.io.seism import StationList,QuakeL
from SeismTool.locate.locate import locator
import numpy as np
from matplotlib import pyplot as plt
import SeismTool.mapTool.mapTool as mt 
import SeismTool.MFT.cudaFunc
from SeismTool.detector import detecQuake
from SeismTool.io import seism,sacTool,tool

laN=90 #subareas in latitude
loN=90
laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]
argL=[['SCYNdoV40', 'SCYNdoV40V0', '0', UTCDateTime(2014,1,1).timestamp, UTCDateTime(2015,7,1).timestamp],\
      ['SCYNdoV10', 'SCYNdoV10V1', '0', UTCDateTime(2015,7,1).timestamp, UTCDateTime(2017,1,1).timestamp],\
      ['SCYNdoV10', 'SCYNdoV10V2', '1', UTCDateTime(2017,1,1).timestamp, UTCDateTime(2018,7,1).timestamp],\
      ['SCYNdoV10', 'SCYNdoV10V3', '1', UTCDateTime(2018,7,1).timestamp, UTCDateTime(2020,1,1).timestamp],]
aMat=sacTool.areaMat(laL,loL,laN,loN)
v_i,p_i=argL[0][:2]
inputDir = '/home/jiangyr/Surface-Wave-Dispersion/accuratePickerV4/'
workDir='/HOME/jiangyr/detecQuake/'# workDir: the dir to save the results
staLstFileL=[inputDir+'../stations/XU_sel.sta',inputDir+'../stations/SCYN_withComp_ac',]
phaseFile=workDir+'phaseDir/phaseLstV%s'%p_i[:-1]+'*'
staInfos=StationList(staLstFileL[0])+StationList(staLstFileL[1])
#qLAll = QuakeL(phaseFile)
qL = QuakeL(phaseFile)

qL.select(req={'minN':10,'minCover':0.8,'locator':locator(staInfos),'maxRes':1.5})
qL.write('selectSCYN')
qLAll.write('phaseSCYN')
#qL.select(req={'minN':12,'minCover':0.9,'maxRes':1.5})
#qL.write('selectSCYNTomo')

qL = seism.QuakeL('phase/phaseSCYN')
qL.adjustMl()
aMat=sacTool.areaMat(laL,loL,laN,loN)
aMat.insert(qL)
reqMore={'minN':5,'minCover':0.8,'locator':locator(staInfos),'maxRes':1.5}
reqLess={'minN':5,'minCover':0.8,'locator':locator(staInfos),'maxRes':1.5}
qL=aMat.select(maxN=90,reqLess=reqLess,reqMore=reqMore)
qL.set('sort','time')
qL.sort()
qL.write('phase/SCYNTOMOSort')
qL=seism.QuakeL('phase/SCYNTOMOSort')
templateDir = '/media/jiangyr/MSSD/output/'

from SeismTool.io import parRead
from time import time
t0 = time()
T3PSLL =seism.getT3LLQuick(qL,staInfos,matDir=templateDir,f=[2,10],batch_size=100,num_workers=4)


t1 = time()
print(t1-t0)
R=laL+loL
R=[-90,90,-180,180]
R1 = laL+loL
from SeismTool.tomoDD import tomoDD
dTM = tomoDD.calDTMQuick(qL,T3PSLL,staInfos,minSameSta=1,num_workers=3,doCut=True)
tomoDD.saveDTM(dTM,workDir+'output/outputV%s/dTM_more'%v_i)
dTM = tomoDD.loadDTM(workDir+'output/outputVSCYNdoV40/dTM_More')
tomoDir = workDir+'output/outputVSCYNdoV40/tomoDD/input/'
if not os.path.exists(tomoDir):
	os.makedirs(tomoDir)

tomoDD.preEvent(qL,staInfos,tomoDir+'/event.dat',R=R1)
tomoDD.preABS(qL,staInfos,tomoDir+'/ABS.dat',isNick=False,R=R)
tomoDD.preSta(staInfos,tomoDir+'/station.dat',isNick=False)
tomoDD.preDTCC(qL,staInfos,dTM,maxD=0.3,minSameSta=1,minPCC=0.7,minSCC=0.7,filename=tomoDir+'/dt.cc',isNick=False,R=R,perCount=1000,minDP=0.5)
tomoDD.preMod(laL+loL,nx=20,ny=22,nz=11,filename=tomoDir+'/../inversion/MOD')
qL.write(tomoDir+'../tomoQuake')

detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          =tomoDir+'../tomoQuake.jpg')
qLNew = seism.QuakeL(tomoDD.getReloc(qL,tomoDir+'/../inversion/tomoDD.reloc'))
qL.write('phase/SCYNTOMOSort_relocV1')
'''
tomoDD.preEvent(qL,staInfos,tomoDir+'/event.dat',R=R1)
tomoDD.preABS(qL,staInfos,tomoDir+'/ABS.dat',isNick=False,R=R)
tomoDD.preSta(staInfos,tomoDir+'/station.dat',isNick=False)
tomoDD.preDTCC(qL,staInfos,dTM,maxD=0.3,minSameSta=1,minPCC=0.75,minSCC=0.75,filename=tomoDir+'/dt.cc',isNick=False,R=R,perCount=500,minDP=0.5)
'''
tomoDD.preMod(laL+loL,nx=20,ny=22,nz=11,filename=tomoDir+'/../inversion/MOD')
tomoDD.preEvent(qL,staInfos,tomoDir+'/eventReloc.dat',R=R1)
tomoDD.preABS(qL,staInfos,tomoDir+'/ABSReloc.dat',isNick=False,R=R)
tomoDD.preDTCC(qL,staInfos,dTM,maxD=0.3,minSameSta=1,minPCC=0.6,minSCC=0.6,filename=tomoDir+'/dtReloc.cc',isNick=False,R=R,perCount=500,minDP=0.5)

for i in range(len(qL)):
	q = qL[i]
	if not isinstance(qLNew.Find(q['filename']),type(None)):
		print(i)
		T3PSLLNew.append(T3PSLL[i])

T3PSLLBak = T3PSLL
qLNew1.write('phase/SCYNTomoRelocV5')
qLNew.write('phase/SCYNTomoRelocV5')
detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          =tomoDir+'../tomoQuakeReloc.jpg')
detecQuake.plotQuakeLDis([],qLNew,laL,loL,filename\
          =tomoDir+'../tomoQuakeRelocNoSta.jpg')
mL=tomoDD.model(tomoDir+'/../inversion/',qLNew,qL,isDWS=True,minDWS=10,R=R)
mL.plot(tomoDir+'/../resFigV4/',['vp'])

from SeismTool.MFT import pyMFTCuda
from SeismTool.detector import detecQuake
from SeismTool.io import seism 

templateFile = 'SCYNTomoReloc'
f_0 = [0.5,20]
f_1 = [2,8]
date = UTCDateTime(2019,6,17)

qL = QuakeL(templateFile)
T3PSLL =0
T3PSLL = [q.loadPSSacs(staInfos,matDir=workDir+'output/outputV%s/'%v_i,f=f_1) for q in qL]

staL = detecQuake.getStaL(staInfos,date=date,f=[0.5,20],isPre=False,f_new=[2,8])
pyMFTCuda.preT3PSLL(T3PSLL,qL)
bTime=pyMFTCuda.preStaL(staL,date,deviceL=['cuda:0'])
qCCL=pyMFTCuda.doMFTAll(staL,T3PSLL,bTime,quakeRefL=qL,staInfos=staInfos,minChannel=6,MINMUL=6,winTime=0.5)
timeL = UTCDateTime(2014,1,1).timestamp+np.arange(6)*86400*365
i0=0
i1=0
for i in range(5):
	for i0 in range(i0,len(qLNew)):
		if qLNew[i0]['time']>timeL[i]:
			break
	for i1 in range(i1,len(qLNew)):
		if qLNew[i1]['time']>timeL[i+1]:
			break
	if i1 == len(qLNew)-1:
		i1 +=1
	print(i0,i1)
	detecQuake.plotQuakeLDis([],qLNew[i0:i1],laL,loL,filename\
          =tomoDir+'../tomoQuakeRelocNoSta%d.jpg'%(i))







from time import perf_counter, time
t0 = time()
T3PSLL = [q.loadPSSacsQuick(staInfos,matDir='/media/jiangyr/MSSD/output/',f=[2,8]) for q in qL[:20]]
t1 = time()
T3PSLL = [q.loadPSSacs(staInfos,matDir='/media/jiangyr/MSSD/output/',f=[2,8]) for q in qL[:20]]
t2 = time()
print(t1-t0,t2-t1)






from SeismTool.io import seism
from SeismTool.mapTool import mapTool
from SeismTool.mathTool import mathFunc_bak
from SeismTool.tomoDD import tomoDD
from imp import reload
import numpy as np
from obspy import UTCDateTime
from glob import glob
import os
import torch


templateFile = 'phase/SCYNTomoRelocV2'
staLstFileL=['/home/jiangyr/Surface-Wave-Dispersion/stations/XU_sel.sta','/home/jiangyr/Surface-Wave-Dispersion/stations/SCYN_withComp_ac',]
workDir='/HOME/jiangyr/detecQuake/'
templateDir = '/media/jiangyr/MSSD/output/'
tomoDir = '/HOME/jiangyr/detecQuake/output/outputVSCYNdoV40/tomoDD/input'
staInfos=seism.StationList(staLstFileL[0])
for staLstFile in staLstFileL[1:]:
	staInfos+=seism.StationList(staLstFile)

templateL = seism.QuakeL(templateFile)
templateL.adjustMl()
pL0 = np.array([[31,101],[33.2,106],[33,106.5]])
R0=mathFunc_bak.R(pL0)

mapTool.plotDep(templateL,R0,'dep0.jpg')

pL0 = np.array([[30,102],[32,101],[35,104]])
R0=mathFunc_bak.R(pL0)

mapTool.plotDep(templateL,R0,'dep1.jpg')

pL0 = np.array([[27.8,104],[29,105],[27.8,105]])
R0=mathFunc_bak.R(pL0)

mapTool.plotDep(templateL,R0,'dep2.jpg')

pL0 = np.array([[25,102],[28,102],[28,104]])
R0=mathFunc_bak.R(pL0)

mapTool.plotDep(templateL,R0,'dep3.jpg')

p_i = '20210106CC8_2-10*'
workDir='/HOME/jiangyr/detecQuake/'
qCCL = seism.QuakeL(workDir+'phaseDir/phaseLstV%s'%p_i,Quake=seism.QuakeCC)
qCCL.adjustMl()

qCCL.write(workDir+'phaseDir/phaseLstV%sReloc'%p_i)
pL0 = np.array([[30,102],[32,101],[35,104]])
R0=mathFunc_bak.R(pL0)

mapTool.plotDep(qCCL,R0,'dep1CC.jpg')

pL0 = np.array([[27.8,104],[29,105],[27.8,105]])
R0=mathFunc_bak.R(pL0)

mapTool.plotDep(qCCL,R0,'dep2CC.jpg')

from SeismTool.locate import locate
laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]
reload(locate)
locator = locate.locator(staInfos)
qCCLNew = []
for q in qCCL:
	tmpName = q['tmpName']
	loc0 = q.loc()
	qRef = templateL.Find(tmpName)
	if isinstance(qRef,type(None)):
		continue
	locator.locateRef(q,qRef,minCC=0.27)
	qCCLNew.append(q)
	loc1  = q.loc()
	print(loc1,np.array(loc1)-np.array(loc0))
from SeismTool.detector import detecQuake
detecQuake.plotQuakeLDis([],templateL,laL,loL,filename\
          ='./withTomo.jpg')

qL = seism.QuakeL('../phaseLPickCEA')
for q in qL:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y m %d %H %M %S')


qL.write('/home/jiangyr/phaseLPick',quakeKeysIn='la lo dep time filename')

qL = seism.QuakeL('phase/phaseSCYN')
qL.adjustMl()
qCCL.write(workDir+'phaseDir/phaseLstV%sReloc'%p_i,Quake=seism.QuakeCC)
pL0 = np.array([[30,102],[33,105]])
R0=mathFunc_bak.Line(pL0,30)
mapTool.plotDepV2(templateL,R0,'dep1.jpg',vModel=mL.vp)
mapTool.plotDepV2(templateL,R0,'dep1Rela.jpg',vModel=mL.vp,isPer=True,isTopo=True)
mapTool.plotDepV2(qCCL,R0,'dep1CC.jpg',vModel=mL.vp)
mapTool.plotDepV2(qCCL,R0,'dep1RelaCC.jpg',vModel=mL.vp,isPer=True,isTopo=True)

mL=tomoDD.model(tomoDir)
qCCL=seism.Quake(workDir+'phaseDir/phaseLstV%sReloc'%p_i)
pL0 = np.array([[32,102],[30,106]])
R0=mathFunc_bak.Line(pL0,20)
mapTool.plotDepV2(templateL,R0,'dep2.jpg',vModel=mL.vp)
mapTool.plotDepV2(templateL,R0,'dep2Rela.jpg',vModel=mL.vp,isPer=True,isTopo=True)
mapTool.plotDepV2(qCCL,R0,'dep2CC.jpg',vModel=mL.vp)
mapTool.plotDepV2(qCCL,R0,'dep2RelaCC.jpg',vModel=mL.vp,isPer=True)
detecQuake.plotQuakeLDis(staInfos,templateL,laL,loL,filename\
          ='./relocWithTomo.jpg',isTopo=True)
detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          ='./allWithTomo.jpg',isTopo=True)


qL=seism.QuakeL('phase/SCYNTOMOSort')
qL.adjustMl()
qLNew = seism.QuakeL(tomoDD.getReloc(qL,tomoDir+'/../inversion/tomoDD.reloc'))
qL=seism.QuakeL('phase/SCYNTOMOSort')('phase/SCYNTOMOSort')
qLNew=seism.QuakeL('phase/SCYNTOMOSort')('phase/SCYNTomoRelocV5')
NL=np.array([[30,102],[33,102],[33,106],[30,106],[27,104.5],[24,103],[23,100],[24,99],[27,100],[30,102]])
mL=tomoDD.model(tomoDir+'/../inversion/',qLNew,[],isDWS=True,minDWS=10,R=R,vR=NL)
mL.plot(tomoDir+'/../resFigV5/',['vp','vs'])
mLSyn=tomoDD.model(tomoDir+'/../Syn/Vel/',isSyn=True,isDWS=True,minDWS=3)
mLSyn.plot(tomoDir+'/../Syn/resFigV5/',nameL=['dVp','dVs','dVpr','dVsr'],doDense=False)

p_i = '20210106CC8_2-10*'
workDir='/HOME/jiangyr/detecQuake/'
qCCL = seism.QuakeL(workDir+'phaseDir/phaseLstV%s'%p_i,Quake=seism.QuakeCC)
qCCL.adjustMl()
qCCLNew = []
count=0
for q in qCCL:
	#if q['cc']<0.27:
	#	continue
	tmpName = q['tmpName']
	loc0 = q.loc()
	if (q['cc']-q['M'])/q['S']<7.5:
		qCCLNew.append(q)
		continue
	qRef = qLNew.Find(tmpName)
	if isinstance(qRef,type(None)):
		continue
	qCCLNew.append(q)
	count+0
	locator.locateRef(q,qRef,minCC=0.27)
	
	loc1  = q.loc()
	if count%10==0:
		print(loc1,np.array(loc1)-np.array(loc0))

qCCLNew = seism.QuakeL(qCCLNew)
for q in qCCLNew:
	q['ml']+=0.26

qCCLNew.write('phase/phaseSCYNCC_reloc_adjust')

qCCLNew2 = seism.QuakeL()
for q in qCCLNew:
	if (q['cc']-q['M'])/q['S']>8:
		qCCLNew2.append(q)

qCCLNew2.write('phase/phaseSCYNCC_reloc_adjust_select')


qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)	
qCCLNew = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust',Quake=seism.QuakeCC)
qLAll = seism.QuakeL('phase/phaseSCYN')
qL=seism.QuakeL('phase/SCYNTOMOSort')
qLNew=seism.QuakeL('phase/SCYNTomoRelocV5')

qCCLNew.adjustMl()
QCCLNew2.adjustMl()
qLAll.adjustMl()
qL.adjustMl()
qLNew.adjustMl()


detecQuake.plotQuakeLDis(staInfos,qLAll,laL,loL,filename\
          ='./resFig/allQuake.jpg',isTopo=True)
detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          ='./resFig/seletcQuake.jpg',isTopo=True)
detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          ='./resFig/tomoQuake.jpg',isTopo=True)
detecQuake.plotQuakeLDis(staInfos,qCCLNew2,laL,loL,filename\
          ='./resFig/ccQuake.jpg',isTopo=True)
detecQuake.plotQuakeLDis(staInfos,qCCL,R[:2],R[2:],filename\
          ='./resFig/ccRawQuake.jpg',isTopo=True)
detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          ='./resFig/tomoQuakeWithLine.jpg',isTopo=True,rL=RL)

RL=[]
pL0 = np.array([[33,103],[30,106]])
RL.append(mathFunc_bak.Line(pL0,20,name='A'))
pL0 = np.array([[32,102],[29,105]])
RL.append(mathFunc_bak.Line(pL0,20,name='B'))
pL0 = np.array([[30,102],[33,106]])
RL.append(mathFunc_bak.Line(pL0,20,name='C'))
pL0 = np.array([[30,100],[25.5,104.5]])
RL.append(mathFunc_bak.Line(pL0,20,name='D'))
pL0 = np.array([[26.5,102],[26.5001,104.5]])
RL.append(mathFunc_bak.Line(pL0,20,name='E'))
pL0 = np.array([[28,102.7],[25,102.71]])
RL.append(mathFunc_bak.Line(pL0,20,name='F'))
for r in RL:
	mapTool.plotDepV2(qLNew,r,'resFig/dep%sRelaP.jpg'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
	mapTool.plotDepV2(qLNew,r,'resFig/dep%sRelaS.jpg'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
	#mapTool.plotDepV2(qCCLNew2,r,'resFig/dep%sRelaPCC.jpg'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
	#mapTool.plotDepV2(qCCLNew2,r,'resFig/dep%sRelaSCC.jpg'%r.name,vModel=mL.vs,isPer=True,isTopo=True)


pL = qLAll.paraL(req={})
timeL  = np.array(pL['time'])
mlL  = np.array(pL['ml'])+seism.dm
laL = np.array(pL['la'])
loL = np.array(pL['lo'])

pCCL = qCCLNew.paraL(keyL=['time','ml','dep','cc','la','lo','S','M'])
timeCCL  = np.array(pCCL['time'])
mlCCL  = np.array(pCCL['ml'])+seism.dm
mulCCL   = (np.array(pCCL['cc'])-np.array(pCCL['M']))/np.array(pCCL['S'])

timeL  = np.array(pL['time'])+8*3600
timeL -=UTCDateTime(2014,1,1).timestamp
timeL %= 86400
timeL /=3600
mlL  = np.array(pL['ml'])+seism.dm
plt.close()
fig=plt.figure(figsize=[4,3])
plt.hist(timeL,np.arange(0,24,0.3))
plt.xlabel('hour')
plt.xlim([0,24])
plt.ylabel('number')
fig.tight_layout()
plt.savefig('resFig/hl_ml.jpg',dpi=300)
plt.close()


plt.close()
fig=plt.figure(figsize=[4,3])
plt.hist(mlL,np.arange(0,7,0.1),cumulative=-1,log=True)
plt.ylabel('number')
plt.xlabel('magnitude')
fig.tight_layout()
plt.savefig('resFig/ml.jpg',dpi=300)
plt.close()

plt.close()
fig=plt.figure(figsize=[4,3])
plt.hist(mulCCL,np.arange(6,15,0.01),cumulative=-1,log=True)
plt.ylabel('number')
plt.xlabel('mul')
fig.tight_layout()
plt.savefig('resFig/mul.jpg',dpi=300)
plt.close()

plt.close()
fig=plt.figure(figsize=[4,3])
plt.hist(pCCL['cc'],np.arange(0,1,0.01),cumulative=-1,log=True)
plt.ylabel('number')
plt.xlabel('cc')
fig.tight_layout()
plt.savefig('resFig/cc.jpg',dpi=300)
plt.close()



pL1 = qLAll.paraL(req={'time0':timeCCL[0],'time1':timeCCL[-1]})
timeL1  = np.array(pL1['time'])
mlL1  = np.array(pL1['ml'])+seism.dm
pCCL2= qCCLNew2.paraL(keyL=['time','ml','dep','cc','la','lo','S','M'])
timeCCL2  = np.array(pCCL2['time'])
mlCCL2  = np.array(pCCL2['ml'])+seism.dm
mulCCL2   = (np.array(pCCL2['cc'])-np.array(pCCL2['M']))/np.array(pCCL2['S'])

plt.close()
plt.figure(figsize=[5,4])
plt.hist(mlCCL,np.arange(0,7,0.1),cumulative=-1,log=True)
plt.hist(mlL1,np.arange(0,7,0.1),cumulative=-1,log=True,alpha=0.8)
plt.ylabel('number')
plt.xlabel('magnitude')
plt.legend(['WMFT','APP'])
plt.savefig('resFig/ml_compare.jpg',dpi=300)

plt.close()
lt.close()
plt.figure(figsize=[5,4])
plt.hist(mlCCL,np.arange(0,7,0.1),cumulative=-1,log=True)
plt.hist(mlCCL2,np.arange(0,7,0.1),cumulative=-1,log=True)
plt.hist(mlL1,np.arange(0,7,0.1),cumulative=-1,log=True,alpha=0.8)
plt.ylabel('number')
plt.xlabel('magnitude')
plt.legend(['WMFT','WMFT_sel','APP++'])
plt.savefig('resFig/ml_compare_select.jpg',dpi=300)
plt.close()


timeDay = (np.array(pL1['time'])-UTCDateTime(2019,6,17).timestamp)/86400
timeDayCC = (np.array(pCCL2['time'])-UTCDateTime(2019,6,17).timestamp)/86400
plt.close()
plt.figure(figsize=[5,5])
plt.subplot(2,1,1)
plt.plot(timeDay,pL1['ml']+seism.dm,'.k',markersize=0.3)
plt.ylabel('magnitude')
plt.ylim([0,7])
plt.xlim([-30,30])
plt.xlabel('Days from 2019:06:17')
plt.legend(['APP++'])
plt.subplot(2,1,2)
plt.plot(timeDayCC,pCCL2['ml']+seism.dm,'.r',markersize=0.3)
plt.ylabel('magnitude')
plt.ylim([0,7])
plt.xlim([-30,30])
plt.xlabel('Days from 2019:06:17')
plt.legend(['WMFT_sel'])
plt.savefig('resFig/ml_day.jpg',dpi=300)
plt.close()

tmpMl=[]
ccMl=[]
for i in range(len(timeL1)):
	dTime = np.abs(timeL1[i]-timeCCL2)
	minDTime=dTime.min()
	if minDTime<4:
		index= dTime.argmin()
		if mlL1[i]<-10 or mlCCL2[index]<-10:
			continue
		tmpMl.append(mlL1[i])
		ccMl.append(mlCCL2[index])

tmpMl=np.array(tmpMl)
ccMl=np.array(ccMl)
dml=tmpMl-ccMl
dml[np.abs(dml)<2].mean()
plt.close()
plt.figure(figsize=[5,4])
plt.plot(ccMl,tmpMl-ccMl,'.k',markersize=1)
plt.ylabel('dMl')
plt.xlabel('magnitude')
plt.ylim([-2,2])
#plt.legend(['WMFT','APP'])
plt.savefig('resFig/dml_select.jpg',dpi=300)
plt.close()

pCCL = qCCLNew.paraL(req={})
xyz=np.array([pCCL['la'],pCCL['lo'],pCCL['dep']])
xyz.save('/fastDir/jiangyr/xyz.npy')
tomoDD.analyReloc(tomoDir+'/../inversion/tomoDD.reloc','resFig/uncertainty.jpg')
tomoDD.analyDTM(dTM,'resFig/dTM.jpg')

loLR = np.arange(98,108,1)
laLR = np.arange(21,35,1)
rMax=100
pointL=np.array([laL.tolist(),loL.tolist()]).transpose()
rM=[[mathFunc_bak.Round([la,lo],rMax) for lo in loLR]for la in laLR]
mlLM=mathFunc_bak.devide(rM,pointL,mlL)
bM=np.array([[mathFunc_bak.calc_B(mlL,min_num=150,min_mag=2.25,max_mag=4.2)for mlL in mlLL]for mlLL in mlLM])
mapTool.showInMap(bM[:,:,0],laLR,loLR,R,resFile='resFig/bMap.jpg',name='b')
mapTool.showInMap(bM[:,:,1],laLR,loLR,R,resFile='resFig/cMap.jpg',name='c')



for q in qLAll:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y:%m:%d-%H:%M:%S')
	q['para1']= 'event'

for q in qL:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y:%m:%d-%H:%M:%S')
	q['para1']= 'event'

for q in qLNew:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y:%m:%d-%H:%M:%S')
	q['para1']= 'event'

for q in qCCL:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y:%m:%d-%H:%M:%S')
	q['para1']= 'event'

for q in qCCLNew:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y:%m:%d-%H:%M:%S')
	q['para1']= 'event'

for q in qCCLNew2:
	q['filename']=q.name('_')
	q['para0']=obspy.UTCDateTime(q['time']).strftime('%Y:%m:%d-%H:%M:%S')
	q['para1']= 'event'

seism.QuakeL(qL).write('resFig/phaseTomo',quakeKeysIn='para1 para0 la lo dep ml')
seism.QuakeL(qLAll).write('resFig/phaseAll',quakeKeysIn='para1 para0 la lo dep ml')
seism.QuakeL(qLNew).write('resFig/phaseTomoReloc',quakeKeysIn='para1 para0 la lo dep ml')
seism.QuakeL(qCCLNew).write('resFig/phaseCC',quakeKeysIn='para1 para0 la lo dep ml cc M S')
seism.QuakeL(qCCLNew2).write('resFig/phaseCC_sel',quakeKeysIn='para1 para0 la lo dep ml cc M S')

pCount=0
sCount=0
for quake in qLAll:

from matplotlib import pyplot as plt
qL = seism.QuakeL('phase/phaseSCYN')
qL0 = seism.QuakeL('phase/catalogSCYN')
for q in qL0:
	q['time']-=3600*8

i0L,i1L,m0L,m1L=qL0.compare(qL,maxDt=120,maxDd=0.5)
m0L=np.array(m0L)
m1L=np.array(m1L)
dm=m0L-np.array(m1L)
plt.close()
plt.figure(figsize=[5,5])
plt.plot(np.array(m1L),dm,'.k')
plt.ylim([-1.5,0])
plt.xlim([0,10])
plt.savefig('resFig/dm.pdf')
#-0.76
