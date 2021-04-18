import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from SeismTool.MFT import pyMFTCuda
from SeismTool.detector import detecQuake
from SeismTool.io import seism
from SeismTool.locate import locate 
from obspy import UTCDateTime
from glob import glob
import torch
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
#for staInfo in staInfos[20:]:
#	staInfo['nameMode']='xxx'
for date in date0.timestamp + dateL*86400:
	dayNum=int(date/86400)
	dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
	if os.path.exists(dayDir):
		print('done')
		continue
	if count >=0:
		#staL=0
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
		#pyMFTCuda.delStaL(staL)
		#staL=[]



'''
reload(locate)
l=locate.locator(staInfos)
#template = templateL.Find(dayQuakeL[0]['tmpName'])
quakeCC,res=l.locateRef(dayQuakeL[0],template,minCC=0.2)

for sta in staL:
	count+=1
	sta.data.data = 0

torch.cuda.empty_cache()
'''
