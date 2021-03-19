import numpy as np
from SeismTool.MFT import pyMFTCuda
from SeismTool.detector import detecQuake
from SeismTool.io import seism
from SeismTool.locate import locate 
from obspy import UTCDateTime
from glob import glob
import os
import torch
def getFile(net,sta,comp,time,staDir):
	sta = sta.split('_')[0]
	pattern='%s/%s/%s.%s.00.DN%s.%s.SAC'\
	%(staDir,sta,net,sta,comp[-1],time.strftime('%Y%m%d'))
	return glob(pattern)

def getStaDirL(net,sta):
	staDirL =['/HOME/jiangyr/XA_HSR_DATA/201908DX/sac/']
	return staDirL

seism.fileP.getStaDirLFunc['nameMode']=getStaDirL
seism.fileP.getFileFunc['nameMode']=getFile

laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]
templateFile = 'phase/SCYNTomoReloc'
f_0 = [0.5,20]
f_1 = [2,10]
date0 = UTCDateTime(2019,6,17)
dateL = np.arange(-30,30)
staLstFileL=['/home/jiangyr/Surface-Wave-Dispersion/stations/XU_sel.sta','/home/jiangyr/Surface-Wave-Dispersion/stations/SCYN_withComp_ac',]
workDir='/HOME/jiangyr/detecQuake/'
templateDir = '/media/jiangyr/MSSD/output/'
v_i = '20210106CC8_2-10'
p_i = '20210106CC8_2-10V2'
deviceL=['cuda:0']
minMul=4
MINMUL=6
minChannel=6
winTime=0.4
maxThreshold=0.45
maxCC=1

staInfos=seism.StationList(staLstFileL[0])
for staLstFile in staLstFileL[1:]:
	staInfos+=seism.StationList(staLstFile)

templateL = seism.QuakeL(templateFile)
T3PSLL =seism.getT3LLQuick(templateL,staInfos,matDir=templateDir,f=f_1,batch_size=100,num_workers=6)

pyMFTCuda.preT3PSLL(T3PSLL,templateL)
quakeCCL=seism.QuakeL()

count=0
for date in date0.timestamp + dateL*86400:
	dayNum=int(date/86400)
	dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
	if os.path.exists(dayDir):
		print('done')
		continue
	if count >=0:
		#staL=0
		staL = detecQuake.getStaL(staInfos,date=date,f=f_0,isPre=False,f_new=f_1)
		bTime=pyMFTCuda.preStaL(staL,date,deviceL=deviceL)
	count+=1
	dayQuakeL=pyMFTCuda.doMFTAll(staL,T3PSLL,bTime,quakeRefL=templateL,staInfos=staInfos,minChannel=minChannel,minMul=minMul,MINMUL=MINMUL,winTime=winTime,locator=locate.locator(staInfos),maxThreshold=maxThreshold,maxCC=maxCC)
	quakeCCL+=dayQuakeL
	if len(dayQuakeL)>0:
		seism.saveSacs(staL, dayQuakeL, staInfos,\
			matDir=workDir+'output/outputV%s/'%v_i,\
			bSec=-10,eSec=40)
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
