from cProfile import label
from glob import glob

from numpy import False_
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
from imp import reload
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import average
from plotTool import figureSet
from ..mathTool import mathFunc
from ..mapTool import mapTool as mt
from ..deepLearning import fcn
from ..io import seism
from . import DSur
from . import dispersion as d
import gc
from scipy import fftpack,interpolate
from matplotlib import colors,cm
import h5py
#主要还是需要看如何分批次更新
#提前分配矩阵可能不影响
#插值之后绘图，避免不对应
#剩最后一部分校验
#以个点为中心点，方块内按节点插值
#是z数量的1.5倍左右即可
#是否去掉SH台站
#尝试通过射线路径确定分辨率范围
#注意最浅能到多少，最多能到多少
#如何挑选与控制
#160_w+_model
#降采样
#能load 但是损失函数未load
#验证新fv
#注意画图的节点和方框对应（shading）
plt.switch_backend('agg')
orignExe='/home/jiangyr/program/fk/'
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
srcSacDirTest='/home/jiangyr/Surface-Wave-Dispersion/srcSacTest/'
T=np.array([0.5,1,2,5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250,275,300])
para={'freq'      :[1/6],'filterName':'lowpass'}
dConfig=d.config(originName='models/prem',srcSacDir=srcSacDir,\
		distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
		layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
		noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
		isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
		T=T,threshold=0.02,expnt=12,dk=0.05,\
		fok='/k',order=0,minSNR=10,isCut=False,\
		minDist=110*10,maxDist=110*170,minDDist=240,\
		maxDDist=1600,para=para,isFromO=True,removeP=True)
def saveListStr(file,strL):
	with open(file,'w+') as f:
		for STR in strL:
			f.write('%s\n'%STR)
def loadListStr(file):
	l = []
	if os.path.exists(file):
		with open(file,'r') as f:
			for line in f.readlines():
				l.append(line[:-1])
		return l
	else:
		return l
class runConfig:
	def __init__(self,para={}):
		sacPara = {#'pre_filt': (1/500, 1/350, 1/2, 1/1.5),\
				   'output':'DISP','freq':[1/240,1/6],\
					#[1/250,1/8*0+1/6]
				   'filterName':'bandpass',\
				   'corners':4,'toDisp':False,\
				   'zerophase':True,'maxA':1e15,'gaussianTail':600}
		self.para={ 'quakeFileL'  : ['phaseLPickCEA'],\
					'stationFileL': ['stations/CEA.sta_sel'],#**********'stations/CEA.sta_know_few'\
					'oRemoveL'    : [False],\
					'avgPairDirL' : ['../models/ayu/Pairs_avgpvt/'],\
					'pairDirL'    : ['../models/ayu/Pairs_pvtsel/'],\
					'minSNRL'     : [3],\
					'isByQuakeL'  : [True],\
					'remove_respL': [True],\
					'isLoadFvL'   : [False],#False********\
					'byRecordL'   : [False],
					'maxCount'    : 4096*3,\
					'randA'       : 0.02,\
					'midV'        : 4,\
					'trainDir'    : 'predict/1015_0.95_0.05_3.2_randMove_W+/',
					'resDir'      : '/fastDir/results/1015_all_V?',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
					'refModel'    : '../models/prem',\
					'sacPara'     : sacPara,\
					'dConfig'     : dConfig,\
					'perN'        : 20,\
					'minSta'      : 5,\
					'eventDir'    : '/HOME/jiangyr/eventSac/',#'/media/commonMount/data2/eventSac/',\
					'T'           : (16**np.arange(0,1.000001,1/49))*10,\
					'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
					'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,240],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
					'surPara'     : { 'nxyz':[50,75,0], 'lalo':[55,108],#[40,60,0][55,108]\
									'dlalo':[0.4,0.4], 'maxN':100,#[0.5,0.5]\
									'kmaxRc':0,'rcPerid':[],'threshold':0.01\
									,'maxIT':8,'nBatch':16,'smoothDV':10,'smoothG':20,'vR':''},\
					'runDir'      : 'DS/1026_CEA160_NE/',#_man/',\
					'gpuIndex'    : 0,\
					'gpuN'        : 1,\
					'lalo'        :[-1,180,-1,180],#[20,34,96,108][]*******,\
					'nlalo'        :[-1,-1,-1,-1],\
					'minThreshold':0.02,\
					'thresholdTrain'   :0.01,\
					'threshold'   :0.01,\
					'qcThreshold':2.5,\
					'minProb'     :0.5,\
					'minP'        :0.5,\
					'laL'         : [],\
					'loL'         : [],\
					'areasLimit'  :  3,
					'up':5}
		self.para.update(sacPara)
		self.para.update(para)
		os.environ["CUDA_VISIBLE_DEVICES"]=str(self.para['gpuIndex'])
		self.para['surPara']['nxyz'][2]=len(self.para['z'])
		self.para['surPara'].update({\
				'kmaxRc':len(self.para['tSur']),'rcPerid': self.para['tSur'].tolist()})

'''
predict/0130_0.95_0.05_3.2_randMove/resStr_220210-083338_model.h5
train&single&1.500&0.9692&0.9864&0.9777&-0.006&0.355\\[2pt]
valid&single&1.500&0.9401&0.9652&0.9525&0.003&0.445\\[2pt]
test&single&1.500&0.9432&0.9694&0.9561&0.009&0.425\\[2pt]
-----------------------------
train&average&1.500&0.9486&0.9863&0.9670&0.059&0.398\\[2pt]
valid&average&1.500&0.9737&0.9829&0.9783&0.100&0.444\\[2pt]
test&average&1.500&0.9499&0.9737&0.9617&0.080&0.480\\[2pt]
'''
def loadResData(file):
	data = []
	with open(file) as f:
		for line in f.readlines():
			if'&' not in line:
				continue
			data.append([float(tmp) for tmp in line[:-8].split('&')[2:]])
	return data
def calData(dataL):
	dataL = np.array(dataL)
	if len(dataL)==1:
		return  dataL.mean(axis=0),dataL.mean(axis=0)*0
	return dataL.mean(axis=0),dataL.std(axis=0)
def output(M,S,file,isRand='T',method='SurfNet',isStd=True):
	#SurfNet&train&T&single&1.5&0.969+0.112&0.986+0.112&0.977+0.112&-0.06+0.11&0.35+0.12\\[2pt]
	tvt=['train','valid','test']*2
	sa=['single']*3+['average']*3
	with open(file,'w+') as f:
		for i in range(6):
			if isStd:
				f.write('%s&%s&%s&%s&%3.1f&%5.3f$\pm$%5.3f&%5.3f$\pm$%5.3f&%5.3f$\pm$%5.3f&%5.2f$\pm$%4.2f&%5.2f$\pm$%4.2f\\\\[2pt]\n'%\
					(method,tvt[i],sa[i],isRand,M[i,0],M[i,1],S[i,1],M[i,2],S[i,2],M[i,3],S[i,3],M[i,4],S[i,4],M[i,5],S[i,5])
					)
			else:
				f.write('%s&%s&%s&%s&%3.1f&%5.3f&%5.3f&%5.3f&%5.2f&%5.2f\\\\[2pt]\n'%\
					(method,tvt[i],sa[i],isRand,M[i,0],M[i,1],M[i,2],M[i,3],M[i,4],M[i,5])
					)
			if i==2:
				f.write('\hline\n')


disRandA=1/15.0
disMaxR =4
delta   =1
up=2
defaultTrainSetDir ='/media/jiangyr/1TSSD/trainSet/'
defaultMatSaveDir ='/media/jiangyr/1TSSD/matDir/'
mul=4
class run:
	def __init__(self,config=runConfig(),self1 = None):
		self.config = config
		self.model  = None
		if self1 != None:
			self.corrL  =  self1.corrL
			self.corrL1 =  self1.corrL1
			self.fvD    =  self1.fvD
			self.fvDAverage = self1.fvDAverage
			self.quakes = self1.quakes
			self.stations = self1.stations
		self.config.para['runDir']='predict/'+self.config.para['resDir'].split('/')[-2]+'/'+'DS/'
	def checkDis(self):
		count=0
		for corr in self.corrL1:
			dDis0 = np.abs(corr.dis[0]-corr.dis[1])
			print(dDis0,corr.dDis,dDis0-corr.dDis)
			if np.abs(dDis0-corr.dDis).sum()>1:
				count+=1
		print('erro Count',count)
	def plotStaDis(self,isAll=False):
		para = self.config.para
		stations = self.stations
		R        = self.config.para['R']
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		if not isAll:
			plt.close()
			plt.figure(figsize=[6,4])
			m        = mt.genBaseMap(R=R)
			#https://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc
			pc=mt.plotTopo(m,R,topo='/media/jiangyr/MSSD/ETOPO1_Ice_g_gmt4NE.grd',isColorbar=False,cpt=mt.cmapETopo,vmin=-11000,vmax=8500)
			haveOne=False
			for fault in mt.faultL:
				if fault.inR(R):
					if haveOne:
						fault.plot(m,linewidth=0.75,)
					else:
						fault.plot(m,linewidth=0.75,label='fault')
						haveOne=True
			stx,sty=m(*stations.loc()[-1::-1])
			vX,vY=m(mt.volcano[:,0],mt.volcano[:,1])
			m.plot(stx,sty,'^r',label='station')
			#m.plot(vX, vY,'^r',label='volcano')
			#m.drawcoastlines(linewidth=0.8, linestyle='dashdot', color='k')
			plt.legend()
			dLa,dLo=mt.getDlaDlo(R)
			mt.plotLaLoLine(m,dLa,dLo,dashes=[3,3],color='dimgrey',linewidth=0.75)
			mt.fs.setColorbar(pc,'elevation(m)')
			plt.savefig(resDir+'staDisWithTopo.pdf')
		else:
			plt.close()
			plt.figure(figsize=[6,4])
			Rall     = [-90,90,0,360]
			m        = mt.genBaseMap(R=Rall)
			qx,qy=m(self.quakes.loL%360,self.quakes.laL)
			#https://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc
			pc=mt.plotTopo(m,Rall,topo='https://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc',isColorbar=False,cpt=mt.cmapETopo,vmin=-11000,vmax=8500,laN=2000,loN=2000)
			haveOne=False
			'''
			for fault in mt.faultL:
				if fault.inR(R):
					if haveOne:
						fault.plot(m,linewidth=0.75,)
					else:
						fault.plot(m,linewidth=0.75,label='fault')
						haveOne=True
			'''
			stx,sty=m(*stations.loc()[-1::-1])
			m.plot(qx,qy,'ok',label='earthquake',markersize=1)
			#vX,vY=m(mt.volcano[:,0],mt.volcano[:,1])
			m.plot(stx,sty,'^r',label='station',markersize=1)
			#m.plot(vX, vY,'^r',label='volcano')
			#m.drawcoastlines(linewidth=0.8, linestyle='dashdot', color='k')
			plt.legend()
			#dLa=[-60,-30,0,30,60]
			#dLo=[-120,-60,0,60,120]
			mt.plotLaLoLine(m,30,60,dashes=[3,3],color='dimgrey',linewidth=0.75)
			mt.fs.setColorbar(pc,'elevation(m)')
			plt.savefig(resDir+'staDisWithTopoAll.pdf')
	def loadCorr(self,isLoad=True,isLoadFromMat=False,isGetAverage=True,isDisQC=False,isAll=True,isSave=False,isAllTrain=True,isControl=False):
		trainSetDir = self.config.para['trainMatDir']
		config     = self.config
		corrL      = []
		stations   = seism.StationList([])
		quakes     = seism.QuakeL()
		fvDAverage = {}
		fvDAverageO = {}
		fvD        = {}
		fvD0       = {}
		fvDO       = {}
		para       = config.para
		N          = len(para['stationFileL'])
		fvDAverage['models/prem']=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		fvDAverageO['models/prem']=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		f = 1/config.para['T']
		for i in range(N):
			sta     = seism.StationList(para['stationFileL'][i])
			sta.inR(para['lalo'])
			sta.set('oRemove', para['oRemoveL'][i])
			sta.getInventory()
			stations += sta
			q       = seism.QuakeL(para['quakeFileL'][i])
			quakes  += q
			fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][i],quakeD=quakes)
			fvdCopy={}
			for key in fvd:
				fvdCopy[key]=fvd[key].copy()
			fvdO = {}
			for key in fvd:
				fvdO[key]=fvd[key].copy()
			d.replaceF(fvd,f)
			d.replaceF(fvdCopy,f)
			d.replaceF(fvdO,f)
			#q0.write('fv.quakes')
			self.q0 =q0
			if isGetAverage:
				print('get average')
				fvMGet  =d.fvD2fvM(fvd,isDouble=True)
				fvDA = d.fvM2Av(fvMGet,threshold=para['qcThreshold'],minThreshold=para['minThreshold'],minSta=para['minSta'],it=para['it'])
			else:
				print('no get average')
				fvDA    = para['dConfig'].loadNEFV(sta,fvDir=para['avgPairDirL'][i])
				d.replaceF(fvDA,f)
			notL=[]
			for key in fvDA:
				if '_' not in key:
					continue
				dis = d.keyDis(key, sta)
				if dis<para['dConfig'].minDDist or dis>para['dConfig'].maxDDist:
					notL.append(key)
			for key in notL:
				fvDA.pop(key)

			fvDAO = {}
			for key in fvDA:
				fvDAO[key]=fvDA[key].copy()

			if isDisQC:
				d.disQC(fvDA,stations,fvDAverage['models/prem'],randA=disRandA,maxR=disMaxR)

			d.qcFvD(fvDA,threshold=para['thresholdTrain'],delta=delta,stations= sta)
			d.qcFvD(fvDAO,threshold=para['thresholdTrain'],delta=delta,stations= sta)
			fvDAverage.update(fvDA)
			fvDAverageO.update(fvDAO)
			
			d.replaceByAv(fvd,fvDA,delta=delta,stations= sta,threshold=para['thresholdTrainDiff'],isControl=isControl)
			if not isControl:
				if isDisQC:
					d.disQC(fvd,stations,fvDAverage['models/prem'],randA=disRandA,maxR=disMaxR)
			d.qcFvD(fvd)
			fvD.update(fvd)
			
			#fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][i],quakeD=quakes)
			d.replaceByAv(fvdCopy,fvDA,isReplace=False,threshold=para['thresholdTrainDiff'],delta=delta,stations= sta,isControl=isControl)
			if not isControl:
				if isDisQC:
					d.disQC(fvdCopy,stations,fvDAverage['models/prem'],randA=disRandA,maxR=disMaxR)
			d.qcFvD(fvdCopy)
			fvD0.update(fvdCopy)
			
			d.replaceByAv(fvdO,fvDAO,isReplace=False,delta=delta,stations= sta,threshold=para['thresholdTrainDiff'],isControl=isControl)
			d.qcFvD(fvdO)
			fvDO.update(fvdO)

			if isLoad:
				if not isLoadFromMat:
					corrL0  = para['dConfig'].quakeCorr(q,sta,\
							byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
							minSNR=para['minSNRL'][i],isLoadFv=para['isLoadFvL'][i],\
							fvD=fvD,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],resDir=para['eventDir'],maxCount=para['maxCount'],up=para['up'])
					corrL   += corrL0
				else:
					corrL = d.corrL()
					if isAll:
						with h5py.File(para['matH5']) as h5:
							for j in range(len(sta)):
								print(j,'of',len(sta))
								for k in range(j,len(sta)):
									print(j,'of',len(sta),k)
									sta0=sta[j]['net']+'.'+sta[j]['sta']
									sta1=sta[k]['net']+'.'+sta[k]['sta']
									if sta0>sta1:
										sta1,sta0=[sta0,sta1]
									corrL.loadByPairsH5([sta0+'_'+sta1],h5)
					else:
						corrL.loadByH5(para['trainH5'])
					self.loadFvL(trainSetDir)
		
		self.fvDAverage = fvDAverage
		fvDNew =fvD
		fvD0New = fvD0
		if isLoad:
			fvDNew ={}
			fvD0New = {}
			fvDAverageNew ={'models/prem':fvDAverage['models/prem']}
			self.fvDAverage = fvDAverageNew
			fvDAverageNewCount={}
			fvDAverageNewAllCount={}
			corrL1=[]
			for corr in corrL:
				key = corr.modelFile
				name0 = key.split('_')[-2]
				name1 = key.split('_')[-1]
				modelName0='%s_%s'%(name0,name1)
				modelName1='%s_%s'%(name1,name0)
				modelName =d.keyConvert('%s_%s'%(name0,name1))
				if modelName not in fvDAverageNewAllCount:
					fvDAverageNewAllCount[modelName] = 0
				fvDAverageNewAllCount[modelName]+=1
				if key in fvD0:
					#print(modelName0)
					if modelName0 in fvDAverage or modelName1 in fvDAverage:
						if modelName0 in fvDAverage:
							modelName =modelName0
						else:
							modelName =modelName1
						if modelName in fvDAverage:
							if modelName not in fvDAverageNewCount:
								fvDAverageNewCount[modelName] = 0
							fvDAverageNewCount[modelName]+=1
							if modelName not in fvDAverageNew and fvDAverageNewCount[modelName]>=para['minSta'] and fvDAverageNewAllCount[modelName]>=8:# and (isAllTrain or fvDAverageNewCount[modelName]>=8):
								fvDAverageNew[modelName] = fvDAverage[modelName]
			for corr in corrL:
				key = corr.modelFile
				key = corr.modelFile
				if len(key.split('_'))>=2:
					name0 = key.split('_')[-2]
					name1 = key.split('_')[-1]
					modelName0 ='%s_%s'%(name0,name1)
					modelName1 ='%s_%s'%(name1,name0)
					#print(modelName0)
					if modelName0 in fvDAverageNew or modelName1 in fvDAverageNew:
						if key in fvD0 or isAllTrain:
							corrL1.append(corr)
						if key in fvD0:
							fvDNew[key]=fvD[key]
							fvD0New[key]=fvD0[key]								
			self.corrL  = d.corrL(corrL,maxCount=para['maxCount'])
			self.corrL1 = d.corrL(corrL1,maxCount=para['maxCount'])
		self.fvDAverageO ={}
		for key in fvDAverage:
			self.fvDAverageO[key]=fvDAverageO[key]
		self.fvD    = fvDNew
		self.fvD0   = fvD0New
		self.fvDO =fvDO
		self.quakes = quakes
		self.stations = stations
		if isSave:
			resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/fv/'
			resDirAv = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/fvAv/'
			d.saveFVD(self.fvD0,self.stations,self.quakes,resDir,'pair',isOverwrite=True)
			d.saveFVD(self.fvDAverage,self.stations,self.quakes,resDirAv,'NEFile',isOverwrite=True)
	def loadFvL(self,trainSetDir=defaultTrainSetDir):
		self.fvL = loadListStr(trainSetDir+'fvL')
		self.fvTrain = loadListStr(trainSetDir+'fvTrain')
		self.fvTest = loadListStr(trainSetDir+'fvTest')
		self.fvValid = loadListStr(trainSetDir+'fvValid')
	def getDisCover(self,filename='disCover'):
		fL = 1/np.array([np.float(self.config.para['T'][0]*0.9)]+(self.config.para['T'].tolist())+[np.float(self.config.para['T'][-1]*1.1)])
		minDis,maxDis,minI,maxI=d.getDisCover(self.fvDAverage,self.stations,fL)
		self.minDis = minDis
		self.maxDis = maxDis
		self.coverFL= fL
		self.minI   = minI
		self.maxI   = maxI
		self.coverR = 0.01
		data = np.array([fL,minDis,maxDis])
		np.savetxt(filename,data)
	def loadDisCover(self,filename='disCover'):
		data=np.loadtxt(filename)
		fL,minDis,maxDis=data
		minI=interpolate.interp1d(fL,minDis)
		maxI=interpolate.interp1d(fL,maxDis)
		self.minDis = minDis
		self.maxDis = maxDis
		self.minI   = minI
		self.maxI   = maxI
		self.coverR = 0.01
		self.coverFL= fL
	def train(self,isRand=True,isShuffle=False,isAverage=False):
		para    = self.config.para
		tTrain = para['T']
		if isRand:
			if isShuffle:
				fvL = []
				disL = []
				for key in self.fvDAverage:
					if '_' not in key:
						continue
					sta0,sta1 = key.split('_')
					s0        = self.stations.Find(sta0)
					s1        = self.stations.Find(sta1)
					disL.append(s0.dist(s1))
					fvL.append(key)
				disL = np.array((disL))
				indexL  = disL.argsort()
				fvTrain = []
				fvValid = []
				fvTest = []
				for i in range(len(fvL)):
					index = indexL[i]
					print(i,index,disL[index],fvL[index])
					if i%10==8:
						fvValid.append(fvL[index])
					elif i%10==9:
						fvTest.append(fvL[index])
					else:
						fvTrain.append(fvL[index])
				#random.shuffle(fvL)
				#fvN = len(fvL)
				#fvn = int(fvN/10)
				#fvTrain = fvL[fvn*2:]
				#fvTest  = fvL[fvn:fvn*2]
				#fvValid = fvL[:fvn]
				self.fvL = fvL
				self.fvTrain = fvTrain
				self.fvTest = fvTest
				self.fvValid = fvValid
				self.saveTrainSet()
			specThreshold = 0.0
			self.corrLTrain = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=self.fvTrain)
			self.corrLTest  = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=self.fvTest)
			#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
			self.corrLValid = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=self.fvValid)
			if para['up']>0:
				for corrL  in [self.corrLTrain,self.corrLValid,self.corrLTest]:
					corrL.reSetUp(para['up'])
			#corrLQuakePTest  = d.corrL(corrLQuakePNE)
			#random.shuffle(corrLQuakePTrain)
			#random.shuffle(corrLQuakePValid)
			#random.shuffle(corrLQuakePTest)
		if isAverage:
			fvD =self.fvD
		else:
			fvD =self.fvD0
		self.corrLTrain.setTimeDis(fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,randA=para['randA'],midV=para['midV'],disAmp=para['disAmp'])
		self.corrLTest.setTimeDis(fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,randA=para['randA'],midV=para['midV'],disAmp=para['disAmp'])
		self.corrLValid.setTimeDis(fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,randA=para['randA'],midV=para['midV'],disAmp=para['disAmp'])
		#self.loadModel()
		sigma = np.ones(len(self.config.para['T']))
		N =len(self.config.para['T'])
		N_5=int(N/5)
		sigma[:N_5]=1.25
		sigma[N_5:2*N_5]= 1.5
		sigma[2*N_5:3*N_5]=1.75
		sigma[3*N_5:4*N_5]=2
		sigma[4*N_5:5*N_5]=3
		self.config.sigma=sigma
		fcn.trainAndTest(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
                        outputDir=para['trainDir'],sigmaL=[sigma],tTrain=tTrain,perN=2048,count0=3,w0=1,k0=3e-3)#w0=3#4
		#fcn.trainAndTest(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   	#	outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=50,count0=200,w0=1.5)#w0=3
	def Sigma(self):
		sigma = np.ones(len(self.config.para['T']))
		N =len(self.config.para['T'])
		N_5=int(N/5)
		N_10=int(N/10)
		sigma[:N_10]        = 1.5
		sigma[1*N_10:2*N_10]   = 1.6
		sigma[2*N_10:3*N_10]   = 1.8
		sigma[3*N_10:4*N_10] = 2.0
		sigma[4*N_10:5*N_10] = 2.25
		sigma[5*N_10:6*N_10] = 2.5
		sigma[6*N_10:7*N_10] = 2.75
		sigma[7*N_10:8*N_10] = 3.0
		sigma[8*N_10:9*N_10] = 3.5
		sigma[9*N_10:10*N_10] = 4.0
		return sigma
	def trainMul(self,isRand=True,isShuffle=False,isAverage=False,isRun=True,isAll=False):
		if isAll:
			corrL = self.corrL
		else:
			corrL = self.corrL1
		para    = self.config.para
		tTrain = para['T']
		if isRand:
			if isShuffle:
				fvL = []
				disL = []
				for key in self.fvDAverage:
					if '_' not in key:
						continue
					sta0,sta1 = key.split('_')
					s0        = self.stations.Find(sta0)
					s1        = self.stations.Find(sta1)
					disL.append(s0.dist(s1))
					fvL.append(key)
				disL = np.array((disL))
				indexL  = disL.argsort()
				fvTrain = []
				fvValid = []
				fvTest = []
				for i in range(len(fvL)):
					index = indexL[i]
					print(i,index,disL[index],fvL[index])
					if i%10==8:
						fvValid.append(fvL[index])
					elif i%10==9:
						fvTest.append(fvL[index])
					else:
						fvTrain.append(fvL[index])
				#random.shuffle(fvL)
				#fvN = len(fvL)
				#fvn = int(fvN/10)
				#fvTrain = fvL[fvn*2:]
				#fvTest  = fvL[fvn:fvn*2]
				#fvValid = fvL[:fvn]
				self.fvL = fvL
				self.fvTrain = fvTrain
				self.fvTest = fvTest
				self.fvValid = fvValid
				self.saveTrainSet()
			specThreshold = 0.0
			self.corrLTrain = d.corrL(corrL,specThreshold=specThreshold,fvD=self.fvTrain)
			self.corrLTest  = d.corrL(corrL,specThreshold=specThreshold,fvD=self.fvTest)
			#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
			self.corrLValid = d.corrL(corrL,specThreshold=specThreshold,fvD=self.fvValid)
			if para['up']>0:
				for corrL  in [self.corrLTrain,self.corrLValid,self.corrLTest]:
					corrL.reSetUp(para['up'])
			#corrLQuakePTest  = d.corrL(corrLQuakePNE)
			#random.shuffle(corrLQuakePTrain)
			#random.shuffle(corrLQuakePValid)
			#random.shuffle(corrLQuakePTest)
		if isAverage:
			fvD =self.fvD
		else:
			fvD =self.fvD0
		isGuassianMove = False
		randR=0.5
		sigma=self.Sigma()
		self.config.sigma=sigma
		self.corrLTrain.setTimeDis(fvD,tTrain,sigma=sigma,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,isGuassianMove=isGuassianMove,\
			randA=para['randA'],randR=randR,midV=para['midV'],mul=para['mul'],disAmp=para['disAmp'])
		self.corrLTest.setTimeDis(fvD,tTrain,sigma=sigma,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,isGuassianMove=isGuassianMove,\
			randA=para['randA'],randR=randR,midV=para['midV'],mul=para['mul'],disAmp=para['disAmp'])
		self.corrLValid.setTimeDis(fvD,tTrain,sigma=sigma,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,isGuassianMove=isGuassianMove,\
			randA=para['randA'],randR=randR,midV=para['midV'],mul=para['mul'],disAmp=para['disAmp'])
		#self.loadModel()
		self.corrDTrain = d.corrD(self.corrLTrain)
		self.corrDTest = d.corrD(self.corrLTest)
		self.corrDValid = d.corrD(self.corrLValid)
		
		'''
		sigma[:N_5]        = 1.75
		sigma[N_5:2*N_5]   = 2.0
		sigma[2*N_5:3*N_5] = 2.25
		sigma[3*N_5:4*N_5] = 2.5
		sigma[4*N_5:5*N_5] = 3.0'''
		
		if isRun:
			self.config.para['modelFile']=fcn.trainAndTestMul(self.model,self.corrDTrain,self.corrDValid,self.corrDTest,\
							outputDir=para['trainDir'],sigmaL=[sigma],tTrain=tTrain,perN=2048,count0=3,w0=1,k0=3e-3,mul=para['mul'])#w0=3#4
		#fcn.trainAndTest(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   	#	outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=50,count0=200,w0=1.5)#w0=3
	def saveTrainSet(self,isMat=False):
		saveDir = self.config.para['trainMatDir']
		saveFile = self.config.para['trainH5']
		if isMat:
			self.corrL.saveH5(saveFile)
		saveListStr(saveDir+'fvL',self.fvL)
		saveListStr(saveDir+'fvTrain',self.fvTrain)
		saveListStr(saveDir+'fvTest',self.fvTest)
		saveListStr(saveDir+'fvValid',self.fvValid)
	def trainSq(self):
		fvL = [key for key in self.fvDAverage]
		random.shuffle(fvL)
		fvN = len(fvL)
		fvn = int(fvN/30)
		fvTrain = fvL[fvn*2:]
		fvTest  = fvL[fvn:fvn*2]
		fvVaild = fvL[:fvn]
		para    = self.config.para
		specThreshold = 0.0
		self.corrLTrain = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTrain)
		self.corrLTest  = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTest)
		#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
		self.corrLValid = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvVaild)
		#corrLQuakePTest  = d.corrL(corrLQuakePNE)
		#random.shuffle(corrLQuakePTrain)
		#random.shuffle(corrLQuakePValid)
		#random.shuffle(corrLQuakePTest)
		tTrain = para['T']
		self.corrLTrain.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,disAmp=para['disAmp'])
		self.corrLTest.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,disAmp=para['disAmp'])
		self.corrLValid.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,disAmp=para['disAmp'])
		self.corrLTrain.newCall(np.arange(10))
		print(self.corrLTrain.t0L)
		self.loadModelSq()
		fcn.trainAndTestSq(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   		outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=20,count0=20,w0=10)
	def trainDt(self):
		fvL = [key for key in self.fvDAverage]
		random.shuffle(fvL)
		fvN = len(fvL)
		fvn = int(fvN/100)
		fvTrain = fvL[fvn*2:]
		fvTest  = fvL[fvn:fvn*2]
		fvVaild = fvL[:fvn]
		para    = self.config.para
		specThreshold = 0.0
		self.corrLTrain = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTrain)
		self.corrLTest  = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTest)
		#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
		self.corrLValid = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvVaild)
		#corrLQuakePTest  = d.corrL(corrLQuakePNE)
		#random.shuffle(corrLQuakePTrain)
		#random.shuffle(corrLQuakePValid)
		#random.shuffle(corrLQuakePTest)
		tTrain = para['T']
		self.corrLTrain.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=True,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,disAmp=para['disAmp'])
		self.corrLTest.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=True,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,disAmp=para['disAmp'])
		self.corrLValid.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=True,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=False,\
		set2One=para['set2One'],move2Int=para['move2Int'],randMove=True,disAmp=para['disAmp'])
		self.corrLTrain.newCall(np.arange(10))
		print(self.corrLTrain.t0L)
		self.loadModelDt()
		fcn.trainAndTestSq(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   		outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=para['perN'],count0=20,w0=3)
	
	def loadModel(self,file=''):
		if self.model == None:
			self.model = fcn.model(channelList=[0,2,3])
		if file != '':
			self.model.load_weights(file, by_name= True)
	def loadModelUp(self,file=''):
		if self.model == None:
			fcn.defProcess()
			self.model = fcn.modelUp(channelList=[0,1],up=self.config.para['up'],mul=self.config.para['mul'],randA=self.config.para['randA'],TL=self.config.para['T'],disRandA=disRandA,disMaxR=disMaxR,maxCount=self.config.para['maxCount'],delta=delta)
		if file != '':
			self.model.load_weights(file, by_name= True)
	def loadModelSq(self,file=''):
		if self.model == None:
			self.model = fcn.modelSq(channelList=[0],maxCount=self.config.para['maxCount'])
		if file != '':
			self.model.load_weights(file, by_name= True)
	def loadModelDt(self,file=''):
		if self.model == None:
			self.model = fcn.modelDt(channelList=[0],maxCount=self.config.para['maxCount'])
		if file != '':
			self.model.load_weights(file, by_name= True)	
	def calResOneByOne(self,isLoadModel=False,isPlot=False,isRand=False):
		config     = self.config
		para       = config.para
		N          = len(para['stationFileL'])
		fvDAverage = {}
		fvDAverage[para['refModel']]=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		if 'modelFile' in para and isLoadModel:
			print('loadFile')
			self.loadModelUp(para['modelFile'])
		if isPlot:
			plotDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/plot/'
		else:
			plotDir=''
		for i in range(N):
			sta     = seism.StationList(para['stationFileL'][i])
			sta.inR(para['lalo'])
			print('sta num:',len(sta))
			sta.set('oRemove', para['oRemoveL'][i])
			#sta.set('oRemove',isORemove)
			sta.getInventory()
			q       = seism.QuakeL(para['quakeFileL'][i])
			print(para['quakeFileL'][i],len(q))
			self.stations = sta
			self.q=q
			q.set('sort','sta')
			q.sort()
			perN= self.config.para['perN']
			fvDAverage={}
			fvDAverage[para['refModel']]=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
			for j in range(self.config.para['gpuIndex'],int(len(q)/perN)+1,self.config.para['gpuN']):#self.config.para['gpuIndex']
				print('doing for %d %d in %d'%(j*perN,min(len(q)-1,j*perN+perN),len(q)))
				corrL0  = para['dConfig'].quakeCorr(q[j*perN:min(len(q)-1,j*perN+perN)],sta,byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],minSNR=para['minSNRL'][i],isLoadFv=False,fvD=fvDAverage,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],resDir=para['eventDir'],maxCount=para['maxCount'],plotDir=plotDir)
				self.corrL  = d.corrL(corrL0,maxCount=para['maxCount'])
				if len(self.corrL)==0:
					continue
				self.corrL.reSetUp(up=para['up'])
				#self.calRes()
				para = self.config.para
				self.corrL.setTimeDis(fvDAverage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
							byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
							set2One=para['set2One'],move2Int=para['move2Int'],modelNameO=para['refModel'],noY=True,randMove=isRand,randA=para['randA'],disAmp=para['disAmp'])
				self.corrL.getAndSaveOld(self.model,'%s/CEA_P_'%para['resDir'],self.stations,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'],)
				corrL0 = 0
				self.corrL = 0
				#self.corrL.save(para['matSaveDir'])
				gc.collect()
				
	def calCorrOneByOne(self,isPlot=False):
		config     = self.config
		para       = config.para
		N          = len(para['stationFileL'])
		if isPlot:
			plotDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/plot/'
		else:
			plotDir=''
		with h5py.File(para['matH5'],'a') as h5:
			for i in range(N):
				sta     = seism.StationList(para['stationFileL'][i])
				sta.inR(para['lalo'])
				print('sta num:',len(sta))
				sta.set('oRemove', para['oRemoveL'][i])
				#sta.set('oRemove',isORemove)
				sta.getInventory()
				q       = seism.QuakeL(para['quakeFileL'][i])
				print(para['quakeFileL'][i],len(q))
				self.stations = sta
				self.q=q
				q.set('sort','sta')
				q.sort()
				perN= self.config.para['perN']
				fvDAverage={}
				fvDAverage[para['refModel']]=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
				for j in range(self.config.para['gpuIndex'],int(len(q)/perN)+1,self.config.para['gpuN']):#self.config.para['gpuIndex']
					print('doing for %d %d in %d'%(j*perN,min(len(q)-1,j*perN+perN),len(q)))
					corrL0  = para['dConfig'].quakeCorr(q[j*perN:min(len(q)-1,j*perN+perN)],sta,\
							byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
							minSNR=para['minSNRL'][i],isLoadFv=False,\
							fvD=fvDAverage,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],\
							resDir=para['eventDir'],maxCount=para['maxCount'],plotDir=plotDir)
					self.corrL  = d.corrL(corrL0,maxCount=para['maxCount'])
					if len(self.corrL)==0:
						continue
					print(self.corrL[0].name0)
					self.corrL.reSetUp(up=para['up'])
					#self.calRes()
					self.corrL.saveH5(h5)
					self.corrL = 0
					gc.collect()
	def calFromCorr(self,isLoadModel=False,M=6000,isRand=False):
		config     = self.config
		para       = config.para
		N          = len(para['stationFileL'])
		matDir     = para['matDir']
		if 'modelFile' in para and isLoadModel:
			print('loadFile')
			self.loadModelUp(para['modelFile'])
		for i in range(N):
			with h5py.File(para['matH5']) as h5:
				sta     = seism.StationList(para['stationFileL'][i])
				sta.inR(para['lalo'])
				print('sta num:',len(sta))
				sta.set('oRemove', para['oRemoveL'][i])
				sta.getInventory()
				q       = seism.QuakeL(para['quakeFileL'][i])
				print(para['quakeFileL'][i],len(q))
				self.stations = sta
				self.q=q
				q.set('sort','sta')
				q.sort()
				perN= self.config.para['perN']
				corrL = d.corrL()
				fvDAverage={}
				fvDAverage[para['refModel']]=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
				for j in range(len(sta)):
					print(j,'of',len(sta))
					for k in range(j,len(sta)):
						print(j,'of',len(sta),k)
						sta0=sta[j]['net']+'.'+sta[j]['sta']
						sta1=sta[k]['net']+'.'+sta[k]['sta']
						if sta0>sta1:
							sta1,sta0=[sta0,sta1]
						dist = sta[j].dist(sta[k])
						if dist<para['dConfig'].minDDist or dist>para['dConfig'].maxDDist:
							continue
						corrL.loadByPairsH5([sta0+'_'+sta1],h5)
						if len(corrL)>M or (j==len(sta)-2 and k==j+1 and len(corrL)>0 ):
							corrL.reSetUp(up=para['up'])
							corrL.setTimeDis(fvDAverage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
							byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
							set2One=para['set2One'],move2Int=para['move2Int'],modelNameO=para['refModel'],noY=True,randMove=isRand,randA=para['randA'],disAmp=para['disAmp'])
							corrD = d.corrD(corrL)
							print('predicting')
							corrD.getAndSaveOld(self.model,'%s/CEA_P_'%para['resDir'],self.stations\
							,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'],mul=para['mul'])
							print('predicted')
							corrD.corrL=0
							corrD=0
							corrL = d.corrL()
						gc.collect()
	def calFromCorrL(self,isRand=False):
		para = self.config.para
		fvDAverage={}
		fvDAverage[para['refModel']]=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		self.corrL.setTimeDis(fvDAverage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
		set2One=para['set2One'],move2Int=para['move2Int'],modelNameO=para['refModel'],noY=True,randMove=isRand,randA=para['randA'],disAmp=para['disAmp'])
		self.corrL.reSetUp(up=para['up'])
		corrD = d.corrD(self.corrL)
		print('predicting')
		corrD.getAndSaveOldPer(self.model,'%s/CEA_P_'%para['resDir'],self.stations\
		,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'],mul=para['mul'])
	def calRes(self):
		para = self.config.para
		self.corrL1.setTimeDis(self.fvDAverage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
		set2One=para['set2One'],move2Int=para['move2Int'],modelNameO=para['refModel'],noY=True,disAmp=para['disAmp'])
		self.corrL1.getAndSaveOld(self.model,'%s/CEA_P_'%para['resDir'],self.stations\
		,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'])
		#print(self.corrL.t0L)
	def loadRes(self,isCoverQC=False,isDisQC=False,isGetQuake=True,isCheck=False):
		stations = []
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		self.stations = seism.StationList(stations)
		self.stations.inR(self.config.para['lalo'])
		print(len(self.stations))
		para    = self.config.para
		q = seism.QuakeL(self.config.para['quakeFileL'][0])
		fvDGet,quakesGet = para['dConfig'].loadQuakeNEFV(self.stations,quakeFvDir=para['resDir'],quakeD=q,isGetQuake=isGetQuake,isCheck=isCheck)
		self.fvDGet  = fvDGet
		
		if isCoverQC:
			self.coverQC(self.fvDGet)
		if isDisQC:
			fvRef=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
			d.disQC(self.fvDGet,self.stations,fvRef,randA=disRandA,maxR=disMaxR)
		d.qcFvD(self.fvDGet)
		#self.getAv()
	def loadResAv(self):
		stations = []
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		self.stations = seism.StationList(stations)
		self.stations.inR(self.config.para['lalo'])
		self.stations.notInR(self.config.para['nlalo'])
		print(len(self.stations))
		para    = self.config.para
		self.fvAvGet,self.quakesGet = para['dConfig'].loadQuakeNEFVAv(self.stations,quakeFvDir=para['resDir'],threshold=para['threshold'],minP=para['minP'],minSta=para['minSta'])
		#for fv in self.fvAvGet:
		#	self.fvAvGet[fv].qc(threshold=para['threshold'])
		d.qcFvD(self.fvAvGet,threshold=para['threshold'])
	def loadAv(self,fvDir ='models/all/',mode='NEFileNew'):
		stations = []
		para    = self.config.para
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		stations = seism.StationList(stations)
		stations.inR(para['lalo'])
		stations.notInR(self.config.para['nlalo'])
		self.stations = seism.StationList(stations)
		self.fvAvGet = para['dConfig'].loadNEFV(stations,fvDir=fvDir,mode=mode)
		#for fv in self.fvAvGet:
		#	self.fvAvGet[fv].qc(threshold=para['threshold'])
		d.qcFvD(self.fvAvGet,threshold=self.config.para['threshold'])
	def coverQC(self,fvD):
		d.coverQC(fvD,self.stations,self.minI,self.maxI,R=self.coverR)
	def getAv(self,isMinP=True,isDisQC=False,isCoverQC=False,delta=1,isWeight=False,weightType='std'):
		if isMinP:
			d.qcFvD(self.fvDGet,threshold=-self.config.para['minP'])
		para = self.config.para
		self.fvMGet  =d.fvD2fvM(self.fvDGet,isDouble=True)
		#print(self.fvMGet)qcThreshold
		self.fvAvGet = d.fvM2Av(self.fvMGet,threshold=para['qcThreshold'],minThreshold=para['minThreshold'],minSta=para['minSta'],isWeight=isWeight,weightType=weightType,it=para['it'])
		#print(self.fvAvGet)
		fvRef=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		#d.disQC(fvDA,stations,fvDAverage['models/prem'],randA=0.1)
		if isDisQC:
			d.disQC(self.fvAvGet,self.stations,fvRef,randA=disRandA,maxR=disMaxR)
		if isCoverQC:
			self.coverQC(self.fvAvGet)
		d.qcFvD(self.fvAvGet,threshold=self.config.para['threshold'],delta=1,stations=self.stations)
		self.fvAvGetO={}
		for key in self.fvAvGet:
			self.fvAvGetO[key]=self.fvAvGet[key].copy()
	def showTest(self):
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/waveform/'
		corrD = d.corrD(self.corrDTest.corrL)
		mul   = self.config.para['mul']
		T     = self.config.para['T']
		x,y0,t= corrD(mul=mul,N=mul,isRand=False)
		y =  self.model.predict(x)
		x   =  self.model.inx(x)
		iL  = np.array(corrD.corrL.iL).reshape([-1,mul])
		d.showCorrD(x,y0,y,t,iL,corrD.corrL,resDir,T,mul=mul,number=3)
		#d.showCorrD(x,y0,y,t,iL,corrD.corrL,resDir,T,mul=mul,number=3)
	def showSigma(self,format='eps'):
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		plt.figure(figsize=[2,2])
		f = 1/self.config.para['T']
		plt.plot(f,self.config.sigma,'-ok',markersize=1,linewidth=0.5)
		plt.gca().set_position([0.2,0.2,0.6,0.6])
		#plt.title('$\sigma$(f))')
		plt.xlabel('$f$(Hz)')
		plt.ylabel('$\sigma$(s)')
		#plt.tight_layout()
		#plt.xlim([1/self.config.para['T'].min()*1.1,1/self.config.para['T'].max()*0.9])
		plt.semilogx()
		plt.savefig(resDir+'sigmaDis.%s'%(format),dpi=300)
	def analyRes(self,threshold=0.015,format='eps',isAverage=False):
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		if not os.path.exists(resDir):
			os.makedirs(resDir)
		if isAverage:
			fvD = self.fvD
		else:
			fvD = self.fvD0
		plt.figure(figsize=[4,4])
		corrD = d.corrD(self.corrL1)
		for key in corrD:
			dis = d.keyDis(key,self.stations)
			plt.plot(dis,len(corrD[key]),'.k')
		plt.title('count(distance))')
		plt.xlabel('dis/km')
		plt.ylabel('count')
		plt.semilogy()
		plt.plot([0,2000],[6,6],'r')
		plt.plot([0,2000],[4,4],'b')
		plt.savefig(resDir+'count_distance_%.3f_%.3f.%s'%(threshold,self.config.para['threshold'],format),dpi=300)
		plt.close()
		self.showSigma(format=format)
		fStrike=2
		averageTrain=d.compareFVD(self.fvDAverage,self.fvAvGet,self.stations,resDir+'erroTrain_%.3f_%.3f.%s'%(threshold,self.config.para['threshold'],format),t=self.config.para['T'],keys=self.fvTrain,fStrike=fStrike,title='E. D. on Train',thresholdForGet=self.config.para['threshold'])
		singleTrain=d.compareFVD(fvD,self.fvDGet,self.stations,resDir+'erroTrainSingle_%.3f_%.3f.%s'%(threshold,self.config.para['minP'],format),t=self.config.para['T'],keys=self.fvTrain,fStrike=fStrike,title='E. D. S. on Train',thresholdForGet=-self.config.para['minP'],isCount=False,plotRate=0.6,fvRef=self.fvDAverage['models/prem'])

		averageValid=d.compareFVD(self.fvDAverage,self.fvAvGet,self.stations,resDir+'erroValid_%.3f_%.3f.%s'%(threshold,self.config.para['threshold'],format),t=self.config.para['T'],keys=self.fvValid,fStrike=fStrike,title='E. D. on Valid',thresholdForGet=self.config.para['threshold'])
		singleValid=d.compareFVD(fvD,self.fvDGet,self.stations,resDir+'erroValidSingle_%.3f_%.3f.%s'%(threshold,self.config.para['minP'],format),t=self.config.para['T'],keys=self.fvValid,fStrike=fStrike,title='E. D. S. on Valid',thresholdForGet=-self.config.para['minP'],isCount=False,plotRate=0.6,fvRef=self.fvDAverage['models/prem'])

		averageTest=d.compareFVD(self.fvDAverage,self.fvAvGet,self.stations,resDir+'erroTest_%.3f_%.3f.%s'%(threshold,self.config.para['threshold'],format),t=self.config.para['T'],keys=self.fvTest,fStrike=fStrike,title='E. D. on test',threshold=threshold,thresholdForGet=self.config.para['threshold'])
		singleTest=d.compareFVD(fvD,self.fvDGet,self.stations,resDir+'erroTestSingle_%.3f_%.3f.%s'%(threshold,self.config.para['minP'],format),t=self.config.para['T'],keys=self.fvTest,fStrike=fStrike,title='E. D. S. on test',thresholdForGet=-self.config.para['minP'])
		with open(resDir+'resOnTrainTestValid','w') as f:
			f.write('%s\n'%self.config.para['modelFile'])
			f.write('%s&single&%s\\\\[2pt]\n'%('train',singleTrain))
			f.write('%s&single&%s\\\\[2pt]\n'%('valid',singleValid))
			f.write('%s&single&%s\\\\[2pt]\n'%('test',singleTest))
			f.write('-----------------------------\n')
			f.write('%s&average&%s\\\\[2pt]\n'%('train',averageTrain))
			f.write('%s&average&%s\\\\[2pt]\n'%('valid',averageValid))
			f.write('%s&average&%s\\\\[2pt]\n'%('test',averageTest))

		#M,V0,V1=d.compareInF(self.fvDAverage,self.fvAvGetO,self.stations,1/self.config.para['T'],R=self.config.para['R'],saveDir=resDir+'compareInF/')
		d.plotFVM(self.fvMGet,self.fvAvGet,self.fvDAverage,resDir=resDir+'/compare/',isDouble=True,fL0=1/self.config.para['T'],format=format,stations=self.stations,keyL=self.fvTest)
	def getAV(self):
		self.fvAvGetL = [self.fvAvGetO[key] for key in self.fvAvGetO]
		self.FVAV     = d.averageFVL(self.fvAvGetL)
	def limit(self,):
		for key in self.fvAvGet:
			self.fvAvGet[key] = self.fvAvGetO[key].copy()
			self.FVAV.limit(self.fvAvGet[key],threshold=self.config.para['areasLimit'])
		d.qcFvD(self.fvAvGet)
	def getAreas(self):
		self.areas=d.areas(laL=self.config.para['laL'],\
			loL=self.config.para['loL'],stations=self.stations)
	def areasLimit(self):
		#self.areas = self.getAreas()
		self.areas.Insert(self.fvAvGet)
		self.areas.getAv()
		self.areas.limit(self.fvAvGet,threshold=self.config.para['areasLimit'])
		d.qcFvD(self.fvAvGet)
	def preDS(self,do=True,isByTrain=False,threshold=-1):
		self.config.para['runDir']='predict/'+self.config.para['resDir'].split('/')[-2]+'/'+'DS/'
		para    = self.config.para
		tSur = para['tSur']
		z= para['z'];surPara= para['surPara'];DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir'])
		self.DS = DS
		if not isByTrain:
			fvAvGet=self.fvAvGet
		else:
			keyL = self.fvDAverage.keys()
			print('by keyL')
			fvAvGet={}
			for key in keyL:
				if '_' not in key:
					continue
				name0 = key
				sta0,sta1 = key.split('_')
				name1 = '%s_%s'%(sta1,sta0)
				if name0 in self.fvAvGet:
					fvAvGet[name0]=self.fvAvGet[name0]
				if name1 in self.fvAvGet:
					fvAvGet[name1]=self.fvAvGet[name1]
		if do:
			indexL,vL = d.fvD2fvL(fvAvGet,self.stations,1/tSur,threshold=threshold)
			self.indexL = indexL
			self.vL   = vL
			DS.test(vL,indexL,self.stations)
	def preDSRef(self,do=True):
		para    = self.config.para
		self.config.para['runDir']='predict/'+self.config.para['resDir'].split('/')[-2]+'/'+'DS/'
		tSur = para['tSur']
		z= para['z'];surPara= para['surPara'];DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'ref/')
		self.DSRef = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations[::3],1/tSur,isRef=True,fvRef=self.FVAV)
			self.indexL = indexL
			self.vL   = vL
			DS.test(vL,indexL,self.stations[::3])
	def preDSTrain(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z'];surPara= para['surPara'];DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'train/')
		self.DSTrain = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvDAverage,self.stations,1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.test(vL,indexL,self.stations)
	def loadAndPlot(self,DS=[],isPlot=True):
		if isinstance(DS,list):
			DS = self.DS
		DS.loadRes()
		if isPlot:
			DS.plotByZ(p2L=self.config.para['p2L'],R=self.config.para['R'])
	def compare(self,DS,DS0,isCompare=False):
		DS.plotByZ(p2L=self.config.para['p2L'],R=self.config.para['R'],self1=DS0,isCompare=isCompare)
	def preDSSyn(self,do=True,isByTrain=False,M=1):
		self.config.para['runDir']='predict/'+self.config.para['resDir'].split('/')[-2]+'/'+'DS/'
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'syn/',mode='syn')
		self.DSSyn = DS
		if not isByTrain:
			fvAvGet=self.fvAvGet
		else:
			keyL = self.fvDAverage.keys()
			print('by keyL')
			fvAvGet={}
			for key in keyL:
				if '_' not in key:
					continue
				name0 = key
				sta0,sta1 = key.split('_')
				name1 = '%s_%s'%(sta1,sta0)
				if name0 in self.fvAvGet:
					fvAvGet[name0]=self.fvAvGet[name0]
				if name1 in self.fvAvGet:
					fvAvGet[name1]=self.fvAvGet[name1]
		if do:
			indexL,vL = d.fvD2fvL(fvAvGet,self.stations[-1::-1],1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.testSyn(vL,indexL,self.stations[-1::-1],M=1)
	def plotTrainDis(self,isAverage=False):
		para = self.config.para
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		if not os.path.exists(resDir):
			os.makedirs(resDir)
		if isAverage:
			fvD = self.fvD
		else:
			fvD = self.fvD0
		fvAverage = d.averageFVL([self.fvDAverage[key] for key in self.fvDAverage],fL=1/self.config.para['T'])
		thresL=[0.015]
		#return fvAverage
		print(fvAverage.f,fvAverage.f)
		with open(resDir+'dataSetting','w') as f:
			f.write('single setting\n')
			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvD0,self.stations,t=self.config.para['T'],keys=self.fvTrain)
			f.write('train %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.corrDTrain.corrL)-len(vL)))
			d.plotFvDist(disL,vL,fL,resDir+'fvDistTrainSingle.eps',isCover=True,minDis=self.minDis,maxDis=self.maxDis,fLDis=self.coverFL,R=self.coverR)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvD0,self.stations,t=self.config.para['T'],keys=self.fvValid)
			f.write('valid %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.corrDValid.corrL)-len(vL)))
			d.plotFvDist(disL,vL,fL,resDir+'fvDistValidSingle.eps',isCover=True,minDis=self.minDis,maxDis=self.maxDis,fLDis=self.coverFL,R=self.coverR)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvD0,self.stations,t=self.config.para['T'],keys=self.fvTest)
			f.write('Test %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.corrDTest.corrL)-len(vL)))
			d.plotFvDist(disL,vL,fL,resDir+'fvDistTestSingle.eps',isCover=True,minDis=self.minDis,maxDis=self.maxDis,fLDis=self.coverFL,R=self.coverR)

			f.write('average setting\n')
			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDAverage,self.stations,t=self.config.para['T'],keys=self.fvTrain)
			f.write('train %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.fvTrain)))
			d.plotFV(vL,fL,resDir+'FVTrain.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTrainRand.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDAverage,self.stations,t=self.config.para['T'],keys=self.fvValid)
			f.write('valid %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.fvValid)))
			d.plotFV(vL,fL,resDir+'FVValid.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVValidRand.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_ ,keyL= d.outputFvDist(self.fvDAverage,self.stations,t=self.config.para['T'],keys=self.fvTest)
			f.write('test %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.fvTest)))
			d.plotFV(vL,fL,resDir+'FVTest.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTestRand.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_ ,keyL= d.outputFvDist(fvD,self.stations,t=self.config.para['T'],keys=self.fvTrain)
			d.plotFV(vL,fL,resDir+'FVTrainSingle.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTrainSingleRand.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(fvD,self.stations,t=self.config.para['T'],keys=self.fvTest)
			d.plotFV(vL,fL,resDir+'FVTestSingle.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTestSingleRand.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(fvD,self.stations,t=self.config.para['T'],keys=self.fvValid)
			d.plotFV(vL,fL,resDir+'FVValidSingle.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVValidSingleRand.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)
	def plotGetDis(self):
		para = self.config.para
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		if not os.path.exists(resDir):
			os.makedirs(resDir)
		fvAverage = d.averageFVL([self.fvDAverage[key] for key in self.fvDAverage],fL=1/self.config.para['T'])
		thresL=[0.015]
		#return fvAverage
		print(fvAverage.f,fvAverage.f)
		with open(resDir+'getSetting','w') as f:
			f.write('single setting\n')
			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDGet,self.stations,t=self.config.para['T'],keys=self.fvTrain)
			f.write('train %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.corrDTrain.corrL)-len(vL)))
			d.plotFvDist(disL,vL,fL,resDir+'fvDistTrainSingleGet.eps',isCover=True,minDis=self.minDis,maxDis=self.maxDis,fLDis=self.coverFL,R=self.coverR)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDGet,self.stations,t=self.config.para['T'],keys=self.fvValid)
			f.write('valid %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.corrDValid.corrL)-len(vL)))
			d.plotFvDist(disL,vL,fL,resDir+'fvDistValidSingleGet.eps',isCover=True,minDis=self.minDis,maxDis=self.maxDis,fLDis=self.coverFL,R=self.coverR)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDGet,self.stations,t=self.config.para['T'],keys=self.fvTest)
			f.write('Test %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.corrDTest.corrL)-len(vL)))
			d.plotFvDist(disL,vL,fL,resDir+'fvDistTestSingleGet.eps',isCover=True,minDis=self.minDis,maxDis=self.maxDis,fLDis=self.coverFL,R=self.coverR)

			f.write('average setting\n')
			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvAvGet,self.stations,t=self.config.para['T'],keys=self.fvTrain)
			f.write('train %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.fvTrain)))
			d.plotFV(vL,fL,resDir+'FVTrainGet.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTrainRandGet.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvAvGet,self.stations,t=self.config.para['T'],keys=self.fvValid)
			f.write('valid %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.fvValid)))
			d.plotFV(vL,fL,resDir+'FVValidGet.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVValidRandGet.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_ ,keyL= d.outputFvDist(self.fvAvGet,self.stations,t=self.config.para['T'],keys=self.fvTest)
			f.write('test %d %d %d\n'%(len(vL),(vL>1).sum(),len(self.fvTest)))
			d.plotFV(vL,fL,resDir+'FVTestGet.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTestRandGet.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_ ,keyL= d.outputFvDist(self.fvDGet,self.stations,t=self.config.para['T'],keys=self.fvTrain)
			d.plotFV(vL,fL,resDir+'FVTrainSingleGet.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTrainSingleRandGet.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDGet,self.stations,t=self.config.para['T'],keys=self.fvTest)
			d.plotFV(vL,fL,resDir+'FVTestSingleGet.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVTestSingleRandGet.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)

			disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvDGet,self.stations,t=self.config.para['T'],keys=self.fvValid)
			d.plotFV(vL,fL,resDir+'FVValidSingleGet.eps',isAverage=True,fvAverage=fvAverage,thresL=thresL)
			d.plotFV(vL,fL,resDir+'FVValidSingleRandGet.eps',isAverage=True,fvAverage=fvAverage,isRand=True,randA=para['randA'],midV=para['midV'],randN=2,randR=0.5,thresL=thresL)
	def plotGetAvDis(self):
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		#fvAverage = self.fvDAverage['models/prem']
		disL,vL,fL,fvAverage_,keyL = d.outputFvDist(self.fvAvGetO,self.stations,t=self.config.para['T'],keys=[])
		d.plotFV(vL,fL,resDir+'FVGet.eps',isAverage=True,fvAverage=self.FVAV,thresL=[self.config.para['areasLimit']])
	def plotDVK(self,fvD,format='jpg',head='',fvD0={},isRight=True,isPlot=True):
		f = 1/self.config.para['T']
		DV,K,V,SNR = d.getDVK(fvD,f,fvD0=fvD0,isRight=isRight)
		resDir = 'predict/'+self.config.para['resDir'].split('/')[-2]+'/'
		if not os.path.exists(resDir):
			os.makedirs(resDir)
		para = self.config.para
		fvRef=self.fvDAverage[para['refModel']]
		vRef = fvRef(f)
		DVRef,KRef = fvRef.getDVK(f)
		per = V/vRef.reshape([1,-1])-1
		self.minDV = d.defineEdge(DV,0.025)
		self.maxDV = d.defineEdge(DV,1-0.025)
		self.maxK = d.defineEdge(K,1-0.05)
		self.minPer = d.defineEdge(per,0.025,-0.5)
		self.maxPer = d.defineEdge(per,1-0.025,-0.5)
		self.minSNR = d.defineEdge(SNR,0.05)
		if not isPlot:
			return
		tail='_all'
		if len(fvD0)>0:
			if isRight:
				tail='right'
			else:
				tail= 'wrong'
		head = head+ tail
		dvBins = np.arange(-2,2,0.025)
		KBins =np.arange(0,15,0.2)
		snrBins =10**(np.arange(-1.5,2,0.05))
		fBins =f.copy()
		fBins.sort()
		perBins = np.arange(-0.3,0.3,0.01)
		F  = f.reshape([1,-1])+DV*0
		plt.close()
		plt.figure(figsize=(4,3))
		plt.hist2d(DV[DV>-100],F[DV>-100],cmap='gray_r', norm=(colors.LogNorm()),bins=(dvBins,fBins))
		plt.gca().set_position([0.15,0.15,0.7,0.7])
		if False:#isRight:
			plt.plot(self.minDV,f,'b',linewidth=0.5)
			plt.plot(self.maxDV,f,'b',linewidth=0.5,label='95%')
		plt.plot(DVRef,f,'r',linewidth=0.5,label='ref')
		plt.plot(np.array(DVRef)-para['dV'],f,'--r',linewidth=0.5,label='$\pm %.1f$'%para['dV'])
		plt.plot(np.array(DVRef)+para['dV'],f,'--r',linewidth=0.5)
		plt.gca().set_yscale('log')
		plt.legend()
		plt.xlabel('$v$\'')
		plt.ylabel('$f$(Hz)')
		ax=plt.gca()
		ax_divider = make_axes_locatable(plt.gca())
		cax = ax_divider.append_axes('right', size="7%", pad="10%",)
		plt.colorbar(cax=cax,label='count')
		plt.savefig(resDir+'dv'+head+'.'+format,dpi=300)
		plt.close()

		plt.figure(figsize=(4,3))
		plt.hist2d(K[K>-100],F[K>-100],cmap='gray_r', norm=(colors.LogNorm()),bins=(KBins,fBins))
		plt.gca().set_position([0.15,0.15,0.7,0.7])
		if isRight:
			plt.plot(self.maxK,f,'b',linewidth=0.5,label='$K_{0.95}$')
		plt.plot(KRef,f,'r',linewidth=0.5,label='ref')
		plt.gca().set_yscale('log')
		plt.xlim([-0.1,KBins.max()])
		plt.xlabel('$K$')
		plt.ylabel('$f$(Hz)')
		plt.legend()
		ax=plt.gca()
		ax_divider = make_axes_locatable(plt.gca())
		cax = ax_divider.append_axes('right', size="7%", pad="10%",)
		plt.colorbar(cax=cax,label='count')
		plt.savefig(resDir+'K'+head+'.'+format,dpi=300)
		plt.close()

		plt.figure(figsize=(4,3))
		plt.hist2d(SNR[SNR>-100],F[SNR>-100],cmap='gray_r', norm=(colors.LogNorm()),bins=(snrBins,fBins))
		plt.gca().set_position([0.15,0.15,0.7,0.7])
		#if isRight:
		plt.plot(self.minSNR,f,'b',linewidth=0.5,label='$SNR_{0.95}$')
		#plt.plot(KRef,f,'r',linewidth=0.5,label='ref')
		plt.gca().set_xscale('log')
		plt.gca().set_yscale('log')
		plt.xlabel('$SNR$')
		plt.ylabel('$f$(Hz)')
		#if isRight:
		plt.legend()
		ax=plt.gca()
		ax_divider = make_axes_locatable(plt.gca())
		cax = ax_divider.append_axes('right', size="7%", pad="10%",)
		plt.colorbar(cax=cax,label='count')
		plt.savefig(resDir+'SNR'+head+'.'+format,dpi=300)
		plt.close()

		plt.figure(figsize=(4,3))
		plt.hist2d(per[per>-0.5],F[per>-0.5],cmap='gray_r', norm=(colors.LogNorm()),bins=(perBins,fBins))
		plt.gca().set_position([0.15,0.15,0.7,0.7])
		#plt.plot(self.maxK,f,'r')
		if False:#isRight:
			plt.plot(self.minPer,f,'b',linewidth=0.5,label='95%')
			plt.plot(self.maxPer,f,'b',linewidth=0.5)
		plt.plot(f*0-para['vPer'],f,'--r',linewidth=0.5,label='$\pm %d$%%'%(para['vPer']*100))
		plt.plot(f*0+para['vPer'],f,'--r',linewidth=0.5)
		plt.legend()
		plt.gca().set_yscale('log')
		plt.xlabel('d$v$/$v_{ref}$')
		plt.ylabel('$f$(Hz)')
		#plt.ylabel('f(Hz)'
		ax=plt.gca()
		ax_divider = make_axes_locatable(plt.gca())
		cax = ax_divider.append_axes('right', size="7%", pad="10%",)
		plt.colorbar(cax=cax,label='count')
		plt.savefig(resDir+'per'+head+'.'+format,dpi=300)
		plt.close()
		
		np.savetxt(resDir+'minDV'+tail,self.minDV)
		np.savetxt(resDir+'maxDV'+tail,self.maxDV)
		np.savetxt(resDir+'maxK'+tail,self.maxK)
		np.savetxt(resDir+'minPer'+tail,self.minPer)
		np.savetxt(resDir+'maxPer'+tail,self.maxPer)
		np.savetxt(resDir+'minSNR'+tail,self.minSNR)
	def calByDKV(self,corrL,k=0,maxCount=-1,fvD0={},**kwags):
		fvD = {}
		f = 1/self.config.para['T']
		#corrL = self.corrDTest.corrL
		if maxCount<0:
			maxCount=len(corrL)
		para = self.config.para
		fvRef=self.fvDAverage[para['refModel']]
		count=0
		count0=0
		for corr in corrL[:maxCount]:
			if len(fvD0)>0:
				if corr.modelFile not in fvD0:
					continue
			FV=d.corr.getFV(corr,f,fvRef,self.minDV*0-para['dV'],self.maxDV*0+para['dV'],self.maxK*para['mulK'],self.minPer*0-para['vPer'],self.maxPer*0+para['vPer'],minSNR=self.minSNR,k=k,**kwags)
			count0+=1
			if len(FV.f)>2:
				fvD[corr.modelFile]=FV
				count+=1
				if count%10:
					print('%d/%d'%(count,count0))
		return fvD


N = 50
N1 =N-1
N2 =N1/2
N3 =N1/3
N4 =N1/4
paraTrainTest={ 'quakeFileL'  : ['CEA_quakesAll'],\
	'stationFileL': ['../stations/CEA.sta_labeled_sort'],#**********'stations/CEA.sta_know_few'\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_220113-025122_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
	'minSNRL'     : [2.5],\
	'oRemoveL'    : [False],\
	'trainMatDir'	  :'/media/jiangyr/1TSSD/',\
	'matDir'	  :'/media/jiangyr/1TSSD/matDirAll/',\
	'trainH5'	  :'/media/jiangyr/1TSSD/trainSet.h5',\
	'matH5'	  :'/media/jiangyr/1TSSD/allClip20220301V1DISP_all.h5',\
	'randA'       : 0.05,\
	'disAmp'      : 0,\
	'midV'        : 4,\
	'mul'		  : 1,\
	'up'          :  1,\
	'move2Int'    :  False,
	'set2One'	  : True,
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/media/jiangyr/MSSD/20220304_15_noG_noloop_noControl_mul1_disAmp0_N5_W0.06_3_5_noInt_One_notAll_smallestest_240_V1/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 400,\
	'eventDir'    : '/media/jiangyr/1TSSD/eventSac/',\
	'z'           :[0,5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,240,300,400,500],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (16**np.arange(0,1.000001,1/N1))*10,\
	'Tav'         : (16**np.arange(0-1/N1,1.000001+1/N1,1/N1))*10,\
	'tSur'        : (16**np.arange(0,1.000001,1/N2))*10,\
	'surPara'     : { 'nxyz':[19,28,15], 'lalo':[55,110],#[40,60,0][55,108]\
					'dlalo':[1,1], 'maxN':60,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.3,\
					'maxIT':10,'nBatch':1,'smoothDV':20,'smoothG':40,'vR':np.array([[53.5,122.1],[48,134.1],[42.1,131.1],[39.1,125],[39.9,115.1],[42,111.9],[45,111.9],[53.5,122.1]]),'perAGs':0.025,'perAGc':0.025,'perN':[4,4,5],'perNG':[6,6,5],'noiselevel':0.00,'perA':0.04,'iso':'F',},\
	'runDir'      : '../DS/20220111_CEA160_TrainTest/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[37.5,55,110,136.5],#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.01,\
	'thresholdTrain'   :0.015,\
	'thresholdTrainDiff'   :0.02,\
	'threshold'   :0.015,\
	'qcThreshold': 0.82285,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'dV'          : 0.3,\
	'mulK'        : 100000,\
	'vPer'        :0.12,\
	'it'          :2,\
	'minSta'      : 4,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  6,\
	'refModel'    : 'models/prem',\
	'p2L':[
		[[42,115],[11,22]],
	],
	'R':[38,55,109,135]}
paraAll={ 'quakeFileL'  : ['CEA_quakesAll'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_220226-130902_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
	'minSNRL'     : [2.5],\
	'oRemoveL'    : [False],\
	'trainMatDir'	  :'/media/jiangyr/1TSSD/matDir/',\
	'matDir'	  :'/media/jiangyr/1TSSD/matDirAll/',\
	'trainH5'	  :'/media/jiangyr/1TSSD/trainSet.h5',\
	'matH5'	  :'/media/jiangyr/1TSSD/allClip20220301V1DISP_all.h5',\
	'randA'       : 0.05,\
	'disAmp'      : 1,
	'midV'        : 4,\
	'mul'		  : 1,\
	'up'          :  1,\
	'move2Int'    :  False,
	'set2One'	  : True,
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/media/jiangyr/MSSD/20220225_15_noG_noloop_noControl_mul1_disAmp1_N5_W0.06_3_5_noInt_One_notAll_smallestest_V15_all/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 10,\
	'eventDir'    : '/media/jiangyr/1TSSD/eventSac/',\
	'z'           :[0,5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,240,300,400,500],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (16**np.arange(0,1.000001,1/N1))*10,\
	'Tav'         : (16**np.arange(0-1/N1,1.000001+1/N1,1/N1))*10,\
	'tSur'        : (16**np.arange(0,1.000001,1/N2))*10,\
	'surPara'     : { 'nxyz':[27,38,16], 'lalo':[55,103],#[40,60,0][55,108]\
					'dlalo':[1,1], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.1,\
					'maxIT':30,'nBatch':30,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[32,122.5],[32,103],[40,103],[43.9,110.9]]),'perAGs':0.0,'perAGc':0.0,'perN':[3,3,4],'perNG':[6,6,4],'noiselevel':0.00,'perA':0.04,'iso':'T'},\
	'runDir'      : '../DS/20220111_CEA160_TrainTest/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[15,55,70,140],#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.01,\
	'thresholdTrain'   :0.015,\
	'thresholdTrainDiff'   :0.02,\
	'threshold'   :0.015,\
	'qcThreshold': 0.82285,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'it'          :2,\
	'minSta'      : 4,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  6,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[15,55,70,140]}


paraNorth={ 'quakeFileL'  : ['CEA_quakesAll'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_220226-130902_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
	'minSNRL'     : [2.5],\
	'oRemoveL'    : [False],\
	'trainMatDir'	  :'/media/jiangyr/1TSSD/matDir/',\
	'matDir'	  :'/media/jiangyr/1TSSD/matDirAll/',\
	'trainH5'	  :'/media/jiangyr/1TSSD/trainSet.h5',\
	'matH5'	  :'/media/jiangyr/1TSSD/allClip20220301V1DISP_all.h5',\
	'randA'       : 0.05,\
	'disAmp'      : 1,
	'midV'        : 4,\
	'mul'		  : 1,\
	'up'          :  1,\
	'move2Int'    :  False,
	'set2One'	  : True,
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/media/jiangyr/MSSD/20220225_15_noG_noloop_noControl_mul1_disAmp1_N5_W0.06_3_5_noInt_One_notAll_smallestest_V15_northV2/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 10,\
	'eventDir'    : '/media/jiangyr/1TSSD/eventSac/',\
	'z'           :[0,5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,240,300,400,500],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (16**np.arange(0,1.000001,1/N1))*10,\
	'Tav'         : (16**np.arange(0-1/N1,1.000001+1/N1,1/N1))*10,\
	'tSur'        : (16**np.arange(0,1.000001,1/N2))*10,\
	'surPara'     : { 'nxyz':[27,38,16], 'lalo':[55,103],#[40,60,0][55,108]\
					'dlalo':[1,1], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.1,\
					'maxIT':30,'nBatch':30,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[32,122.5],[32,103],[40,103],[43.9,110.9]]),'perAGs':0.0,'perAGc':0.0,'perN':[3,3,4],'perNG':[6,6,4],'noiselevel':0.00,'perA':0.04,'iso':'T'},\
	'runDir'      : '../DS/20220111_CEA160_TrainTest/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[32,180,103,135],#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.01,\
	'thresholdTrain'   :0.015,\
	'thresholdTrainDiff'   :0.02,\
	'threshold'   :0.015,\
	'qcThreshold': 0.82285,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'it'          :2,\
	'minSta'      : 4,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  6,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[32,55,103,136]}
'''
41.9,114.9],[45.1,116.9],[50.1,116.9],[50.1,119.9],[53.1,119.9],[53.1,126.1],[50.1,127.1],[50.1,132.1],[41.9,132.1],[41.9,127.1],[38.9,127.9],[38.9,116.9],[40.9,116.9],[40.9,114.9],[41.9,114.9]
[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
'''
paraSC={ 'quakeFileL'  : ['CEA_quakesAll'],\
	'stationFileL': ['../stations/CEA.sta_labeled_sort'],#**********'stations/CEA.sta_know_few'\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_211127-133742_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
	'randA'       : 0.02,\
	'midV'        : 4,\
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/media/jiangyr/MSSD/20211128V2/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 20,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,40,45,50,60,70,80,100,120,150,200,300],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (10**np.arange(0,1.000001,1/49))*10,\
	'Tav'         : (10**np.arange(0-1/49,1.000001+1/49,1/49))*10,\
	'tSur'        : (10**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[11,30,15], 'lalo':[55,110],#[40,60,0][55,108]\
					'dlalo':[1,1], 'maxN':60,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.3,\
					'maxIT':20,'nBatch':1,'smoothDV':20,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[38.5,122.1],[39.5,113],[41.5,110.9],[43.9,110.9]]),'perAGs':0.01,'perAGc':0.01,'perN':[2,2,4],'noiselevel':0.000,'perA':0.05},\
	'runDir'      : '../DS/20211016_CEA160_TrainTest/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[38,55,110,135],#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.015,\
	'thresholdTrain'   :0.015,\
	'threshold'   :0.015,\
	'qcThreshold': 2,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'minSta'      : 3,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  5,\
	'up'          :  up,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[38,55,110,135]}
'''paraNorth={ 'quakeFileL'  : ['CEA_quakesAll'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_220113-025122_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'oRemoveL'    : [False],\
	'maxCount'    : 512*3,\
	'minSNRL'     : [2.5],\
	'trainMatDir' :'/media/jiangyr/1TSSD/matDir/',\
	'matDir'	  :'/media/jiangyr/1TSSD/matDirAll/',\
	'trainH5'	  :'/media/jiangyr/1TSSD/trainSet.h5',\
	'matH5'	  :'/media/jiangyr/1TSSD/allClip20220109V2.h5',\
	'randA'       : 0.0,\
	'midV'        : 4,\
	'mul'		  : 1,\
	'up'          :  1,
	'resDir'      : '/media/jiangyr/MSSD/20220113NorthV3/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 2,\
	'eventDir'    : '/media/jiangyr/1TSSD/eventSac/',\
	'z'           : [0,5,10,15,20,30,40,50,60,80,100,120,160,300],#[0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (12**np.arange(0,1.000001,1/N1))*10,\
	'Tav'         : (12**np.arange(0-1/N1,1.000001+1/N1,1/N1))*10,\
	'tSur'        : (12**np.arange(0,1.000001,1/N4))*10,\
	'surPara'     : { 'nxyz':[27,38,16], 'lalo':[55,103],#[40,60,0][55,108]\
					'dlalo':[1,1], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.1,\
					'maxIT':30,'nBatch':30,'smoothDV':10,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[32,122.5],[32,103],[40,103],[43.9,110.9]]),'perAGs':0.0,'perAGc':0.0,'perN':[3,3,4],'perNG':[6,6,4],'noiselevel':0.00,'perA':0.04,'iso':'T'},\
	'runDir'      : '../DS/20211016_CEA160_TrainTest_North/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[32,180,103,135],#,#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.01,\
	'thresholdTrain'   :0.015,\
	'threshold'   :0.015,\
	'qcThreshold': 0.82285,\
	'it'          :2,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'minSta'      : 4,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  5,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[32,55,103,136]}
'''
paraOrdos={ 'quakeFileL'  : ['CEA_quakesAll'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_220113-025122_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'oRemoveL'    : [False],\
	'maxCount'    : 512*3,\
	'minSNRL'     : [2.5],\
	'trainMatDir' :'/media/jiangyr/1TSSD/matDir/',\
	'matDir'	  :'/media/jiangyr/1TSSD/matDirAll/',\
	'trainH5'	  :'/media/jiangyr/1TSSD/trainSet.h5',\
	'matH5'	  :'/media/jiangyr/1TSSD/allClip20220109.h5',\
	'randA'       : 0.0,\
	'midV'        : 4,\
	'mul'		  : 1,\
	'up'          :1,\
	'resDir'      : '/media/jiangyr/MSSD/20220113OrdosV1/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 2,\
	'eventDir'    : '/media/jiangyr/1TSSD/eventSac/',\
	'z'           : [0,5,10,15,20,30,40,50,60,80,100,120,150,250,350,500],#[0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (12**np.arange(0,1.000001,1/N1))*10,\
	'Tav'         : (12**np.arange(0-1/N1,1.000001+1/N1,1/N1))*10,\
	'tSur'        : (12**np.arange(0,1.000001,1/N1))*10,\
	'surPara'     : { 'nxyz':[13,13,16], 'lalo':[43,104],#[40,60,0][55,108]\
					'dlalo':[1,1], 'maxN':80,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 2,\
					'maxIT':20,'nBatch':1,'smoothDV':10,'smoothG':20,'vR':np.array([[42,105],[42,113],[34,113],[34,105],[42,103]]),'perAGs':0.02,'perAGc':0.02,'perN':[2,2,3],'perNG':[4,4,3],'noiselevel':0.00,'perA':0.04,'iso':'T'},\
	'runDir'      : '../DS/20211016_CEA160_TrainTest_Ordos/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[34,42,105,113],#,#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.01,\
	'thresholdTrain'   :0.015,\
	'threshold'   :0.015,\
	'qcThreshold': 0.82285,\
	'it'          :2,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'minSta'      : 4,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  5,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[33,43,104,114]}
'''
	'surPara'     : { 'nxyz':[40,60,15], 'lalo':[55,110],#[40,60,0][55,108]\
					'dlalo':[0.5,0.5], 'maxN':60,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.3,\
					'maxIT':20,'nBatch':1,'smoothDV':20,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[38.5,122.1],[39.5,113],[41.5,110.9],[43.9,110.9]]),'perAGs':0.01,'perAGc':0.01},\
paraOrdos={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 20,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [5,10,20,30,45,60,80,100,125,150,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'surPara'     : { 'nxyz':[40,47,0], 'lalo':[43,102],#[40,60,0][55,108]\
					'dlalo':[0.3,0.3], 'maxN':100,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01\
					,'maxIT':8,'nBatch':4,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_Ordos_0.03/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[32,42,103,115],#[20,34,96,108][-1,180,-1,180]*******,\
	'nlalo'        :[-1,-1,-1,-1],\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7}
paraYNSC={ 'quakeFileL'  : ['phaseLPickCEA'],\
					'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
					'oRemoveL'    : [False],\
					'avgPairDirL' : ['../models/ayu/Pairs_avgpvt/'],\
					'pairDirL'    : ['../models/ayu/Pairs_pvt/'],\
					'minSNRL'     : [6],\
					'isByQuakeL'  : [True],\
					'remove_respL': [True],\
					'isLoadFvL'   : [False],#False********\
					'byRecordL'   : [False],
					'maxCount'    : 4096*3,\
					'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',
					'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
					'refModel'    : 'models/prem',\
					'perN'        : 20,\
					'eventDir'    : '/HOME/jiangyr/eventSac/',\
					'T'           : (16**np.arange(0,1.000001,1/49))*10,\
					'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
					'z'           : [5,10,20,30,45,60,80,100,125,150,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
					'surPara'     : { 'nxyz':[40,35,0], 'lalo':[36,96],#[40,60,0][55,108]\
									'dlalo':[0.4,0.4], 'maxN':100,#[0.5,0.5]\
									'kmaxRc':0,'rcPerid':[],'threshold':0.01\
									,'maxIT':32,'nBatch':16,'smoothDV':20,'smoothG':40},\
					'runDir'      : 'DS/1013_CEA160_YNSC/',#_man/',\
					'gpuIndex'    : 0,\
					'gpuN'        : 1,\
					'lalo'        :[20,34,96,108],#[20,34,96,108][-1,180,-1,180]*******,\
					'threshold'   :0.05,\
					'minProb'     :0.5,\
					'minP'        :0.5}
NL=np.array([[45,110],[50,115],[55,120],[55,130],[50,136],\
	[43,136],[40,130],[30,125],[30,125],[20,120],[15,110],[18,100],[28,93],[30,80],[40,70],[45,80],[50,90],[45,100],[45,110]])
paraAll={ 'quakeFileL'  : ['../phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : [],
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : '../DS/1015_CEA160_all/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[38,54,110,134],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}
paraAll2={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160,'vR':NL},\
	'runDir'      : '../DS/1015_CEA160_all/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,-1,180],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [35,30, 28,  35, 45],\
	'loL'         : [95,108,118,115,125],\
	'areasLimit'  :  3,\
	'p2L':[\
	[[30,98,0],[34,103,250]],\
	[[45,110,0],[35,105,250]],
	[[45,115,0],[35,110,250]],
	[[41,105,0],[41,125,250]],
	[[33,105,0],[50,130,250]],
	],\
	'R':[]}

paraWest={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160,'vR':NL},\
	'runDir'      : 'DS/1026_CEA160_west/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,-1,100],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [35,30, 28,  35, 45],\
	'loL'         : [95,108,118,115,125],\
	'areasLimit'  :  3}

paraEest={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[112,96,0], 'lalo':[56,102],#[40,60,0][55,108]\
					'dlalo':[0.4,0.4], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : 'DS/1026_CEA160_east/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [28,  35, 45],\
	'loL'         : [110,115,125],\
	'areasLimit'  :  3}

paraNECE={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/NEsta_all.locSensorDas'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : 'models/NEFVSEL/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,240],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[21,52,0], 'lalo':[48,115],#[40,60,0][55,108]\
					'dlalo':[0.4,0.4], 'maxN':129,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.4,\
					'maxIT':8,'nBatch':8,'smoothDV':10,'smoothG':20},\
	'runDir'      : 'DS/1026_CEA160_NECE_SEL/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,-1,-1,-1],\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}

paraNorth={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[65,96,0], 'lalo':[56,102],#[40,60,0][55,108]\
					'dlalo':[0.4,0.4], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':30,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_north/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.05,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3}

paraNorthLager={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[52,71,0], 'lalo':[56,102],#[40,60,0][55,108]\
					'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':30,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_north_lager/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.05,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3}
NL=np.array([[40,103],[45,105],[48,110],[55,120],[53,130],\
	[50,135],[45,135],[40,130],[35,125],[30,125],\
	[30,115],[30,105],[35,103],[40,103]])
paraNorthLagerNew={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[52,71,0], 'lalo':[56,102],#[40,60,0][55,108]\
					'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':30,'smoothDV':20,'smoothG':40,'vR':NL},\
	'runDir'      : '../DS/1026_CEA160_north_lager_new_real/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.04,\
	'minProb'     :0.5,\
	'minP'        :0.6,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3,\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	]}

paraNorthLagerNew2={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,50,55,60,65,70,80,90,100,115,130,160,200,260,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/49))*10,\
	'surPara'     : { 'nxyz':[52,71,0], 'lalo':[56,102],#[40,60,0][55,108]\
					'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1.5,\
					'maxIT':120,'nBatch':60,'smoothDV':20,'smoothG':40,'vR':NL},\
	'runDir'      : '../DS/1026_CEA160_north_lager_new_real2/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.03,\
	'minProb'     :0.7,\
	'minP'        :0.7,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3,\
	'p2L':[\
	[[45,115,0],[35,115,250]],\
	[[45,110,0],[35,105,250]],
	[[45,115,0],[35,110,250]],
	[[41,105,0],[41,125,250]],
	[[33,105,0],[50,130,250]],
	]}
NL=np.array([[30,98],[34,102.6],[31.5,107.5],[26,107.5],[22.5,106],\
	[21,102],[22,98],[24,97.5],[30,98]])
paraYNSC={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['../stations/SCYN_withComp_ac'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,50,55,60,65,70,80,90,100,115,130,160,200,230,260,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (32**np.arange(0,1.000001,1/49))*10,\
	'surPara'     : { 'nxyz':[30,25,0], 'lalo':[34,97],#[40,60,0][55,108]\
					'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1.5,\
					'maxIT':30,'nBatch':1,'smoothDV':20,'smoothG':40,'vR':NL},\
	'runDir'      : '../DS/SCYNLonger/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[0,180,0,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,-1,-1,-1],\
	'threshold'   :0.03,\
	'minProb'     :0.7,\
	'minP'        :0.7,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'R'           :[20, 35, 95, 110],
	'areasLimit'  :  3,\
	'p2L':[\
	[[30,98,0],[34,103,250]],\
	[[45,110,0],[35,105,250]],
	[[45,115,0],[35,110,250]],
	[[41,105,0],[41,125,250]],
	[[33,105,0],[50,130,250]],
	]}

paraYNSCV2={ 'quakeFileL'  : ['phaseLPickCEA'],\
	'stationFileL': ['../stations/SCYN_withComp_ac'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [False],#False********\
	'byRecordL'   : [False],\
	'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,50,55,60,65,70,80,90,100,115,130,160,200,260,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'tSur'        : (16**np.arange(0,1.000001,1/49))*10,\
	'surPara'     : { 'nxyz':[30,25,0], 'lalo':[34,97],#[40,60,0][55,108]\
					'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1.5,\
					'maxIT':30,'nBatch':1,'smoothDV':20,'smoothG':40,'vR':NL},\
	'runDir'      : '../DS/SCYNV2/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[0,180,0,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,-1,-1,-1],\
	'threshold'   :0.03,\
	'minProb'     :0.7,\
	'minP'        :0.7,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'R'           :[20, 35, 95, 110],
	'areasLimit'  :  3,\
	'p2L':[\
	[[30,98,0],[34,103,250]],\
	[[45,110,0],[35,105,250]],
	[[45,115,0],[35,110,250]],
	[[41,105,0],[41,125,250]],
	[[33,105,0],[50,130,250]],
	]}

paraAllSq={ 'quakeFileL'  : ['../phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [False],\
	'maxCount'    : 1024,\
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 20,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : [],
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : '../DS/1015_CEA160_all/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[38,54,110,134],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}

paraAllDt={ 'quakeFileL'  : ['../phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [False],\
	'maxCount'    : 512*3,\
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 20,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : [],
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : '../DS/1015_CEA160_all/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[38,54,110,134],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}

paraAllO={ 'quakeFileL'  : ['../phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [False],\
	'maxCount'    : 512*3,\
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 20,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (16**np.arange(0,1.000001,1/49))*10,
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : '../DS/1015_CEA160_all/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[38,54,110,134],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}

paraAllONew={ 'quakeFileL'  : ['../phaseLPickCEA'],\
	'stationFileL': ['../stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_211017-150906_model.h5',
	'isLoadFvL'   : [True],#False********\
	'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
	'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
	'resDir'      : '/fastDir/results/20210408/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
	'perN'        : 1,\
	'eventDir'    : '/HOME/jiangyr/eventSac/',\
	'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
	'T'           : (16**np.arange(0,1.000001,1/49))*10,
	'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
	'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
					'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : '../DS/20210408_CEA160_all/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[38,54,110,134],#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3,\
	'up'          :  up}
'''
