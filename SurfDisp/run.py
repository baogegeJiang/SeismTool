from glob import glob
import os
from imp import reload
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ..mathTool import mathFunc
from ..deepLearning import fcn
from ..io import seism
from . import DSur
from . import dispersion as d
import gc
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
        minDist=110*10,maxDist=110*170,minDDist=200,\
        maxDDist=1801,para=para,isFromO=True,removeP=True)
def saveListStr(file,strL):
	with open(file,'w+') as f:
		for STR in strL:
			f.write('%s\n'%STR)
def loadListStr(file):
	with open(file,'r') as f:
		l = []
		for line in f.readlines():
			l.append(line[:-1])
	return l
class runConfig:
	def __init__(self,para={}):
		sacPara = {'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
                   'output':'VEL','freq':[1/250,1/8*0+1/6],\
                   'filterName':'bandpass',\
                   'corners':4,'toDisp':False,\
                   'zerophase':True,'maxA':1e15}
		self.para={ 'quakeFileL'  : ['phaseLPickCEA'],\
		            'stationFileL': ['stations/CEA.sta_sel'],#**********'stations/CEA.sta_know_few'\
		            'oRemoveL'    : [False],\
		            'avgPairDirL' : ['../models/ayu/Pairs_avgpvt/'],\
		            'pairDirL'    : ['../models/ayu/Pairs_pvtsel/'],\
		            'minSNRL'     : [5],\
		            'isByQuakeL'  : [True],\
		            'remove_respL': [True],\
		            'isLoadFvL'   : [False],#False********\
		            'byRecordL'   : [False],
		            'maxCount'    : 4096*3,\
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
		
class run:
	def __init__(self,config=runConfig(),self1 = None):
		self.config = config
		self.model  = None
		if self1 != None:
			self.corrL  =  self1.corrL
			self.corrL1 =  self1.corrL1
			self.fvD    =  self1.fvD
			self.fvDAvarage = self1.fvDAvarage
			self.quakes = self1.quakes
			self.stations = self1.stations
	def loadCorr(self,isLoad=True,isLoadFromMat=False,trainSetDir='/fastDir/trainSet/'):
		config     = self.config
		corrL      = []
		stations   = seism.StationList([])
		quakes     = seism.QuakeL()
		fvDAvarage = {}
		fvD        = {}
		fvD0       = {}
		para       = config.para
		N          = len(para['stationFileL'])
		fvDAvarage['models/prem']=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		for i in range(N):
			sta     = seism.StationList(para['stationFileL'][i])
			sta.inR(para['lalo'])
			sta.set('oRemove', para['oRemoveL'][i])
			sta.getInventory()
			stations += sta
			q       = seism.QuakeL(para['quakeFileL'][i])
			quakes  += q
			#fvDA    = para['dConfig'].loadNEFV(sta,fvDir=para['avgPairDirL'][i])
			#d.qcFvD(fvDA,threshold=para['thresholdTrain'])
			#
			fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][i])
			fvM     = d.fvD2fvM(fvd,isDouble=True)
			fvDA    = d.fvM2Av(fvM,threshold=para['qcThreshold'],fL=1/self.config.para['Tav'],minThreshold=para['minThreshold'],minSta=para['minSta'])
			d.qcFvD(fvDA,threshold=para['thresholdTrain'])
			fvDAvarage.update(fvDA)

			d.replaceByAv(fvd,fvDA,threshold=para['thresholdTrain']*3)
			fvD.update(fvd)
			fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][i])
			#fvd = { for key in fvd0}
			d.replaceByAv(fvd,fvDA,isReplace=False,threshold=para['thresholdTrain']*2)
			fvD0.update(fvd)
			if isLoad:
				if not isLoadFromMat:
					corrL0  = para['dConfig'].quakeCorr(q,sta,\
							byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
							minSNR=para['minSNRL'][i],isLoadFv=para['isLoadFvL'][i],\
							fvD=fvD,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],resDir=para['eventDir'],maxCount=para['maxCount'],up=para['up'])
					corrL   += corrL0
				else:
					corrL = d.corrL()
					corrL.load(trainSetDir)
					self.fvL = loadListStr(trainSetDir+'fvL')
					self.fvTrain = loadListStr(trainSetDir+'fvTrain')
					self.fvTest = loadListStr(trainSetDir+'fvTest')
					self.fvValid = loadListStr(trainSetDir+'fvValid')
		if isLoad:
			self.corrL  = d.corrL(corrL,maxCount=para['maxCount'])
			self.corrL1 = d.corrL(self.corrL,maxCount=para['maxCount'],fvD=fvD)
		fvDNew ={}
		fvD0New = {}
		fvDAvarageNew ={'models/prem':fvDAvarage['models/prem']}
		for corr in self.corrL1:
			key = corr.modelFile
			fvDNew[key]=fvD[key]
			fvD0New[key]=fvD0[key]
			if len(key.split('_'))>=2:
				name0 = key.split('_')[-2]
				name1 = key.split('_')[-1]
				modelName ='%s_%s'%(name0,name1)
				#print(modelName0)
				if modelName in fvDAvarage:
					if modelName not in fvDAvarageNew:
						fvDAvarageNew[modelName] = fvDAvarage[modelName]
		self.fvD    = fvDNew
		self.fvD0   = fvD0New
		self.fvDAvarage = fvDAvarageNew
		self.quakes = quakes
		self.stations = stations
	def train(self,up=1,isRand=True,isShuffle=False,isAverage=False):
		para    = self.config.para
		tTrain = para['T']
		if isRand:
			if isShuffle:
				fvL = [key for key in self.fvDAvarage]
				random.shuffle(fvL)
				fvN = len(fvL)
				fvn = int(fvN/10)
				fvTrain = fvL[fvn*2:]
				fvTest  = fvL[fvn:fvn*2]
				fvValid = fvL[:fvn]
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
			if up>1:
				for corrL  in [self.corrLTrain,self.corrLValid,self.corrLTest]:
					corrL.reSetUp(up)
			#corrLQuakePTest  = d.corrL(corrLQuakePNE)
			#random.shuffle(corrLQuakePTrain)
			#random.shuffle(corrLQuakePValid)
			#random.shuffle(corrLQuakePTest)
		if isAverage:
			fvD =self.fvD
		else:
			fvD =self.fvD0
		self.corrLTrain.setTimeDis(fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLTest.setTimeDis(fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLValid.setTimeDis(fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)	
		self.loadModel()
		fcn.trainAndTest(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   		outputDir=para['trainDir'],sigmaL=[1],tTrain=tTrain,perN=100,count0=30,w0=8*1/1.5)#w0=3#4
		#fcn.trainAndTest(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   	#	outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=50,count0=200,w0=1.5)#w0=3
	def saveTrainSet(self,saveDir='/fastDir/trainSet/',isMat=False):
		if isMat:
			self.corrL.save(saveDir)
		saveListStr(saveDir+'fvL',self.fvL)
		saveListStr(saveDir+'fvTrain',self.fvTrain)
		saveListStr(saveDir+'fvTest',self.fvTest)
		saveListStr(saveDir+'fvValid',self.fvValid)
	def trainSq(self):
		fvL = [key for key in self.fvDAvarage]
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
		set2One=True,move2Int=False,randMove=True)
		self.corrLTest.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLValid.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLTrain.newCall(np.arange(10))
		print(self.corrLTrain.t0L)
		self.loadModelSq()
		fcn.trainAndTestSq(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   		outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=20,count0=20,w0=10)
	def trainDt(self):
		fvL = [key for key in self.fvDAvarage]
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
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLTest.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLValid.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
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
			self.model = fcn.modelUp(channelList=[0,1])
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
	
	def calResOneByOne(self,isLoadModel=True):
		config     = self.config
		para       = config.para
		N          = len(para['stationFileL'])
		fvDAvarage = {}
		fvDAvarage[para['refModel']]=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		if 'modelFile' in para and isLoadModel:
			print('loadFile')
			self.loadModelUp(para['modelFile'])
		for i in range(N):
			sta     = seism.StationList(para['stationFileL'][i])
			sta.inR(para['lalo'])
			print('sta num:',len(sta))
			sta.set('oRemove', para['oRemoveL'][i])
			sta.getInventory()
			q       = seism.QuakeL(para['quakeFileL'][i])
			print(para['quakeFileL'][i],len(q))
			self.stations = sta
			q.set('sort','sta')
			q.sort()
			perN= self.config.para['perN']
			for j in range(self.config.para['gpuIndex'],int(len(q)/perN),self.config.para['gpuN']):#self.config.para['gpuIndex']
				print('doing for %d %d in %d'%(j*perN,min(len(q)-1,j*perN+perN),len(q)))
				corrL0  = para['dConfig'].quakeCorr(q[j*perN:min(len(q)-1,j*perN+perN)],sta,\
	    				byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
	    				minSNR=para['minSNRL'][i],isLoadFv=False,\
    					fvD=fvDAvarage,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],\
    					resDir=para['eventDir'])
				self.corrL  = d.corrL(corrL0,maxCount=para['maxCount'])
				if len(self.corrL)==0:
					continue
				self.corrL.reSetUp(up=5)
				#self.calRes()
				para = self.config.para
				self.corrL.setTimeDis(fvDAvarage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
				byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
				set2One=True,move2Int=False,modelNameO=para['refModel'],noY=True)
				self.corrL.getAndSaveOld(self.model,'%s/CEA_P_'%para['resDir'],self.stations\
				,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'])
				corrL0 = 0
				self.corrL = 0
				gc.collect()
	def calRes(self):
		para = self.config.para
		self.corrL1.setTimeDis(self.fvDAvarage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
		set2One=True,move2Int=False,modelNameO=para['refModel'],noY=True)
		self.corrL1.getAndSaveOld(self.model,'%s/CEA_P_'%para['resDir'],self.stations\
		,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'])
		#print(self.corrL.t0L)
	def loadRes(self):
		stations = []
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		self.stations = seism.StationList(stations)
		self.stations.inR(self.config.para['lalo'])
		print(len(self.stations))
		para    = self.config.para
		fvDGet,quakesGet = para['dConfig'].loadQuakeNEFV(self.stations,quakeFvDir=para['resDir'])
		self.fvDGet  = fvDGet
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
	def getAv(self):
		for fv in self.fvDGet:
			self.fvDGet[fv].qc(threshold=-self.config.para['minP'])
		para = self.config.para
		self.fvMGet  =d.fvD2fvM(self.fvDGet,isDouble=True)
		#print(self.fvMGet)qcThreshold
		self.fvAvGet = d.fvM2Av(self.fvMGet,threshold=para['qcThreshold'],minThreshold=para['minThreshold'],minSta=para['minSta'])
		#print(self.fvAvGet)
		for fv in self.fvAvGet:
			self.fvAvGet[fv].qc(threshold=self.config.para['threshold'])
		d.qcFvD(self.fvAvGet)
	def getAV(self):
		self.fvAvGetL = [self.fvAvGet[key] for key in self.fvAvGet]
		self.FVAV     = d.averageFVL(self.fvAvGetL)
	def limit(self,threshold=3):
		for key in self.fvAvGet:
			self.FVAV.limit(self.fvAvGet[key],threshold=threshold)
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
	def preDS(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z'];surPara= para['surPara'];DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir'])
		self.DS = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations,1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.test(vL,indexL,self.stations)
	def preDSRef(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z'];surPara= para['surPara'];DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'ref/')
		self.DSRef = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations[::3],1/tSur,isRef=True,fvRef=self.FVAV)
			self.indexL = indexL
			self.vL   = vL
			DS.test(vL,indexL,self.stations[::3])
	def preDSSyn(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'syn/',mode='syn')
		self.DS = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations[-1::-1],1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.testSyn(vL,indexL,self.stations[-1::-1])
	def preDSSynOld(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'syn/',mode='syn')
		self.DS = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations,1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.testSyn(vL,indexL,self.stations)
	def preDSOld(self):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir'])
		self.DS = DS
		indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations,1/tSur)
		self.indexL = indexL
		self.vL   = vL
		DS.test(vL,indexL,self.stations)
	def loadAndPlot(self,isPlot=True):
		self.DS.loadRes()
		if isPlot:
			self.DS.plotByZ(p2L=self.config.para['p2L'],R=self.config.para['R'])
	def test(self):
		self.loadCorr()
		self.train()
	def plotTrainDis(self):
		R = self
		disL,vL,fL,fvAverage = d.outputFvDist(self.fvD,R.stations,t=R.config.para['T'],keys=R.fvTrain)
		d.plotFvDist(disL,vL,fL,'predict/fvDistTrain.eps')

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvD,R.stations,t=R.config.para['T'],keys=R.fvValid)
		d.plotFvDist(disL,vL,fL,'predict/fvDistValid.eps')

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvD,R.stations,t=R.config.para['T'],keys=R.fvTest)
		d.plotFvDist(disL,vL,fL,'predict/fvDistTest.eps')

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvDAvarage,R.stations,t=R.config.para['T'],keys=R.fvTrain)
		d.plotFV(vL,fL,'predict/FVTrain.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvDAvarage,R.stations,t=R.config.para['T'],keys=R.fvValid)
		d.plotFV(vL,fL,'predict/FVValid.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvDAvarage,R.stations,t=R.config.para['T'],keys=R.fvTest)
		d.plotFV(vL,fL,'predict/FVTest.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvD0,R.stations,t=R.config.para['T'],keys=R.fvTrain)
		d.plotFV(vL,fL,'predict/FVTrainSingle.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = run.d.outputFvDist(self.fvD0,R.stations,t=R.config.para['T'],keys=R.fvValid)
		d.plotFV(vL,fL,'predict/FVValidSingle.eps',isAverage=True,fvAverage=fvAverage)
	def plotGetDis(self):
		R = self
		disL,vL,fL,fvAverage = d.outputFvDist(self.fvAvGet,R.stations,t=R.config.para['T'],keys=R.fvTrain)
		d.plotFV(vL,fL,'predict/FVTrainGet.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvAvGet,R.stations,t=R.config.para['T'],keys=R.fvValid)
		d.plotFV(vL,fL,'predict/FVValidGet.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvAvGet,R.stations,t=R.config.para['T'],keys=R.fvTest)
		d.plotFV(vL,fL,'predict/FVTestGet.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvD0,R.stations,t=R.config.para['T'],keys=R.fvTest)
		d.plotFV(vL,fL,'predict/FVTestSingle.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvDGet,R.stations,t=R.config.para['T'],keys=R.fvTrain)
		d.plotFV(vL,fL,'predict/FVTrainSingleGet.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvDGet,R.stations,t=R.config.para['T'],keys=R.fvValid)
		d.plotFV(vL,fL,'predict/FVValidSingleGet.eps',isAverage=True,fvAverage=fvAverage)

		disL,vL,fL,fvAverage = d.outputFvDist(self.fvDGet,R.stations,t=R.config.para['T'],keys=R.fvTest)
		d.plotFV(vL,fL,'predict/FVTestSingleGet.eps',isAverage=True,fvAverage=fvAverage)

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
	'up'          :  5}

paraTrainTest={ 'quakeFileL'  : ['CEA.quakes'],\
    'stationFileL': ['../stations/CEA.sta_labeled_sort'],#**********'stations/CEA.sta_know_few'\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_211017-150906_model.h5',
    'isLoadFvL'   : [True],#False********\
    'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
    'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/20211018/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 20,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'T'           : (16**np.arange(0,1.000001,1/49))*10,\
	'Tav'         : (16**np.arange(0-1/49,1.000001+1/49,1/49))*10,\
	'tSur'        : (16**np.arange(0,1.000001,1/49))*10,\
    'surPara'     : { 'nxyz':[40,60,15], 'lalo':[55,110],#[40,60,0][55,108]\
                    'dlalo':[0.5,0.5], 'maxN':60,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':8,'smoothDV':20,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[38.5,122.1],[39.5,113],[41.5,110.9],[43.9,110.9]])},\
	'runDir'      : '../DS/20211016_CEA160_TrainTest/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[38,55,110,135],#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.015,\
	'thresholdTrain'   :0.015,\
	'threshold'   :0.015,\
	'qcThreshold': 3,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'minSta'      : 5,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3,\
	'up'          :  5,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[38,55,110,135]}

paraNorth={ 'quakeFileL'  : ['CEA.quakes'],\
    'stationFileL': ['../stations/CEA.sta_know_few'],\
	'modelFile'   : '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_211021-194504_model.h5',
    'isLoadFvL'   : [True],#False********\
    'byRecordL'   : [True],\
	'maxCount'    : 512*3,\
    'trainDir'    : 'predict/0130_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/media/jiangyr/MSSD/20211025NorthV1/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 3,\
    'eventDir'    : '/media/jiangyr/1TSSD/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'T'           : (16**np.arange(0,1.000001,1/49))*10,\
	'Tav'         : (16**np.arange(0-1/49,1.000001+1/49,1/49))*10,\
	'tSur'        : (16**np.arange(0,1.000001,1/49))*10,\
    'surPara'     : { 'nxyz':[52,74,15], 'lalo':[55,103],#[40,60,0][55,108]\
                    'dlalo':[0.5,0.5], 'maxN':60,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':8,'smoothDV':20,'smoothG':20,'vR':np.array([[43.9,110.9],[54.5,122],[48.5,134],[41.5,131.1],[40,125.1],[32,122.5],[32,103],[37.5,103],[43.9,110.9]])},\
	'runDir'      : '../DS/20211016_CEA160_TrainTest_North/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[32,180,103,135],#,#[-90,90,0,180],#[38,54,110,134],#[20,34,96,108][]*******,\
	'minThreshold':0.015,\
	'thresholdTrain'   :0.015,\
	'threshold'   :0.010,\
	'qcThreshold': 3,\
	'minProb'     :0.5,\
	'minP'        :0.6,\
	'minSta'      : 5,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3,\
	'up'          :  5,\
	'refModel'    : 'models/prem',\
	'p2L':[\
	[[45,115],[35,115]],\
	[[45,110],[35,105]],
	[[45,115],[35,110]],
	[[41,105],[41,125]],
	[[33,105],[50,130]],
	],\
	'R':[32,55,103,135]}
