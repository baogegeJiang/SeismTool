import obspy 
import numpy as np
from obspy import UTCDateTime,read,Trace,Stream
from obspy.io import sac
import os 
import random
from matplotlib import pyplot as plt
import time
from time import ctime
from glob import glob
from numba import jit
from ..mathTool.distaz import DistAz
from .dataLib import filePath
from ..mathTool.mathFunc import rotate,getDetec
from ..mathTool.mathFunc_bak import R as mathR
from . import parRead
from SeismTool.plotTool import figureSet as fs
dm = 0
printDetail=True
comp3='RTZ'
comp33=[]
cI33=[]
for i in range(3):
	for j in range(3):
		cI33.append([i,j])
		comp33.append(comp3[i]+comp3[j])
plt.switch_backend('agg')
fileP = filePath()
def tolist(s,d='/'):
	return s.split(d)
nickStrL='1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
strType={'S':str,'f':float,'F':float,'i':int, 'l':tolist,'b':bool,'u':UTCDateTime}
NoneType = type(None)


class Dist:
	'''
	class which can be easily extended for different object with 
	accesses to inputing and outputing
	it acts similar with dictionary class but have list structure inside to reduce usage of memory
	'''
	def __init__(self,*argv,**kwargs):
		self.defaultSet()
		self.splitKey = ' '
		self.l = [None for i in range(len(self.keys))]
		for i in range(len(self.keys0)):
			self.l[i] = self.keys0[i]
		if 'keysIn' in kwargs :
			if isinstance(kwargs['keysIn'],list):
				self.keysIn     = kwargs['keysIn']
			else:
				self.keysIn     = kwargs['keysIn'].split()
		for i in range(len(argv)):
			self[self.keysIn[i]] = argv[i]
		if 'splitKey' in kwargs:
			self.splitKey = kwargs['splitKey']

		if 'line' in kwargs:
			self.setByLine(kwargs['line'])
		
		#print(kwargs)
		for key in self.keys:
			if key in kwargs:
				self[key]=kwargs[key]
	def defaultSet(self):
		self.keys = ['']
		self.keysType =['s']
		self.keysIn = ['']
		self.keysName = ['']
		self.keys0 = [None]
	def index(self,key):
		if key in self.keys:
			return self.keys.index(key)
		return -1
	def __getitem__(self,key):
		if not key in self.keys:
			print('no ',key)
		return self.l[self.index(key)]
	def __setitem__(self,key,value):
		if not key in self.keys:
			print('no ',key)
		self.l[self.index(key)] = value

	def setByLine(self, line):
		if self.splitKey != ' ':
			tmp = line.split(self.splitKey)
		else:
			tmp = line.split()
		#print(tmp,self.splitKey)
		for i in range(min(len(tmp),len(self.keysIn))):
			tmp[i] = tmp[i].strip()
			index = self.index(self.keysIn[i])
			if tmp[i]!='-99999':
				#print(tmp[i],strType[self.keysType[index][0]])
				#print(self.keysIn[i],index,self.keysType[index][0])
				self[self.keysIn[i]] = strType[self.keysType[index][0]](tmp[i])
			else:
				self[self.keysIn[i]] = None
	def __str__(self,*argv):
		line = ''
		if len(argv)>0:
			keysOut = argv[0]
		else:
			keysOut =self.keysIn
		s= self.splitKey
		if len(argv)>1:
			s = argv[1]
		for key in keysOut:
			if not isinstance(self[key],type(None)):
				line += str(self[key])+s
			else:
				line += '-99999 '
		return line[:-1]
	def __repr__(self):
		return self.__str__()
	def __iter__(self):
		return self.keys.__iter__()
	def copy(self):
		type(self)()
		inD = {'keysIn':self.keys}
		for key in self:
			inD[key]  = self[key]
		selfNew=type(self)(**inD)
		selfNew.keysIn = self.keysIn.copy()
		selfNew.splitKey = self.splitKey
		return selfNew
	def update(self,selfNew):
		for key in selfNew:
			if not isinstance(selfNew[key],NoneType):
				self[key] = selfNew[key]
	def keyIn(self):
		keyIn = ''
		for tmp in self.keysIn:
			keyIn += tmp  + ' ' 
		return keyIn[:-1]
	def name(self,s =' '):
		return self.__str__(self.keysName,s)
	def __eq__(self,name1):
		name0 = ''
		if  isinstance(name1,type(self)):
			self1 = name1
			name1 = ''
			for key in self1.keysName:
				name1 = self1.name()
		for key in self.keysName:
			name0 = self.name()
		return name0 == name1
	def loc(self):
		return [self['la'],self['lo'],self['dep']]
	def distaz(self,loc):
		if isinstance(loc,list) or isinstance(loc,np.ndarray):
			dis = DistAz(self['la'],self['lo'],loc[0],loc[1])
		else:
			dis = DistAz(self['la'],self['lo'],loc['la'],loc['lo'])
		return dis
	def dist(self,loc):
		dis = self.distaz(loc)
		return dis.degreesToKilometers(dis.getDelta())
	def az(self,loc):
		dis = self.distaz(loc)
		return dis.getAz()
	def baz(self,loc):
		dis = self.distaz(loc)
		return dis.getBaz()
	def __gt__(self,self1):
		if 'time' in self.keys:
			return self['time']>self1['time']
		else:
			return self['pTime']>self1['pTime']
	def __ge__(self,self1):
		if 'time' in self.keys:
			return self['time']>=self1['time']
		else:
			return self['pTime']>=self1['pTime']

	#def __eq__(self,self1):
	#    return self['time']==self1['time']

	def __lt__(self,self1):
		if 'time' in self.keys:
			return self['time']<self1['time']
		else:
			return self['pTime']<self1['pTime']

	def __le__(self,self1):
		if 'time' in self.keys:
			return self['time']<=self1['time']
		else:
			return self['pTime']<=self1['pTime']
	def inR(self,lalo):
		if isinstance(lalo,mathR):
			return lalo.isIn(self.loc()[:2])
		if self['la']<=lalo[0] or self['la']>lalo[1] or self['lo']<=lalo[2] or self['lo']>lalo[3]:
			return False
		return True

defaultStats = {'network':'00','station':'00000','channel':'000'}
class Station(Dist):
	def __init__(self,*argv,**kwargs):
		super().__init__(*argv,**kwargs)
		#if not isinstance(self['compBase'],NoneType): 
		#    self['comp'] = [self['compBase']+s for s in 'ENZ' ]
		#print(self['index'])
		if isinstance( self['index'],NoneType)==False and isinstance( self['nickName'],NoneType):
			self['nickName'] = self.getNickName(self['index'])
		self.defaultStats = defaultStats
	def defaultSet(self):
		super().defaultSet()
		self.keysIn   = 'net sta compBase lo la erroLo erroLa dep erroDep '.split()
		self.keys     = 'net sta compBase lo la erroLo erroLa dep erroDep nickName \
		comp index nameFunc sensorName dasName sensorNum nameMode netSta doFilt oRemove baseSacName starttime endtime sensitivity sort'.split()
		self.keysType ='S S S f f f f f f S l f F S S S S S b b S u u f S'.split()
		self.keys0 =    [None,None,'BH',None,None,0    ,   0, 0   ,0,  None,     \
		None,None, fileP,'','','','','',True,False,'net.sta.info.compBase',UTCDateTime(1970,1,1),UTCDateTime(2099,1,1),1.239000e+09,'sta']
		self.keysName = ['net','sta']
	def getNickName(self, index):
		nickName = ''
		N      = len(nickStrL)
		for i in range(4):
			tmp    = index%N
			nickName += nickStrL[tmp]
			index = int(index/N)
		return nickName
	def getFileNames(self, time0,time1=None):
		if isinstance(time1, NoneType):
			time1 = time0+86399
		if time0>self['endtime'] or time1<self['starttime']:
			return [[],[],[]]
		return [self['nameFunc'](self['net'],self['sta'], \
			comp, time0,time1,self['nameMode']) for comp in self['comp']]
	def __setitem__(self,key,value):
		super().__setitem__(key,value)
		if not key in self.keys:
			print('no ',key)
		self.l[self.index(key)] = value
		if key =='compBase':
			self['comp'] = [self['compBase']+s for s in 'ENZ']
		if key =='net' and self['nameMode']=='':
			self['nameMode'] = self['net']
		if key =='netSta' and self[key]!='' and isinstance(self['net'],NoneType)\
		 and isinstance(self['sta'],NoneType):
			self['net'], self['sta'] = self[key].split('.')[:2]
	def baseSacName(self,resDir='',strL='ENZ',infoStr=''):
		nameKeyL = self['baseSacName'].split('.')
		fileL =[]
		for comp in strL:
			fileStr = ''
			for nameKey in nameKeyL:
				if nameKey in self.keys:
					tmpStr = self[nameKey]
				else:
					tmpStr = nameKey
				if nameKey == 'BH' or nameKey == 'compBase':
					tmpStr += comp
				if nameKey == 'info' and infoStr == '':
					continue
				if nameKey == 'info' and infoStr != '':
					tmpStr = infoStr
				fileStr += tmpStr +'.'
			if 'info' not in nameKeyL and infoStr != '':
				fileStr += infoStr+'.'
			fileL.append(resDir+'/'+fileStr[:-1])
		return fileL
	def set(self,key,value):
		for tmp in self:
			tmp[key] = value
	def getInventory(self):
		self.sensor=[]
		self.das=[]
		print(self)
		for i in range(3):
			sensor, das=  self['nameFunc'].getInventory(self['net'],self['sta'],\
				self['sensorName'],self['dasName'],comp=self['comp'][i],nameMode=self['nameMode'])
			self.sensor.append(sensor)
			self.das.append(das)
		return self.sensor,self.das
	def getSensorDas(self):
		if self['sensorName'] == '' or self['dasName']=='':
			sensorName,dasName,sensorNum=self['nameFunc'].getSensorDas(self['net'],self['sta'],nameMode=self['nameMode'])
			self['sensorName'] = sensorName
			self['dasName']    = dasName
			self['sensorNum'] = sensorNum
			if self['net'] == 'YP':
				self['compBase']=self['sensorName'][-3:-1]
				self['comp'] = [self['compBase']+s for s in 'ENZ' ]
		return self['sensorName'],self['dasName'],self['sensorNum']
	def loc(self):
		return [self['la'],self['lo'],self['dep']]

class StationList(list):
	def __init__(self,*argv,**kwargs):
		super().__init__()
		self.inD = {}
		if len(argv)>0:
			if isinstance(argv[0],list):
				for sta in argv[0]:
					self.append(sta)
			if isinstance(argv[0],str):
				self.read(argv[0],**kwargs)
	def inR(self,lalo):
		indexL = []
		for i in range(len(self)-1,-1,-1):
			sta = self[i]
			if sta['la']<lalo[0] or sta['la']>lalo[1] or sta['lo']<lalo[2] or sta['lo']>lalo[3]:
				indexL.append(i)
		for i in  indexL:
			self.pop(i)
	def notInR(self,lalo):
		indexL = []
		for i in range(len(self)-1,-1,-1):
			sta = self[i]
			if sta['la']<lalo[0] or sta['la']>lalo[1] or sta['lo']<lalo[2] or sta['lo']>lalo[3]:
				pass
			else:
				indexL.append(i)
		for i in  indexL:
			self.pop(i)
	def __add__(self,self1):
		selfNew = StationList([])
		for station in self:
			selfNew.append(station)
		for station in self1:
			selfNew.append(station)
		return selfNew
	def read(self,fileName,isUnique=False,uniqueKeys=['sta']):
		self.header = []
		lines = []
		nameL = []
		for sta in self:
			staName = ''
			for uniqueKey in uniqueKeys:
				staName += sta[uniqueKey]
			nameL.append(staName)
		for fileName in glob(fileName):
			with open(fileName,'r') as staFile:
				lines += staFile.readlines()
		inD = {}
		index=0
		for line in lines:
			if line[0]=='#':
				keys = line[1:]
				inD['keysIn'] = keys
				continue
			inD['line'] = line
			inD['index']= index
			sta = Station(**inD)
			if isUnique:
				staName = ''
				for uniqueKey in uniqueKeys:
					staName += sta[uniqueKey]
				if staName in nameL:
					continue
				nameL.append(staName)
			self.append(sta)
			index+=1
		self.inD = inD
	def write(self,fileName,*argv,**kwargs):
		with open(fileName,'w+') as f:
			keysOut = ''
			if 'keysIn' in self.inD:
				keysOut = self.inD['keysIn']
			if len(argv)>0:
				keysOut = argv[0]
			if len(keysOut) >0:
				keysOut = keysOut.split()
				for sta in self:
					sta.keysIn = keysOut
			keysOut = '#' + self[0].keyIn()+'\n'
			f.write(keysOut)
			if 'indexL' in kwargs:
				f.write(self.__str__(kwargs['indexL']))
			else:
				f.write(self.__str__())
	def __str__(self,indexL=[]):
		line =''
		if len(indexL)==0:
			for sta in self:
				line += '%s\n'%sta
		else:
			for i in indexL:
				sta = self[i]
				line += '%s\n'%sta
		return line 
	def loc0(self):
		loc = np.zeros(3)
		count = 0
		strL = ['la','lo','dep']
		for station in self:
			for i in range(3):
				tmpStr = strL[i]
				#print(station[tmpStr],i)
				if station[tmpStr] !=None:
					loc[i] = loc[i] + station[tmpStr]
		return loc/len(self)
	def getInventory(self):
		for station in self:
			sensorName, dasName, sensorNum =  station.getSensorDas()
			if sensorName != 'UNKNOWN' and dasName != 'UNKNOWN':
				sensor,das=station.getInventory()
	def getSensorDas(self):
		for station in self:
			sensorName, dasName,sensorNum =  station.getSensorDas()
	def find(self,sta,net=''):
		for station in self:
			if station['sta'] != sta:
				continue
			if net !='' and station['net'] != net:
				continue
			return station
		return None
	def Find(self,netSta, spl='.'):
		netSta = netSta.split('/')[-1]
		net, sta = netSta.split(spl)[:2]
		return self.find(sta,net)
	def index(self,net,sta):
		for i in range(len(self)):
			station = self[i]
			if station['sta'] != sta:
				continue
			if net !='' and station['net'] != net:
				continue
			return i
		return None

	def set(self,key,value):
		for tmp in self:
			tmp[key] = value
	def plot(self,filePath='station.jpg',isSave=True,animated=False):
		if isSave:
			plt.close()
		p =[]
		for sta in self:
			p.append(plt.plot(sta['lo']%360,sta['la'],'^k'))
		if isSave:
			plt.savefig(filePath,dpi=300)
			plt.close()
		return p
	def loc(self):
		laL=[]
		loL=[]
		for sta in self:
			laL.append(sta['la'])
			loL.append(sta['lo'])
		return np.array(laL),np.array(loL)

		
class Record(Dist):
	def __init__(self,*argv,**kwargs):
		super().__init__(*argv,**kwargs)
	def defaultSet(self):
		super().defaultSet()
		self.keysIn   = 'staIndex pTime sTime pProb sProb'.split()
		self.keys     = 'staIndex pTime sTime pProb sProb pCC  sCC  pM   pS   sM   sS staName no'.split()
		self.keysType = 'i        f     f     f      f    f    f    f    f    f    f  S'.split()
		self.keys0    = [0,       -1,  -1,-1,  -1,-1,-1,-1,-1,-1,-1,-1]
		self.keysName = ['staIndex','pTime','sTime']
	def select(self,req):
		return True

class RecordCC(Record):
	def defaultSet(self):
		super().defaultSet()
		self.keysIn   = 'staIndex pTime sTime pCC sCC pM pS sM sS'.split()
	def getPMul(self):
		return (self['pCC']-self['pM'])/self['pS']
	def getSMul(self):
		return (self['sCC']-self['sM'])/self['sS']


defaultStrL='ENZ'
class Quake(Dist):
	def __init__(self,*argv,**kwargs):
		super().__init__(*argv,**kwargs)
		self.records = []
		if not isinstance(self['strTime'],NoneType):
			#print('**',self['strTime'])
			self['time'] = UTCDateTime(self['strTime']).timestamp
		if self['randID'] == None:
			self['randID']=int(10000*np.random.rand(1))
		if self['filename'] == None:
			self['filename'] = self.getFileName() 
	def defaultSet(self):
		#               quake: 34.718277 105.928949 1388535219.080064 num: 7 index: 0    randID: 1    filename: 16071/1388535216_1.mat -0.300000
		super().defaultSet()
		self.keysIn   = 'type     la       lo          time          para0  num  para1 index   para2    randID para3   filename ml   dep'.split()
		self.keys     = 'type     la       lo          time          para0  num  para1 index   para2    randID para3   filename ml   dep stationList strTime no YMD HMS sort'.split()
		self.keysType = 'S        f        f           f             S      F       S  f        S       f      S        S        f    f   l  S S S S S'.split()
		self.keys0    = ['quake',  None,     None,      None,       'num', None,'index',None, 'randID', None,'filename',  None,   None,0 ,'','','time']
		self.keysName = ['time','la','lo']
	def Append(self,tmp):
		if isinstance(tmp,Record):
			self.records.append(tmp)
		else:
			print('please pass in Record type')
	def calCover(self,stationList=[],maxDT=None):
		if len(stationList) ==0:
			stationList = self['stationList']
		if isinstance(stationList,type(None)) or len(stationList)==0:
			print('no stationInfo')
			return None
		coverL=np.zeros(360)
		for record in self.records:
			if record['pTime']==0 and record['sTime']==0:
				continue
			if maxDT!=None:
				if record['pTime']-self['time']>maxDT:
					continue
			staIndex= int(record['staIndex'])
			la      = stationList[staIndex]['la']
			lo      = stationList[staIndex]['lo']
			dep     = stationList[staIndex]['dep']/1e3
			delta,dk,Az = self.calDelta(la,lo,dep)
			R=int(60/(1+dk/200)+60)
			N=((int(Az)+np.arange(-R,R))%360).astype(np.int64)
			coverL[N]=coverL[N]+1
		L=((np.arange(360)+180)%360).astype(np.int64)
		coverL=np.sign(coverL)*np.sign(coverL[L])*(coverL+coverL[L])
		coverRate=np.sign(coverL).sum()/360
		return coverRate
	def getFileName(self,randID=''):
		if len(randID)==0:
			randID = str(self['randID'])
		time = self['time']
		if isinstance(time,UTCDateTime):
			time=time.timestamp
		dayDir = str(int(self['time']/86400))+'/'
		return dayDir+str(int(self['time']))+'_'+randID+'.mat'
	def calDelta(self,la,lo,dep=0):
		D=DistAz(la,lo,self['la'],self['lo'])
		delta=D.getDelta()
		dk=D.degreesToKilometers(delta)
		dk=np.linalg.norm(np.array([dk,self['dep']+dep]))
		Az=D.getAz()
		return delta,dk,Az
	def num(self):
		return len(self.records)
	def __str__(self, *argv):
		self['num'] = self.num()
		return super().__str__(*argv )
	def __lt__(self,self1):
		return self[self['sort']]<self1[self1['sort']]
	def __eq__(self,self1):
		return self[self['sort']]==self1[self1['sort']]
	def staIndexs(self):
		return [record['staIndex'] for record in self.records]
	def getReloc(self,line):
		self['time']=self.tomoTime(line)
		self['la']=float(line[1])
		self['lo']=float(line[2])
		self['dep']=float(line[3])
		return self
	def tomoTime(self,line):
		m=int(line[14])
		sec=float(line[15])
		return UTCDateTime(int(line[10]),int(line[11]),int(line[12])\
			,int(line[13]),m+int(sec/60),sec%60).timestamp
	def resDir(self,resDir):
		return '%s/%s/'%(resDir,self.name(s='_'))
	def select(self,req):
		if 'R' in req:
			if not self.inR(req['R']):
				return False
		if 'time0' in req:
			if self['time']<req['time0']:
				return False
		if 'time1' in req:
			if self['time']>req['time1']:
				return False    
		if 'loc0' in req:
			dist = self.dist(req['loc0'])
			if 'maxDist' in req:
				if dist > req['maxDist']:
					return False
			if 'minDist' in req:
				if dist < req['minDist']:
					return False
		if 'maxDep' in req:
			if self['dep']>req['maxDep']:
				return False
		if 'minDep' in req:
			if self['dep']<req['minDep']:
				return False
		if 'minCover' in req and 'staInfos' in req:
			if self.calCover(req['staInfos'])<req['minCover']:
				return False
		if 'minMl' in req:
			if self['ml']<req['minMl']:
				return False
		if 'minN' in req:
			if len(self.records)<req['minN']:
				return False
		for record in self.records:
			if not record.select(req):
				self.records.pop(self.records.index(record))
		if 'maxRes' in req:
			if 'locator' in req:
				q,res=req['locator'].locate(self)
				if res>req['maxRes'] and self['dep']>50:
					return False
		if 'minSNR' in req:
			T3L = self.loadSacs(req['staInfos'],req['matDir'],f=req['f'])
			count=0
			allCount =0
			for i in range(len(self.records)):
				record = self.records[i] 
				allCount +=1
				if record['pTime']>0:
					T3 = T3L[i]
					dataAt = T3.Data(record['pTime']-0,record['pTime']+3)
					dataBe = T3.Data(record['pTime']-9,record['pTime']-3)
					if len(dataAt)!=0 and len(dataBe)!=0:
						snr = np.abs(dataAt).max()/np.abs(dataBe).max()
						#print('snr %.2f not enough count %d %d'%(snr,count,allCount))
						if snr<req['minSNR']:
							count+=1
							#print('snr %.2f not enough count %d %d'%(snr,count,allCount))
							record['pTime']=0
							record['sTime']=0
					else:
						record['pTime']=0
						record['sTime']=0
			print('snr count %d %d'%(allCount,count))
			self.removeZeros()
			if 'minN' in req:
				if len(self.records)<req['minN']:
					return False
			if 'minCover' in req and 'staInfos' in req:
				if self.calCover(req['staInfos'])<req['minCover']:
					return False
		
				
		return True
	def __setitem__(self,key,value):
		super().__setitem__(key,value)
		if key == 'time' :
			self['strTime'] = UTCDateTime(self['time']).strftime('%Y:%m:%d %H:%M:%S.%f')
		if key =='HMS' and self['YMD']!='' and self['HMS']!='':
			self['time'] = UTCDateTime(self['YMD'] + ' ' + self['HMS'])
	def __getitem__(self,key):
		if key=='num':
			return len(self.records)
		return super().__getitem__(key) 
	def removeZeros(self):
		records = []
		for record in self.records:
			if record['pTime']>0 or record['sTime']>0:
				records.append(record)
		self.records = records
	def saveSacs(self,staL, staInfos, matDir='output/'\
	,bSec=-10,eSec=40,dtype=np.float32):
		eventDir = matDir+'/'+self['filename'].split('.mat')[0]+'/'
		loc=self.loc
		if not os.path.exists(eventDir):
			os.makedirs(eventDir)
		ml=0
		T3L =[]
		for i in range(len(self.records)):
			record = self.records[i]
			staIndex = record['staIndex']
			pTime = record['pTime']
			sTime = record['sTime']
			bTime = self['time']+bSec
			eTime = max(self['time']+(pTime-self['time'])*1.7,sTime)+eSec
			bTime,eTime=staL[staIndex].data.getTimeLim(bTime,eTime)
			filenames=staL[staIndex].sta.baseSacName(resDir=eventDir)
			T3 = staL[staIndex].data.slice(bTime,eTime,nearest_sample=True)
			T3L.append(T3)
			if T3.bTime<0:
				continue
			T3.adjust(kzTime=self['time'],pTime=pTime,sTime=sTime,net=staL[staIndex].sta['net'],\
				sta=staL[staIndex].sta['sta'],stloc=staL[staIndex].sta.loc(),eloc=self.loc())
			T3.write(filenames)
		return self.calML(staInfos=staInfos,T3L=T3L)
	def loadSacs(self,staInfos,matDir='output',\
		f=[-1,-1],filtOrder=2,randID='',delta0=-1):
		T3L=[]
		filename = self['filename']
		if len(randID)>0:
			filename = self.getFileName(randID=randID)
		eventDir = matDir+'/'+filename.split('.mat')[0]+'/'
		for record in self.records:
			staIndex = record['staIndex']
			sacsFile = staInfos[staIndex].baseSacName(resDir=eventDir)
			sacFilesL = [glob(tmp) for tmp in sacsFile]
			T3L.append(getTrace3ByFileName(sacFilesL,pTime=record['pTime'],\
				sTime=record['sTime'],delta0=delta0))
			T3L[-1].filt(f,filtOrder)
		return T3L
	def plotSacs(self,T3L,fileDir='',channel=2,A=12,linewidth=0.3,key='ZGKX',quakeRef='',T3LRef='',threshold=0.4,alpha=1):
		bSec=-3
		eSec=+4
		if not os.path.exists(fileDir):
			os.makedirs(fileDir)
		filename = '%s/%d.pdf'%(fileDir,self['time'])
		print(filename)
		fs.init(key)
		plt.close()
		plt.figure(figsize=[3.5,3.2])
		count=0
		if len(quakeRef)>0:
			indexLRef = self.staIndexs()
		for T3 in T3L:
			countRef=-1
			if len(T3)<0:
				continue
			loc=[T3[0].stats['sac']['stla'],T3[0].stats['sac']['stlo']]
			dist = self.dist(loc)
			data = T3.Data()[:,channel]
			maxA = np.abs(data).max()
			data*= A/maxA
			timeL = T3.bTime.timestamp+np.arange(len(data))*T3.Delta()-self['time']
			plt.plot(timeL,data+dist,'gray',linewidth=linewidth)
			record = self.records[count]
			index  = record['staIndex']
			if len(quakeRef)>0:
				countRef = indexLRef.index(index)
			if T3.pTime>self['time']:
				dTp =  T3.pTime.timestamp - self['time']
				plt.plot([dTp,dTp],[dist-A,dist+A],'b',linewidth=linewidth)
				if len(quakeRef)>0 and countRef>=0:
					T3Ref = T3LRef[countRef]
					if T3Ref.pTime>quakeRef['time']:
						T3Ref = T3LRef[countRef]
						data0 = T3.Data(T3.pTime.timestamp+bSec,T3.pTime.timestamp+eSec)[:,channel]
						data1  = T3Ref.Data(T3Ref.pTime.timestamp+bSec,T3Ref.pTime.timestamp+eSec)[:,channel]
						data1  *= data0.std()/data1.std()
						data1  *=A/maxA
						if record['pCC']>threshold:
							color ='k'
						else:
							color = 'g'
						plt.plot(T3.pTime.timestamp+np.arange(len(data1))*T3.Delta()-self['time']+bSec,data1+dist,color,linewidth=linewidth,alpha=alpha)
			if T3.sTime>self['time']:
				dTs =  T3.sTime.timestamp - self['time']
				plt.plot([dTs,dTs],[dist-A,dist+A],'r',linewidth=linewidth)
				if len(quakeRef)>0 and countRef>=0:
					T3Ref = T3LRef[countRef]
					if T3Ref.sTime>quakeRef['time']:
						data0 = T3.Data(T3.sTime.timestamp+bSec,T3.sTime.timestamp+eSec)[:,channel]
						data1  = T3Ref.Data(T3Ref.sTime.timestamp+bSec,T3Ref.sTime.timestamp+eSec)[:,channel]
						data1  *= data0.std()/data1.std()
						data1  *=A/maxA
						if record['sCC']>threshold:
							color ='k'
						else:
							color = 'g'
						plt.plot(T3.sTime.timestamp+np.arange(len(data1))*T3.Delta()-self['time']+bSec,data1+dist,color,linewidth=linewidth,alpha=alpha)
			count+=1
		plt.xlim([-5,70])
		plt.xlabel('t/s')
		plt.ylabel('distance/km')
		plt.savefig(filename,dpi=300)

	def loadPSSacs(self,staInfos,matDir='output',\
		isCut=False,index0=-250,index1=250,\
		f=[-1,-1],filtOrder=2,randID='',delta0=-1):
		print(self)
		T3L = self.loadSacs(staInfos,f=f,filtOrder=filtOrder,matDir=matDir,randID=randID,delta0=delta0)
		T3PL=[]
		T3SL=[]
		for T3 in T3L:
			T3PL.append(T3.slice(T3.pTime+index0*T3.delta,\
				T3.pTime+index1*T3.delta))
			T3SL.append(T3.slice(T3.sTime+index0*T3.delta,\
				T3.sTime+index1*T3.delta))
		return T3PL,T3SL
	def loadPSSacsQuick(self,staInfos,matDir='output',\
		isCut=False,index0=-250,index1=250,\
		f=[-1,-1],filtOrder=2):
		print(self)
		#  not faster
		T3L = self.loadSacs(staInfos,filtOrder=filtOrder,matDir=matDir)
		T3PL=[]
		T3SL=[]
		for T3 in T3L:
			T3P=T3.slice(T3.pTime+index0*T3.delta-3,\
				T3.pTime+index1*T3.delta+3)
			T3P.filt(f,filtOrder)
			T3PL.append(T3P.slice(T3.pTime+index0*T3.delta,\
				T3.pTime+index1*T3.delta))
			T3S=T3.slice(T3.sTime+index0*T3.delta-3,\
				T3.sTime+index1*T3.delta+3)
			T3S.filt(f,filtOrder)
			T3SL.append(T3S.slice(T3.sTime+index0*T3.delta,\
				T3.sTime+index1*T3.delta))
		return T3PL,T3SL
	def cutSac(self, stations,bTime=-100,eTime=3000,resDir = 'eventSac/',para={},byRecord=True,isSkip=False):
		time0  = self['time'] + bTime
		time1  = self['time'] + eTime
		tmpDir = self.resDir(resDir)
		#print(tmpDir)
		if not os.path.exists(tmpDir):
			os.makedirs(tmpDir)
		staIndexs = self.staIndexs()
		for staIndex in range(len(stations)):
			station = stations[staIndex]
			if len(staIndexs) >0 and staIndex not in staIndexs and byRecord:
				continue
			if staIndex in staIndexs:
				record = self.records[staIndexs.index(staIndex)]
			resSacNames = station.baseSacName(tmpDir)
			isF = True
			for resSacName in resSacNames:
				if not os.path.exists(resSacName):
					isF = False
			if isF and isSkip:
				#print(resSacNames,'done')
				continue
			sacsL = station.getFileNames(time0,time1+2)
			#print(sacsL)
			for i in range(3):
				sacs = sacsL[i]
				if len(sacs) ==0:
					continue
				data = mergeSacByName(sacs, **para)
				if isinstance(data,NoneType):
					continue
				data.data -= data.data.mean()
				data.detrend()
				#print(data)
				#print(data.stats.starttime.timestamp-time0,\
				#    data.stats.endtime.timestamp-time1)
				if data.stats.starttime<=time0 and data.stats.endtime >= time1:
					data=data.slice(starttime=UTCDateTime(time0), \
						endtime=UTCDateTime(time1), nearest_sample=True)
					#print('#',data.stats.starttime.timestamp-time0)
					#print('##',data.stats.endtime.timestamp -time1)
					#print('###',time0 -time1)
					#print('####',data.stats.starttime.timestamp-data.stats.endtime.timestamp)
					decMul=-1
					if 'delta0' in para:
						decMul = para['delta0']/data.stats.delta
						if np.abs(int(decMul)-decMul)<0.001:
							decMul=decMul
							#print('decMul: ',decMul)
						else:
							decMul=-1
					data=adjust(data,decMul=decMul,stloc=station.loc(),eloc = self.loc(),\
						kzTime=self['time'],sta = station['sta'],net=station['net'])
					data.write(resSacNames[i],format='SAC')
		print(tmpDir,len(glob(tmpDir+'/*Z')))
		return None
	def calML(self, staInfos=[],minSACount=3,T3L=None,minCover=-10.0,minDk=20,maxDk=3000):
		def getSA(data):
			data=data-data.mean()
			return np.abs(data.cumsum(axis=0)).max(axis=0)[:2].mean()
		ml = 0
		sACount=0
		if self.calCover(staInfos)<minCover:
			return -999
		for i in range(len(self.records)):
			record = self.records[i]
			T3 =  T3L[i]
			if T3.bTime <=0 or  record['pTime']<=0 :
				continue
			mTime = record['pTime']
			data = T3.Data(mTime-5,mTime+30)
			if len(data)==0:
				continue
			if T3.delta<0:
				continue
			sA  = getSA(data)*T3.delta#/staInfos[record['staIndex']]['sensitivity']\
			#*1e3#1/sensitivity m/s*1000mm/m delta_s
			dk = self.dist(staInfos[record['staIndex']])
			if dk<minDk:
				continue
			if dk>maxDk:
				continue
			ml+=np.log10(sA)+1.1*np.log10(dk)+0.00189*dk-2.09#-0.23#np.log10(sA)+2.76*np.log10(dk)-2.48
			sACount+=1
		if sACount<minSACount:
			ml=-999
		else:
			ml/=sACount
		return ml
	def __len__(self):
		return len(self.records)
	def getSacFiles(self,stations,isRead=False,resDir = 'eventSac/',strL='ENZ',\
		byRecord=True,maxDist=-1,minDist=-1,remove_resp=False,isPlot=False,\
		isSave=False,respStr='_remove_resp',para={},isSkip=False):
		sacsL = []
		staIndexs = self.staIndexs()
		tmpDir = self.resDir(resDir)
		para0 ={\
		'delta0'    :0.02,\
		'freq'      :[-1, -1],\
		'filterName':'bandpass',\
		'corners'   :2,\
		'zerophase' :True,\
		'maxA'      :1e18,\
		'output': 'VEL',\
		'toDisp': False
		}
		para0.update(para)
		print(para0)
		para = para0
		respStr += '_'+para['output']
		respDone = False
		if para['freq'][0] > 0:
			print('filting ',para['freq'])
		if 'pre_filt' in para:
			print('pre filting ',para['pre_filt'])
		if 'output' in para:
			print('outputing ',para['output'])
		for staIndex in range(len(stations)):
			station = stations[staIndex]
			if remove_resp and station['oRemove'] ==False:
				sensorName,dasName,sensorNum = station.getSensorDas()
				if sensorName=='UNKNOWN' or sensorName==''  \
				or dasName=='UNKNOWN' or dasName=='':
					continue
				if station['net']=='YP' or station['nameMode'] == 'CEA':
					rigthResp = True
					for channelIndex in range(3):
						#channelIndexO = defaultStrL.index(strL[channelIndex])
						if station.sensor[channelIndex][0].code != station['net'] or \
							station.sensor[channelIndex][0][0].code != station['sta'] or \
							station.sensor[channelIndex][0][0][0].code != station['comp'][channelIndex]:
							print('no Such Resp')
							print(station.sensor[channelIndex][0][0][0].code,station['comp'][channelIndex])
							#time.sleep(1)
							rigthResp=False
					if rigthResp == False:
						print('#### no Such Resp')
						continue
				if self['time']<UTCDateTime(2009,7,1).timestamp and station['nameMode']=='CEA'\
					and len(station.sensor[0])<2:
					#print('noOld')
					continue
			if len(staIndexs) > 0 and staIndex not in staIndexs and byRecord:
				continue
			if staIndex in staIndexs:
				record = self.records[staIndexs.index(staIndex)]
			if maxDist>0 and self.dist(station)>maxDist:
				continue
			if minDist>0 and self.dist(station)<minDist:
				continue
			resSacNames = station.baseSacName(tmpDir,strL=strL)
			respDone = False
			if remove_resp and isSave==False and station['oRemove'] ==False:
				isF = True
				resSacNames = station.baseSacName(tmpDir,strL=strL,infoStr=respStr)
				for resSacName in resSacNames:
					if not os.path.exists(resSacName):
						isF = False
						break
				if isF:
					respDone = True
					#print(station,'done')
				else:
					resSacNames = station.baseSacName(tmpDir,strL=strL)
			if remove_resp and isSkip==True and isSave==True and station['oRemove'] ==False:
				isF = True
				resSacNamesTmp = station.baseSacName(tmpDir,strL=strL,infoStr=respStr)
				for resSacName in resSacNamesTmp:
					if not os.path.exists(resSacName):
						isF = False
						#print('need do')
						break
				if isF:
					#print(station,'done')
					continue   
			isF = True
			for resSacName in resSacNames:
				if not os.path.exists(resSacName):
					isF = False
					#print('no origin sac')
					break
			if isF == True:
				if remove_resp and respDone == False and station['oRemove'] ==False and isSave == False \
				and isSkip:
					continue
				if isRead:
					#print(resSacNames)
					try:
						sacsL.append([ obspy.read(resSacName)[0] for resSacName in resSacNames])
						if isPlot:
							plt.close()
							for i in range(3):
								plt.subplot(3,1,i+1)
								data = sacsL[-1][i].data
								delta= sacsL[-1][i].stats['sac']['delta']
								timeL = np.arange(len(data))*delta+sacsL[-1][i].stats['sac']['b']
								plt.plot(timeL,data/data.std(),'b',linewidth=0.5)
						if remove_resp and respDone==False and station['oRemove'] ==False:
							print('remove_resp ',station)
							
							for channelIndex in range(len(strL)):
								sac = sacsL[-1][channelIndex]
								channelIndexO = defaultStrL.index(strL[channelIndex])
								sensor = station.sensor[channelIndexO]
								if self['time'] >= UTCDateTime(2009,7,1).timestamp and station['nameMode']=='CEA':
									sensor = station.sensor[channelIndexO][:1]
								if self['time'] < UTCDateTime(2009,7,1).timestamp and station['nameMode']=='CEA':
									sensor = station.sensor[channelIndexO][1:]
								das    = station.das[channelIndexO]
								originStats={}
								for key in station.defaultStats:
									originStats[key] = sac.stats[key]
								if station['net'] == 'hima':
									sac.stats.update(station.defaultStats)
								#print(sac.stats)
								if station['net'] == 'YP' or station['nameMode'] == 'CEA':
									sac.stats.update({'channel': station['compBase']+strL[channelIndex]})
									sac.stats.update({'knetwk': station['net'],'network': station['net']})
								#print(sac.stats,sensor[0][0][0],sensor[1][0][0])
								if 'pre_filt' in para:
									sac.remove_response(inventory=sensor,\
										output=para['output'],water_level=60,\
										pre_filt=para['pre_filt'])
								else:
									sac.remove_response(inventory=sensor,\
										output=para['output'],water_level=60)                           
								sac.stats.update(station.defaultStats)
								if station['nameMode'] != 'CEA':
									sac.remove_response(inventory=das,\
										output="VEL",water_level=60)
								sac.stats.update(originStats)
							if isPlot:
								for i in range(3):
									plt.subplot(3,1,i+1)
									data = sacsL[-1][i].data
									delta= sacsL[-1][i].stats['sac']['delta']
									timeL = np.arange(len(data))*delta+sacsL[-1][i].stats['sac']['b']
									plt.plot(timeL,data/data.std(),'r',linewidth=0.5)
					except:
						print('no resp continue')
						continue
					else:
						pass
					
					for sac in sacsL[-1]:
						sac.detrend()
						sac.data -= sac.data.mean()

					if para['toDisp']:
						for sac in sacsL[-1]:
							sac.integrate()
							if np.random.rand()<0.01:
								print('integrate to Disp')
					
					for sac in sacsL[-1]:
						sac.detrend()
						sac.data -= sac.data.mean()

					if para['freq'][0] > 0  and station['doFilt']:
						for sac in sacsL[-1]:
							if para['filterName']=='bandpass':
								if np.random.rand()<0.01:
									print('do do bandpass**************************')
								sac.filter(para['filterName'],\
									freqmin=para['freq'][0], freqmax=para['freq'][1], \
									corners=para['corners'], zerophase=para['zerophase'])
							elif para['filterName']=='highpass':
								sac.filter(para['filterName'],\
									freq=para['freq'][0],  \
									corners=para['corners'], zerophase=para['zerophase'])
							elif para['filterName']=='lowpass':
								sac.filter(para['filterName'],\
									freq=para['freq'][0],  \
									corners=para['corners'], zerophase=para['zerophase'])
					if isPlot:
						plt.savefig(resSacNames[0]+'.jpg',dpi=200)
					if isSave and remove_resp and station['oRemove'] ==False:
						resSacNames = station.baseSacName(tmpDir,strL=strL,infoStr=respStr)
						for i in range(3):
							sacsL[-1][i].write(resSacNames[i],format='SAC')
				else:
					sacsL.append(resSacNames)
		return sacsL
	def nearest(self,staInfos,N=20,near=45,delta=0.01):
		distL = []
		azL    = []
		if len(self.records)<=N:
			return
		for record in self.records:
			staInfo = staInfos[record['staIndex']]
			distL.append(self.dist(staInfo))
			azL.append(self.az(staInfo))
		distL = np.array(distL)
		azL   = np.array(azL)
		DL    = [] 
		for i in range(len(azL)):
			mul =np.sum(np.abs((azL-azL[i]+180)%360-180)<near)-1+delta
			DL.append(distL[i]*mul)
		count=0
		records =[]
		for i in np.array(DL).argsort():
			records.append(self.records[i])
			count+=1
			if count==N:
				break
		self.records = records
		return 



class QuakeCC(Quake):
	def defaultSet(self):
		#               quake: 34.718277 105.928949 1388535219.080064 num: 7 index: 0    randID: 1    filename: 16071/1388535216_1.mat -0.300000
		super().defaultSet()
		self.keysIn   = 'type   la       lo          time          para0 num para1 index para2 randID para3 filename cc M S ml   dep tmpName'.split()
		self.keys     = 'type   la       lo          time          para0 num \
		para1 index para2 randID para3 filename tmpName ml   dep stationList strTime no YMD HMS cc M S sort'.split()
		self.keysType = 'S      f        f           f             S     F     \
		S     f     S      f      S     S        S      f    f        l              S     S   S  S  f   f f S'.split()
		self.keys0    = [None,  None,     None,      None,         None, None,None,None, None, None,  None,  None,   None,0 ,'','',10,1,0.1,'UN','time']
		self.keysName = ['time','la','lo']
	def getMul(self):
		return (self['cc']-self['M'])/self['S']

class QuakeL(list):
	def __init__(self,*argv,**kwargs):
		super().__init__()
		self.inQuake = {}
		self.inRecord= {}
		self.keys = ['#','*','q-','d-',' ',' ']
		if 'Quake' in kwargs:
			self.Quake = kwargs['Quake']
		else:
			self.Quake = Quake
		if 'Record' in kwargs:
			self.Record = kwargs['Record']
		else:
			self.Record = Record
		if 'mode' in kwargs:
			if kwargs['mode']=='CC':
				self.Quake= QuakeCC
				self.Record=RecordCC
		if 'quakeKeysIn' in kwargs:
			self.inQuake['keysIn'] = kwargs['quakeKeysIn']
		if 'recordKeysIn' in kwargs:
			self.inRecord['keysIn'] = kwargs['recordKeysIn']
		if 'keys' in kwargs:
			self.keys = kwargs['keys']
		if 'quakeSplitKey' in kwargs:
			self.inQuake['splitKey'] = kwargs['quakeSplitKey']
		if 'recordSplitKey' in kwargs:
			self.inRecord['splitKey'] = kwargs['recordSplitKey']
		if len(argv)>0:
			if isinstance(argv[0],str):
				self.read(argv[0],**kwargs)
			elif isinstance(argv[0],list):
				for tmp in argv[0]:
					self.append(tmp)
		if 'file' in kwargs:
			self.read(kwargs['file'],**kwargs)
	def __add__(self,self1):
		for quake in self1:
			self.append(quake)
		return self
	def set(self,key,value):
		for tmp in self:
			tmp[key] = value
	def __getitem__(self,index):
		quakesNew = super().__getitem__(index)
		if isinstance(index,slice):
			quakesNew = QuakeL(quakesNew)
			quakesNew.inQuake = self.inQuake
			quakesNew.inRecord = self.inRecord
			quakesNew.kyes = self.keys
		return quakesNew
	def find(self,quake0):
		for i in range(len(self)):
			dTime = np.abs(self[i]['time']-quake0['time'])
			dLa = np.abs(self[i]['la']-quake0['la'])
			dLo = np.abs(self[i]['lo']-quake0['lo'])
			if dTime<10 and dLa<0.5 and dLo<0.5:
				return i
		return -1
	def Find(self,value,key='filename'):
		for q in self:
			if q[key]==value:
				return q
		return None
	def read(self,file,**kwargs):
		if 'keys' in kwargs:
			self.keys = kwargs['keys']
		if 'quakeSplitKey' in kwargs:
			self.inQuake['splitKey'] = kwargs['quakeSplitKey']
		if 'recordSplitKeys' in kwargs:
			self.inRecord['splitKey'] = kwargs['recordSplitKey']
		if '*' in file or '?' in file:
			for tmpFile in glob(file):
				self.read(tmpFile,**kwargs)
			return
		with open(file,'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.strip()
			if len(line)<0:
				continue
			if line[0] == '^':
				self.keys = line[1:].split()
				if len(self.keys) >=6:
					self.inQuake['splitKey']  = self.keys[4]
					self.inRecord['splitKey'] = self.keys[5]
				continue
			if line[0] in self.keys[0]:
				self.inQuake['keysIn'] = line[1:]
				continue
			if line[0] in self.keys[1]:
				self.inRecord['keysIn'] = line[1:]
				continue
			if line[0] in self.keys[2]:
				self.inQuake['line'] = line
				self.append(self.Quake(**self.inQuake))
				continue
			if line[0] in self.keys[3]:
				continue
			#print(line[0],self.keys)
			self.inRecord['line'] = line
			self[-1].Append(self.Record(**self.inRecord))
	def write(self,file,**kwargs):
		if 'quakeSplitKey' in kwargs:
			self.inQuake['splitKey'] = kwargs['quakeSplitKey']
		if 'recordSplitKeys' in kwargs:
			self.inRecord['splitKey'] = kwargs['recordSplitKey']
		with open(file,'w+') as f:
			f.write('^')
			for key in self.keys:
				f.write(key+' ')
			f.write('\n')
			if 'quakeKeysIn' in kwargs:
				self.inQuake['keysIn'] = kwargs['quakeKeysIn']
			if 'recordKeysIn' in kwargs:
				self.inRecord['keysIn'] = kwargs['recordKeysIn']              
			if 'keysIn' in self.inQuake:
				tmp = self.inQuake['keysIn'].split()
				for quake in self:
					quake.keysIn = tmp
			if 'keysIn' in self.inRecord:
				tmp = self.inRecord['keysIn'].split()
				for quake in self:
					for record in quake.records:
						record.keysIn =  tmp
			quakeKeysIn = ''
			recordKeysIn= ''
			for quake in self:
				if quake.keyIn() != quakeKeysIn:
					quakeKeysIn = quake.keyIn()
					f.write('#%s\n'%quakeKeysIn)
				f.write('%s\n'%quake)
				for record in quake.records:
					if record.keyIn() != recordKeysIn:
						recordKeysIn = record.keyIn()
						f.write('*%s\n'%recordKeysIn)
					f.write('%s\n'%record)
	def select(self,req):
		count =0
		count0 = 0
		for index in range(len(self)-1,-1,-1):
			count0+=1
			quake = self[index]
			if not quake.select(req):
				self.pop(index)
			else:
				count+=1
				print('find %d in %d'%(count,count0), quake)
		#quakes = self.copy()
		#self.clear()
		#for i in index:
		#    self.append(quakes[i])
	def cutSac(self, *argv,**kwargs):
		for quake in self:
			print(quake)
			quake.cutSac(*argv,**kwargs)
	def copy(self):
		quakes = QuakeL()
		for quake in self:
			quakes.append(quake.copy())
		quakes.keys         = self.keys.copy()
		quakes.inQuake      = self.inQuake.copy()
		quakes.inRecord     = self.inRecord.copy()
		return quakes
	def getSacFiles(self,*argv,**kwargs):
		sacsL = []
		for quake in self:
			sacsL+=quake.getSacFiles(*argv,**kwargs)
		return sacsL
	def paraL(self,keyL=['time','la','lo','dep','ml'],req=None):
		pL = {key: []for key in keyL}
		for quake in self:
			if not isinstance(req,NoneType):
				if not quake.select(req):
					continue
			for key in keyL:
				pL[key].append(quake[key])
		return pL
	def compare(self,self1,maxDt=10,maxDd=0.3):
		paraL0 = self.paraL(keyL=['time','la','lo'])
		time0  = np.array(paraL0['time'])
		la0    = np.array(paraL0['la'])
		lo0    = np.array(paraL0['lo'])
		paraL1 = self1.paraL(keyL=['time','la','lo'])
		time1  = np.array(paraL1['time'])
		la1    = np.array(paraL1['la'])
		lo1    = np.array(paraL1['lo'])
		index0L= []
		index1L= []
		m0L    = []
		m1L    = []
		for index0 in range(len(self)):
			T0 = time0[index0]
			La0= la0[index0]
			Lo0= lo0[index0]
			absDTime = np.abs(time1-T0)
			absDd =((la1-La0)**2+(lo1-Lo0)**2)**0.5
			diff = (absDTime>maxDt)*1000+ (absDd>maxDd)*1000+absDTime+absDd*110/5
			if diff.min()<120:
				index1 = diff.argmin()
				index0L.append(index0)
				index1L.append(index1)
				m0L.append(self[index0]['ml'])
				m1L.append(self1[index1]['ml'])
		return index0L,index1L,m0L,m1L            
	def analy(self):
		eCount = len(self)
		pCount = 0
		sCount = 0
		for quake in self:
			for record in quake.records:
				if record['pTime']>0:
					pCount+=1
				if record['sTime']>0:
					sCount+=1
		print('event: %d | P phase: %d | S phase: %d|'%(eCount,pCount,sCount))
		return eCount,pCount,sCount
	def adjustMl(self,a=0.05,b=-0.9,c=-0.3,maxMl=4.2):
		for q in self:
			if q['ml']>-10 and q['ml']<10:
				if q['ml']<maxMl:
					q['ml']+=a+b*q['ml']
				else:
					q['ml']+=c
	def dis_time(self,staInfos):
		pTime=[]
		pDist=[]
		sTime=[]
		sDist=[]
		for quake in self:
			for record in quake.records:
				dist = quake.dist(staInfos[record['staIndex']])
				dist = (dist**2+(quake['dep']+staInfos[record['staIndex']]['dep']/1000)**2)**0.5
				if record['pTime']>quake['time']:
					pTime.append(record['pTime']-quake['time'])
					pDist.append(dist)
				if record['sTime']>quake['time']:
					sTime.append(record['sTime']-quake['time'])
					sDist.append(dist)
		return pDist,pTime,sDist,sTime
	def Slice(self,bTime,eTime):
		self1 = QuakeL([])
		for quake in self:
			if quake['time']>bTime and  quake['time']<eTime:
				self1.append(quake)
		return self1
def adjust(data,stloc=None,kzTime=None,tmpFile='test.sac',decMul=-1,\
	eloc=None,chn=None,sta=None,net=None,o=None,pTime=None,sTime=None):
	if decMul>1 :
		decMul = int(decMul)
		if decMul<16:
			data.decimate(int(decMul),no_filter=False)#True
		else:
			if decMul%2==0:
				data.decimate(2,no_filter=False)
				decMul/=2
			if decMul%2==0:
				data.decimate(2,no_filter=False)
				decMul/=2
			if decMul%5==0:
				data.decimate(5,no_filter=False)
				decMul/=5
			if decMul%5==0:
				data.decimate(5,no_filter=False)
				decMul/=5
			if decMul>1 :
				data.decimate(int(decMul),no_filter=False)
			if np.random.rand()<0.01:
				print(data)
	if data.stats['_format']!='SAC':
		tmp=sac.SACTrace.from_obspy_trace(data)
		data=tmp.to_obspy_trace()
		#data.write(tmpFile,format='SAC')
		#data=obspy.read(tmpFile)[0]
		#print(data)
	if stloc!=None:
		data.stats['sac']['stla']=stloc[0]
		data.stats['sac']['stlo']=stloc[1]
		data.stats['sac']['stel']=stloc[2]
	if eloc!=None:
		data.stats['sac']['evla']=eloc[0]
		data.stats['sac']['evlo']=eloc[1]
		data.stats['sac']['evdp']=eloc[2]
		dis=DistAz(eloc[0],eloc[1],stloc[0],stloc[1])
		dist=dis.degreesToKilometers(dis.getDelta())
		data.stats['sac']['dist']=dist
		data.stats['sac']['az']=dis.getAz()
		data.stats['sac']['baz']=dis.getBaz()
		data.stats['sac']['gcarc']=dis.getDelta()
	if chn!=None:
		data.stats['sac']['kcmpnm']=chn
		data.stats['channel']=chn
	if sta!=None and net !=None:
		data.stats['sac']['kstnm']=sta
		data.stats['station']=sta
		data.stats['sac']['knetwk']=net
		data.stats['network']=net
	if o!=None:
		data.stats['sac']['o']=o
	if kzTime!=None:
		kzTime=UTCDateTime(kzTime)
		data.stats['sac']['nzyear'] = int(kzTime.year)
		data.stats['sac']['nzjday'] = int(kzTime.julday)
		data.stats['sac']['nzhour'] = int(kzTime.hour)
		data.stats['sac']['nzmin']  = int(kzTime.minute)
		data.stats['sac']['nzsec']  = int(kzTime.second)
		data.stats['sac']['nzmsec'] = int(kzTime.microsecond/1000)
		data.stats['sac']['b']      = data.stats.starttime.timestamp-kzTime.timestamp
		data.stats['sac']['e']      = data.stats['sac']['b']+(data.data.size-1)*data.stats.delta
		#tmp=sac.SACTrace.from_obspy_trace(data)
		#data=tmp.to_obspy_trace()
	if pTime !=None:
		if pTime>data.stats.starttime.timestamp and pTime <data.stats.endtime.timestamp:
			data.stats['sac']['t0']=data.stats['sac']['b']+pTime-data.stats.starttime.timestamp
			#print(data.stats['sac']['t0'])
	if sTime !=None:
		if sTime>data.stats.starttime.timestamp and sTime <data.stats.endtime.timestamp:
			data.stats['sac']['t1']=data.stats['sac']['b']+sTime-data.stats.starttime.timestamp
		#data.write(tmpFile)
		#data=obspy.read(tmpFile)[0]
		#print(data.stats['sac']['b'],data.stats['sac']['e'])
	return data



def mergeSacByName(sacFileNames, **kwargs):
	para ={\
	'delta0'    :0.02,\
	'freq'      :[-1, -1],\
	'filterName':'bandpass',\
	'corners'   :2,\
	'zerophase' :True,\
	'maxA'      :1e18,\
	}
	count       =0
	sacM        = None
	tmpSacL     =None
	para.update(kwargs)
	#print(ctime(),'reading')
	for sacFileName in sacFileNames:
		try:
			#print(ctime(),'read sac')
			tmpSacs=obspy.read(sacFileName, debug_headers=True,dtype=np.float32)
			#print(ctime(),'read sac done')
			if para['freq'][0] > 0:
				tmpSacs.detrend('demean').detrend('linear').filter(para['filterName'],\
						freqmin=para['freq'][0], freqmax=para['freq'][1], \
						corners=para['corners'], zerophase=para['zerophase'])
			else:
				tmpSacs.detrend('demean').detrend('linear')
		except:
			print('wrong read sac')
			continue
		else:
			if tmpSacL==None:
				tmpSacL=tmpSacs
			else:
				tmpSacL+=tmpSacs
	#print(ctime(),'read done')
	if tmpSacL!=None and len(tmpSacL)>0:
		ID=tmpSacL[0].id
		for tmp in tmpSacL:
			try:
				tmp.id=ID
			except:
				pass
			else:
				pass
		try:
			#print('start merge')
			sacM=tmpSacL.merge(fill_value=0,method=1,interpolation_samples=0)[0]
			#print('end merge')
			std=sacM.std()
			if std>para['maxA']:
				#if printDetail:
				print('#####too many noise std : %f#####'%std)
				sacM=None
			else:
				if printDetail:
					print('#####not too many noise std : %f#####'%std)
				pass
		except:
			print('wrong merge')
			sacM=None
		else:
			pass
			
	return sacM

class Noises:
	def __init__(self,noisesL,mul=0.2):
		for noises in noisesL:
			for noise in noises:
				noise.data /= (noise.data.std()+1e-15)
		self.noisesL  = noisesL
		self.mul = mul
	def __call__(self,sacsL,channelL=[0,1,2]):
		for sacs in sacsL:
			for i in channelL:
				self.noiseSac(sacs[i],i)
	def noiseSac(self,sac,channelIndex=0):
		noise = random.choice(self.noisesL)[channelIndex]
		nSize = noise.data.size
		sSize = sac.data.size
		randI = np.random.rand()*nSize
		randL = (np.arange(sSize)+randI)%nSize
		sac.data+=(np.random.rand()**3)*noise.data[randL.astype(np.int)]\
		*self.mul*sac.data.std()


class PZ:
	def __init__(self,poles,zeros):
		self.poles=poles
		self.zeros=zeros
	def __call__(self,f):
		output = f.astype(np.complex)*0+1
		for pole in self.poles:
			output/=(f-pole)
		for zero in self.zeros:
			output*=(f-zero)
		return output

def sacFromO(sac):
	if sac.stats['sac']['b']>0:
		kzTime = sac.stats.starttime-sac.stats['sac']['b']
		d  = int((sac.stats['sac']['b']+1)/sac.stats['sac']['delta'])*\
			sac.stats['sac']['delta']
		dN = int(d/sac.stats['sac']['delta'])
		N  = sac.data.size
		data = np.zeros(dN+N)
		data[dN:] = sac.data
		sac.data = data
		sac.stats.starttime -=d
		sac = adjust(sac,kzTime=kzTime.timestamp)
	kzTime= sac.stats.starttime - sac.stats['sac']['b']
	endtime  = sac.stats.endtime
	sac=sac.slice(starttime=kzTime, \
		endtime=endtime, nearest_sample=True)
	sac=adjust(sac,kzTime=kzTime.timestamp)

	return sac

from obspy.taup import TauPyModel
from scipy import interpolate
iasp91 = TauPyModel(model="iasp91")

class taup:
	def __init__(self,quickFile='iasp91_time',phase_list=['P','p'], \
		recal=False):
		self.dep = np.arange(0,1500,25)
		self.dist= np.arange(0.1,180,2)
		quickFileNew = quickFile
		for phase in phase_list:
				quickFileNew += phase
		if not os.path.exists(quickFileNew) or recal:
			self.genFile(quickFileNew,phase_list)
		self.M = np.loadtxt(quickFileNew)
		self.interpolate = interpolate.interp2d(self.dep,self.dist,\
			self.M,\
			kind='linear',bounds_error=False,fill_value=1e-8)
	def genFile(self,quickFile='iasp91_timeP',phase_list=['P','p'] ):
		M = np.zeros([len(self.dist),len(self.dep)])
		time=0
		for i  in range(len(self.dep)):
			print(i)
			for j in range(len(self.dist)) :
				arrs=iasp91.get_travel_times(source_depth_in_km=self.dep[i],\
					distance_in_degree=self.dist[j],\
					phase_list=phase_list)
				if len(arrs)>0:
					time = arrs[0].time
				else:
					print(self.dep[i],self.dist[j],time)
				M[j,i] = time
				
		np.savetxt(quickFile,M)
	def __call__(self,dep,dist):
		return self.interpolate(dep,dist)
t0=UTCDateTime(1900,1,1)
t1=UTCDateTime(2099,1,1)
class Trace3(obspy.Stream):
	def __init__(self,traces=[],pTime=-1,sTime=-1,delta=-1,bTime=t0,\
		eTime=t1,compStr = 'ENZ',isData=False):
		super().__init__(traces)
		if len(self)==0:
			self.dis=180*110
			self.pTime=pTime
			self.sTime=sTime
			self.bTime,self.eTime=[t0,t0]
			self.delta=-1
			self.data=np.zeros([0,3])
			return
		self.compStr =compStr
		self.dis=180*110
		self.pTime=pTime
		self.sTime=sTime
		self.pTime,self.sTime=self.getPSTime()
		self.delta=self.Delta()
		self.bTime,self.eTime=self.getTimeLim(bTime=t0,\
			eTime=t1,delta=delta)
		if self.bTime>=self.eTime:
			self.dis=180*110
			self.pTime=pTime
			self.sTime=sTime
			self.bTime,self.eTime=[t0,t0]
			self.delta=-1
			self.data=np.zeros([0,3])
			return
		if delta>0 and delta!=self.delta:
			#for i in range(len(self)):
			#    self[i].interpolate(1/self.delta, method='nearest',\
			#     starttime=self.bTime,\
			#     npts=int((self.eTime-self.bTime)/self.delta))
			decMul= round(delta/self.delta)
			#print(ctime(),'start dec')
			self.decimate(decMul,no_filter=True)
			#print(ctime(),'end dec')
		if isData:
			self.data = self.Data()
	def getPSTime(self):
		pTime = self.pTime
		sTime = self.sTime
		if 'sac' in self[0].stats:
			if 't0' in self[0].stats['sac'] and pTime<0:
				pTime=self[0].stats['starttime']+self[0].stats['sac']['t0']\
				-self[0].stats['sac']['b']
			if 't1' in self[0].stats['sac'] and sTime<0:
				sTime=self[0].stats['starttime']+self[0].stats['sac']['t1']\
				-self[0].stats['sac']['b']
			if pTime>0 and sTime>0 and self.dis==180*110:
				self.dis=(sTime.timestamp-pTime.timestamp)*6/0.7
		return pTime,sTime
	def getTimeLim(self,bTime=t0,eTime=t1,delta=-1):
		bTime = UTCDateTime(bTime)
		eTime = UTCDateTime(eTime)
		if delta<0:
			delta=self.delta
		for trace in self:
			bTime = max(trace.stats['starttime'],bTime)
		n = round((eTime.timestamp-bTime.timestamp)/delta)
		eTime = bTime+n*delta
		for trace in self:
			eTime = min(trace.stats['endtime'],eTime)
		return bTime,eTime
	def decimate(self,decMul,no_filter=False):
		if decMul<16:
			super().decimate(decMul,no_filter=no_filter)#True
		else:
			if decMul%2==0:
				super().decimate(2,no_filter=no_filter)
				decMul/=2
			if decMul%2==0:
				super().decimate(2,no_filter=no_filter)
				decMul/=2
			if decMul%5==0:
				super().decimate(5,no_filter=no_filter)
				decMul/=5
			if decMul%5==0:
				super().decimate(5,no_filter=no_filter)
				decMul/=5
			if decMul>1 :
				super().decimate(5,no_filter=no_filter)
			if np.random.rand()<0.01:
				print(self)
		self.delta=self.Delta()
		self.bTime,self.eTime=self.getTimeLim()
	def Data(self,bTime=t0,eTime=t1):
		bTime,eTime = self.getTimeLim(bTime,eTime)
		if self.bTime<=t0 or self.eTime<=t0:
			return np.zeros([0,3])
		#print(bTime,eTime)
		new = self.slice(bTime,eTime,nearest_sample=True)
		n = round((eTime.timestamp-bTime.timestamp)/self.delta)+1
		if n<0:
			return np.zeros([0,len(self)],np.float32)
		data = np.zeros([n,len(self)],np.float32)
		for i in range(len(self)):
			if new[i].data.size >= n:
				pass
			else:
				print('bad Data')
			tmpN =min(n,new[i].data.size)
			data[:tmpN,i]=new[i].data[:n]
		return data
	def copy(self,*args,**kwargs):
		new=Trace3(super().copy(*args,**kwargs))
		new.compStr =self.compStr
		new.dis=self.dis
		new.pTime=self.pTime
		new.sTime=self.sTime
		return new
	def slice(self,bTime,eTime,nearest_sample=True):
		if self.bTime<=t0:
			return T0
		if bTime<self.bTime or eTime>self.eTime:
			return T0
		bTime = UTCDateTime(bTime)
		eTime = UTCDateTime(eTime)
		if bTime> eTime:
			return T0
		new=Trace3(super().slice(bTime,eTime,nearest_sample=nearest_sample))
		new.compStr =self.compStr
		new.dis=self.dis
		new.pTime=self.pTime
		new.sTime=self.sTime
		return new
	def write(self,filenames):
		for i in range(len(self)):
			if self[i].stats['starttime']<=t0 or len(self[i].data)==0:
				continue
			self[i].write(filenames[i],format='SAC')
	def Delta(self):
		return 1/self[0].stats['sampling_rate']
	def getDataByTimeLQuick(self,timeL):
		return self.Data(UTCDateTime(timeL[0]),UTCDateTime(timeL[-1]))
	def filt(self,f,filtOrder):
		if f[0]>0:
			self.filter('bandpass',freqmin=f[0], freqmax=f[1], \
						corners=filtOrder, zerophase=True)
	def resample(self,decMul):
		if decMul>0:
			self.decimate(decMul,no_filter=True)
	def adjust(self,*argv,**kwargs):
		for i in range(len(self)):
			self[i]=adjust(self[i],*argv,**kwargs)
	def rotate(self,theta=0):
		#RTZ,theta
		if theta ==0:
			return self
		rad = theta/180*np.pi
		bTime,eTime=self.getTimeLim()
		#print('data')
		Data= self.Data()
		#print('data done')
		#print('math rotate')
		Data = rotate(rad,Data)
		#print('math rotate Done' )
		T3New=[Trace(Data[:,i])for i in range(3)]
		for t3 in T3New:
			t3.stats.starttime = bTime
			t3.stats.sampling_rate = self[0].stats.sampling_rate
		return Trace3(T3New)
	def average(self):
		bTime,eTime=self.getTimeLim()
		Data = self.Data()
		t=Trace(Data.mean(axis=1))
		t.stats['_format']='average'
		t.stats.bTime = bTime
		t.stats.sampling_rate=self[0].stats.sampling_rate
		return t
	def getDetec(self,minValue=2000,minD=60,comp=2):
		l=[]
		bTime,eTime=self.getTimeLim()
		data = self.Data()
		delta= self.Delta()
		minDelta=int(minD/delta)
		indexL,vL=getDetec(data[:,comp], minValue=minValue, minDelta=minDelta)
		return indexL*delta+bTime.timestamp,vL
	def getSpec(self,comp=2,isNorm=False):
		data = self.Data()
		if isNorm:
			data/=data.std()
		return np.fft.fft(data[:,comp]),np.arange(len(data[:,comp]))/(self.Delta()*len(data[:,comp]))
	def __del__(self):
		try:
			del(self.data)
			del(self.Data0)
			del(self.Data1)
		except:
			pass
		else:
			pass
	def get(self,key):
		for t in self:
			if key in t.stats['sac']:
				return t.stats['sac'][key]
		return None

def checkSacFile(sacFileNamesL):
	if len(sacFileNamesL)==0:
		return False
	for sacFileNames in sacFileNamesL:
		count = 0
		for sacFileName in sacFileNames:
			if os.path.exists(sacFileName):
				count=count+1
		if count==0:
			return False
	return True
T0=Trace3([])
def getTrace3ByFileName(sacFileNamesL, delta0=0.02, freq=[-1, -1], \
	filterName='bandpass', corners=2, zerophase=True,maxA=1e18,\
	bTime=UTCDateTime(1900,1,1),eTime=UTCDateTime(2099,1,1),isPrint=False,mode='norm',
	pTime=-1,sTime=-1,isData=True):
	if not checkSacFile(sacFileNamesL):
		if isPrint:
			print('not find')
		return Trace3([])
	sacs = []
	#time0=time.time()
	#print(ctime(),'start merge')
	for sacFileNames in sacFileNamesL:
		tmp=mergeSacByName(sacFileNames, delta0=delta0,freq=freq,\
			filterName=filterName,corners=corners,zerophase=zerophase,maxA=maxA)
		sacs.append(tmp)
	#print(ctime(),'end merge')
	#print(ctime(),'start T3')
	if None not in sacs:
		return Trace3(sacs,delta=delta0,bTime=bTime,eTime=eTime,isData=isData,pTime=-1,sTime=-1)
		#print(ctime(),'end T3')
	else:
		#print(ctime(),'not good')
		return Trace3([])
   
	#print('read',time1-time0,'dec',time2-time1)
	#return Data(dataL[0], dataL[1], dataL[2], dataL[3], freq,dataL[4],dataL[5],dataL[6])
def saveSacs(staL, quakeL, staInfos,matDir='output/',\
	bSec=-10,eSec=40,dtype=np.float32):
	if not os.path.exists(matDir):
		os.mkdir(matDir)
	for i in range(len(quakeL)):
		 quakeL[i]['ml']=quakeL[i].saveSacs(staL, staInfos,\
		  matDir=matDir,bSec=bSec,eSec=eSec,dtype=dtype)
#用一个文件存相关信息，然后根据文件载入,实验一下，小台阵
#明天看有无问题
#降采和对齐的先后顺序可能有影响

def getQuakeNoise(qL,staInfos,workDir,resFile='noise_lst'):
	
	with open(resFile,'w') as f:
		for q in qL:
			name = q['filename']
			T3L = q.loadSacs(staInfos,workDir)
			for i in range(len(q.records)):
				record = q.records[i]
				T3     = T3L[i]
				if T3.bTime<=0 or T3.pTime<=0:
					continue
				Data = T3.Data(T3.bTime+5,T3.pTime-3)
				if len(Data)<=0:
					continue
				nStd =Data.std(axis=0)
				f.write('%d %.2f %.5f %.5f %.5f\n'%(record['staIndex'],T3.pTime,\
					nStd[0],nStd[1],nStd[2]))

def loadNoiseRes(resFile='noise_lst'):
	data = np.loadtxt(resFile)
	staIndexL=data[:,0].astype(np.int32)
	timeL =data[:,1]
	noiseL=data[:,2:]
	return staIndexL,timeL,noiseL

def plotStaNoiseDay(staIndexL,timeL,noiseL,resDir):
	if not os.path.exists(resDir):
		os.makedirs(resDir)
	timeL = timeL-UTCDateTime(2014,1,1).timestamp
	for staIndex in np.unique(staIndexL):
		plt.close()
		color='rbk'
		for i in range(3):
			plt.subplot(4,1,i+1)
			plt.subplot
			#plt.plot(((timeL[staIndexL==staIndex]+8*3600)%86400)/3600,\
			#    noiseL[staIndexL==staIndex][:,i],'.'+color[i],markersize=1)
			plt.hist2d(((timeL[staIndexL==staIndex]+8*3600)%86400)/3600,\
				np.log(np.abs(noiseL[staIndexL==staIndex][:,i])+1),bins=[8,15],cmap='cool')
		#plt.ylim([0,noiseL[staIndexL==staIndex].std()*5])
		plt.subplot(4,1,4)
		plt.hist(((timeL[staIndexL==staIndex]+8*3600)%86400)/3600,bins=8)
		plt.xlim([0,24])
		plt.savefig(resDir+'/%d_noise.jpg'%staIndex,dpi=300)
		plt.close()

def corrTrace(Trace0,Trace1):
	data = np.correlate(Trace0.data,Trace1.data,'full')
	bTime = Trace0.stats.starttime.timestamp-Trace1.stats.starttime.timestamp
	Trace01 = obspy.Trace(data)
	Trace01.stats.starttime=bTime
	Trace01.stats.sampling_rate=Trace0.stats.sampling_rate
	Trace01.stats['_format']='cross'
	Trace01=adjust(Trace01,kzTime=UTCDateTime(0))
	return Trace01

RTV='RTZ'
class StationPair:
	def __init__(self,sta0,sta1):
		self.pair=[sta0,sta1]
		self.az=sta0.az(sta1)
		self.dist=sta0.dist(sta1)
	def resDir(self,parentDir=''):
		return parentDir+'/'+self.pair[0].name('.')+'/'+self.pair[1].name('.')+'/'
	def loadTraces(self,parentDir=''):
		tracesL=[]
		resDir =self.resDir(parentDir=parentDir)
		for comp in comp33:
			tracesL.append([]) 
			for file in glob(resDir+'*'+comp):
				tracesL[-1].append(read(file)[0])
			tracesL[-1] = Trace3(tracesL[-1])
		return tracesL
	def getNum(self,parentDir=''):
		resDir =self.resDir(parentDir=parentDir)
		for comp in comp33:
			return(len(glob(resDir+'*.'+comp)))
	def average(self,tracesL):
		averageTraces=[]
		for traces in tracesL:
			averageTraces.append(traces.average())
		return averageTraces
	def getAverage(self,parentDir=''):
		resDir =self.resDir(parentDir=parentDir)
		tracesL = self.loadTraces(parentDir)
		averageTraces=self.average(tracesL)
		for i in range(9):
			adjust(averageTraces[i],kzTime=UTCDateTime(0))
			averageTraces[i].write(resDir+comp33[i]+'.average',format='SAC')
			print(resDir+comp33[i]+'.average')
	def loadAverage(self,parentDir=''):
		resDir =self.resDir(parentDir=parentDir)
		for comp in comp33:
			if not os.path.exists(resDir+comp+'.average'):
				return Trace3([])
		return Trace3([read(resDir+comp+'.average')[0] for comp in comp33])
def StationPairM(staInfos):
	return [[StationPair(sta0,sta1) for sta1 in staInfos]for sta0 in staInfos]

def plotStationPairL(StationPairL,resDir='./',parentDir='',mul=3):
	if not os.path.exists(resDir):
		os.makedirs(resDir)
	plt.close()
	count=0
	for StationPair in StationPairL:
		T3 = StationPair.loadAverage(parentDir)
		if len(T3)>0:
			data=T3.Data()
			data/=data.max(axis=0)
			timeL=np.arange(data.shape[0])*T3.delta+T3.getTimeLim()[0].timestamp
			dist = StationPair.dist
			plt.plot(timeL,data[:,0]*mul+dist,'k',linewidth=0.3)
			count+=1
	if count<=5:
		plt.close()
		return
	plt.xlabel('t/s')
	plt.ylabel('distance/km')
	plt.title(StationPair.pair[0])
	plt.savefig(resDir+StationPair.pair[0].name('_')+'.jpg',dpi=300)
	plt.close()



def getT3LLQuick(qL,staInfos,matDir,f=[-1,-1],batch_size=100,num_workers=6,**kwargs):
	T3PSLL=[]
	reader = parRead.Reader(qL,staInfos,matDir=matDir,f=f,**kwargs)
	parReader = parRead.DataLoader(reader,collate_fn=parRead.collate_function,batch_size=batch_size,num_workers=num_workers)
	for tmp in parReader:
		for t in tmp:
			T3PSLL.append(t)
	return T3PSLL
