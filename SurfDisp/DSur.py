from SeismTool.plotTool import figureSet
import numpy as np
import os 
from scipy import interpolate,stats
from matplotlib import pyplot as plt
import pycpt
from netCDF4 import Dataset
from ..mapTool import mapTool as mt
from ..mapTool.mapTool import getDlaDlo
from ..mathTool.distaz import DistAz
from ..mathTool.mathFunc_bak import Model as Model0,outR
#import cmath

#cmap = pycpt.load.gmtColormap('cpt/temperatureInv')
cmap = pycpt.load.gmtColormap(os.path.dirname(__file__)+'/../data/temperatureInv')
cmap = pycpt.load.gmtColormap(os.path.dirname(__file__)+'/../data/seis.cpt')
cmap = 'jet_r'
diffMap = pycpt.load.gmtColormap(os.path.dirname(__file__)+'/../data/seismicNew.cpt')
##程序中又控制的大bug
##必须按顺序(period source)
faultL = mt.readFault(os.path.dirname(__file__)+'/../data/Chinafault_fromcjw.dat')
#faultL = mt.readFault('Chinafault_fromcjw.dat')
class config:
    def __init__(self,para={},name='ds',z=[10,20,40,80,120,160,200,320]):
        self.name = name
        self.z    = z
        config.keyList = ['dataFile', 'nxyz', 'lalo', 'dlalo', 'maxN','damp',\
        'sablayers','minmaxV', 'maxIT','sparsity','kmaxRc','rcPerid','kmaxRg','rgPeriod',\
        'kmaxLc','lcPeriod','kmaxLg','lgPeriod','isSyn','noiselevel','threshold',\
        'vnn']
        config.keyList = ['dataFile', 'nxyz', 'lalo', 'dlalo', 'sablayers','minmaxV',\
        'maxN','sparsity', 'maxIT','iso','c','smoothDV','smoothG','Damp',\
        'c','kmaxRc','rcPerid','nBatch']
        config.keyListSyn = ['dataFile', 'nxyz', 'lalo', 'dlalo','maxN', 'sablayers',\
        'sparsity','rayT','kmaxRc','rcPerid','noise']
        self.para = {'dataFile':name+'in', 'nxyz':[18,18,9], 'lalo':[130,30],\
         'dlalo':[0.01,0.01], 'maxN':[20],'damp':[4.0,1.0],\
        'sablayers':3,'minmaxV':[1,7],'maxIT':10, 'sparsity':0.4,\
        'kmaxRc':10,'rcPerid':np.arange(1,11).tolist(),'kmaxRg':0,'rgPeriod':[],\
        'kmaxLc':0,'lcPeriod':[],'kmaxLg':0,'lgPeriod':[],'isSyn':0,'noiselevel':0.02,'threshold':0.05,\
        'vnn':[0,100,50],'iso':'F','c':'c','smoothDV':10,'smoothG':20,'Damp':0,'perN':[6,6,4],'perA':0.05,\
        'modelPara': {'config':self,'mode':'prem','runPath':'','file':'../models/prem','la':'','lo':'','z':'','self1':'',\
        },'rayT':'F','noise':0,\
        'GSPara': {'config':self,'mode':'GS','runPath':'','file':'','la':'','lo':'','z':'','self1':'',\
        },\
        'GCPara': {'config':self,'mode':'GC','runPath':'','file':'','la':'','lo':'','z':'','self1':'',\
        },'vR':np.array([[-90,-180],[-90,180],[90,180],[90,-180],[-90,-180]])}
        self.para.update(para)
    def output(self):
        nxyz = self.para['nxyz']
        la = self.para['lalo'][0]-np.arange(nxyz[0])*self.para['dlalo'][0]
        lo = self.para['lalo'][1]+np.arange(nxyz[1])*self.para['dlalo'][1]
        return nxyz,la,lo,self.z
    def outputP(self):
        nxyz = self.para['nxyz'][:2]+[len(self.para['rcPerid'])]
        la = self.para['lalo'][0]-np.arange(nxyz[0])*self.para['dlalo'][0]
        lo = self.para['lalo'][1]+np.arange(nxyz[1])*self.para['dlalo'][1]
        return nxyz,la,lo,np.array(self.para['rcPerid'])
    def findLaLo(self,la,lo):
        laNew = -round((self.para['lalo'][0]-la)/self.para['dlalo'][0])*self.para['dlalo'][0]+self.para['lalo'][0]+0.001
        loNew =  round((lo-self.para['lalo'][1])/self.para['dlalo'][1])*self.para['dlalo'][1]+self.para['lalo'][1]+0.001
        return la,lo
class DS:
    """docstring for ClassName"""
    def __init__(self,runPath='DS/',config=config(),mode='real'):
        self.runPath = runPath
        if not os.path.exists(runPath):
            os.makedirs(runPath)
        self.config = config
        self.config.para['modelPara']['runPath']= runPath
        self.config.para['GSPara']['runPath']= runPath
        self.config.para['GCPara']['runPath']= runPath
        self.mode = mode
        self.fastP=''
        self.fast=''

    def writeInput(self,perI=-1):
        '''
        cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        c SurfAnisoForward Input
        cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        Surfphase_RV3_5_40s_1s.dat          c: blank traveltime data file
        17 17 4                             c: nx ny nz (grid number in lat lon and depth directions)
        26.5  101.25                        c: goxd gozd (upper left point,[lat,lon])
        0.25 0.25                           c: dvxd dvzd (grid interval in lat and lon directions)
        5000                                c: max(sources, receivers)
        2                                   c: sublayers (2~5)
        0.2                                 c: sparsity fraction
        F                                   c: T: output raypath; F: not output raypath
        36                                  c: kmaxRc (followed by periods)
        5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
        0                                   c: noise level e.g.: 0.5s

        cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        c INPUT PARAMETERS
        cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        surfdataTB.dat                   c: data file
        18 18 9                          c: nx ny nz (grid number in lat lon and depth direction)
        25.2  121.35                     c: goxd gozd (upper left point,[lat,lon])
        0.015 0.017                      c: dvxd dvzd (grid interval in lat and lon direction)
        20                               c: max(sources, receivers)
        4.0  1.0                         c: weight damp
        3                                c: sablayers (for computing depth kernel, 2~5)
        0.5 2.8                          c: minimum velocity, maximum velocity (a priori information)
        10                               c: maximum iteration
        0.2                              c: sparsity fraction
        26                               c: kmaxRc (followed by periods)
        0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0
        0                                c: kmaxRg
        0                                c: kmaxLc
        0                                c: kmaxLg
        0                                c: synthetic flag(0:real data,1:synthetic)
        0.02                             c: noiselevel
        0.05                             c: threshold
        0 100 50                         c: vorotomo,ncells,nrelizations
        '''
        with open(self.runPath+self.config.name, 'w+') as f:
            f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
            f.write('c INPUT PARAMETERS\n')
            f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
            for key in self.config.keyList:
                value = self.config.para[key]
                if isinstance(value,list):
                    valueNew = ''
                    for v in value:
                        valueNew += str(v)+' '
                    value = valueNew
                value = str(value)
                if value =='':
                    continue
                f.write(value+' c: '+key+'\n')
        with open(self.runPath+self.config.name+'syn', 'w+') as f:
            f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
            f.write('c INPUT PARAMETERS\n')
            f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
            for key in self.config.keyListSyn:
                value = self.config.para[key]
                if isinstance(value,list):
                    valueNew = ''
                    for v in value:
                        valueNew += str(v)+' '
                    value = valueNew
                value = str(value)
                if value =='':
                    continue
                if perI >=0:
                    if key =='kmaxRc':
                        value = '1'
                    if key =='rcPerid':
                        value = str(self.config.para[key][perI])
                f.write(value+' c: '+key+'\n')
    def writeData(self,fvLL,indexL,stations,waveType=0,perI=-1,perJ=0,M=1):
        staN = len(stations)
        distM = np.zeros([staN, staN])
        for i in range(staN):
            for j in range(i+1):
                dist = DistAz(0,0,0,0).degreesToKilometers(\
                    DistAz(stations[i]['la'],stations[i]['lo'],\
                        stations[j]['la'],stations[j]['lo']).getDelta())
                distM[i,j] = dist
                distM[j,i] = dist
        with open(self.runPath+'/'+self.config.para['dataFile'],'w') as f:
            for j in range(self.config.para['kmaxRc']):
                if perI >=0:
                    if j != perI:
                        continue
                for i in range(staN):
                    if i%M!=perJ:
                        continue
                    if len(indexL[i])==0:
                        continue
                    vL =np.zeros(len(indexL[i]))
                    for k in range(len(indexL[i])):
                        vL[k]=fvLL[i][k][j]
                    nvL = (vL>2).sum()
                    if nvL<2:
                        continue
                    la,lo=self.config.findLaLo(stations[i]['la'],\
                        stations[i]['lo'])
                    if perI>=0:
                        f.write('# %.3f %.3f %d 2 0\n'%(la,lo,j+1-perI))
                    else:
                        f.write('# %.3f %.3f %d 2 0\n'%(la,lo,j+1))
                    for k in range(len(indexL[i])):
                        kk = indexL[i][k]
                        #print(kk)
                        if vL[k]>2 and vL[k]<6:
                            la,lo=self.config.findLaLo(stations[kk]['la'],\
                            stations[kk]['lo'])
                            f.write('%.3f %.3f %f\n'%(la,lo,vL[k]))
    def writeMod(self):
        nx,ny,nz=self.config.para['nxyz']
        dep1=np.array(self.config.z)
        nxyz,la,lo,z = self.config.output()
        #dep1=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.8,2.1,2.5])
        nz=len(dep1)
        #ends
        vs1=np.zeros(nz)
        self.model = Model(**self.config.para['modelPara'])
        self.model.write(self.runPath+'/MOD',la,lo,z)
        if self.mode == 'syn':
            self.GC    = Model(**self.config.para['GCPara'])
            self.GS    = Model(**self.config.para['GSPara'])
            self.model.write(self.runPath+'/MODVs.true',la,lo,z,isDiff=True,N=self.config.para['perN']\
                ,A=self.config.para['perA'])
            self.GS.write(self.runPath+'/MODGs.true',la,lo,z,isDiff=True,N=self.config.para['perNG']\
                ,A=self.config.para['perAGs']*0,base=0)
            self.GC.write(self.runPath+'/MODGc.true',la,lo,z,isDiff=True,N=self.config.para['perNG']\
                ,A=self.config.para['perAGc'],base=0)
        for i in range(nz):
          print (dep1[i]),
    def test(self,fvLL,indexL,stations,isRun=False):
        self.writeInput()
        self.writeMod()
        self.writeData(fvLL,indexL,stations)
        if isRun:
            self.run()
    def run(self):
        os.system('cd %s;DAzimSurfTomo_batch_20220111 ds &'%(self.runPath))
    def testSyn(self,fvLL,indexL,stations,M=1,isRun=False):
        self.writeMod()
        periods = ['' for i in range(self.config.para['kmaxRc'])]
        for i in range(self.config.para['kmaxRc']):
            self.writeInput(perI=i)
            for j in range(M):
                self.writeData(fvLL,indexL,stations,perI=i,perJ=j,M=M)
                os.system('cd %s;SurfAAForward dssyn'%self.runPath)
                with open('%s/surfphase_forward.dat'%self.runPath,'r') as res:
                    for line in res.readlines():
                        if line[0] == '#':
                            # 39.914 116.241 11 2 0
                            tmp = line.split()
                            tmp[-3] = '%d'%(i+1)
                            line='#'
                            for t in tmp[1:]:
                                line += ' '+t
                            line += '\n'
                        else:
                            tmp = line.split()
                            v   = float(tmp[-1])
                            #'  32.051991 118.853989  3.00739'
                            v   *= 1+2*(np.random.rand()-0.5)*self.config.para['noiselevel']
                            tmp[-1] = '%.6f'%v
                            line=''
                            for t in tmp:
                                line += ' '+t
                            line += '\n'
                        periods[i]+=line
                        if np.random.rand()<0.001:
                            print(line)
        with open('%s/dsin'%self.runPath,'w+') as f:
            for period in periods:
                f.write(period)
        if isRun:
            self.run()
    def plotByZ(self,p2L=[],R=[],self1='',isCompare=False,**kwargs):
        if self.mode == 'syn':
            if self.config.para['iso']=='F':
                self.modelTrue.plotByZ(self.runPath,self1=self.fastTrue,head='true',R=R,isVol=False,maxDeep=9000,isFault=False,**kwargs)
                self.modelRes.plotByZ(self.runPath,self1=self.fast,R=R,head='depth',isVol=False,maxDeep=9000,isFault=False,**kwargs)
                #self.fast.plotArrByZ(self.runPath,head='fast')
                self.modelPeriod.plotByZ(self.runPath,self1=self.fastP,head='period',R=R,isVol=False,isFault=False,**kwargs)
            else:
                self.modelTrue.plotByZ(self.runPath,head='true',R=R,isVol=False,maxDeep=9000,isFault=False,**kwargs)
                self.modelRes.plotByZ(self.runPath,R=R,head='depth',isVol=False,maxDeep=9000,isFault=False,**kwargs)
                #self.fast.plotArrByZ(self.runPath,head='fast')
                self.modelPeriod.plotByZ(self.runPath,head='period',R=R,isVol=False,isFault=False,**kwargs)
            return
        '''
        for p2 in p2L:
            self.modelRes.plotByP2(self.runPath,head='P2',P2=p2,vR=self.config.para['vR'])
        '''
        if self1!='' and isCompare:
            self.modelRes.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fast,R=R,head='depth',selfRef=self.modelRes,**kwargs)
            self.modelPeriod.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fastP,head='period',R=R,selfRef=self.modelPeriod,**kwargs)
            self1.modelRes.plotByZ(self1.runPath,vR=self.config.para['vR'],self1=self1.fast,R=R,head='depth',selfRef=self.modelRes,**kwargs)
            self1.modelPeriod.plotByZ(self1.runPath,vR=self.config.para['vR'],self1=self1.fastP,head='period',R=R,selfRef=self.modelPeriod)
            if self.config.para['iso']=='F':
                self.fastDiff = self.fast.copy()
                v0 = self.fastDiff.v - self1.fast.v
                v1 = self.fastDiff.v + self1.fast.v
                v0[np.abs(v1)<np.abs(v0)]=v1[np.abs(v1)<np.abs(v0)]
                self.fastDiff.v=v0
                self.fastPDiff = self.fastP.copy()
                v0 = self.fastPDiff.v - self1.fastP.v
                v1 = self.fastPDiff.v + self1.fastP.v
                v0[np.abs(v1)<np.abs(v0)]=v1[np.abs(v1)<np.abs(v0)]
                self.fastPDiff.v = v0
            self.modelResDiff = self.modelRes.copy()
            self.modelResDiff.v = self.modelResDiff.v/self1.modelRes.v-1
            self.modelPeriodDiff = self.modelPeriod.copy()
            self.modelPeriodDiff.v = self.modelPeriodDiff.v/self1.modelPeriod.v-1
            self.modelResDiff.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fastDiff,R=R,head='depth',isDiff=True,**kwargs)
            self.modelPeriodDiff.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fastPDiff,head='period',R=R,isDiff=True,**kwargs)
        elif self1!='' :
            self.modelRes.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fast,R=R,head='depth',selfRef=self1.modelRes,**kwargs)
            #self.fast.plotArrByZ(self.runPath,head='fast')
            self.modelPeriod.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fastP,head='period',R=R,selfRef=self1.modelPeriod,**kwargs)
        else:
            self.modelRes.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fast,R=R,head='depth',**kwargs)
            #self.fast.plotArrByZ(self.runPath,head='fast')
            self.modelPeriod.plotByZ(self.runPath,vR=self.config.para['vR'],self1=self.fastP,head='period',R=R,**kwargs)
    def plotTK(self):
        nxyz,la,lo,z = self.config.output()
        la = la[::-1]
        z0,la0,lo0,vsv0= loadModelTK()
        resDir = self.runPath+'/'+'plot/'
        if not os.path.exists(resDir):
            os.mkdir(resDir)
        for i in range(nxyz[-1]):
            index = np.abs(z[i]-z0).argmin()
            v = interpolate.interp2d(lo0,la0,vsv0[index],kind='cubic')(lo,la) 
            plt.close()
            plt.pcolormesh(lo,la,-v,cmap=cmap,rasterized=True)
            plt.colorbar()
            plt.title('%f.jpg'%z[i])
            plt.savefig('%s/TK_%f.jpg'%(resDir,z[i]),dpi=200)
            plt.ylim([35,55])
            plt.close()
    def plotHJ_(self):
        nxyz,la,lo,z = self.config.output()
        la = la[::-1]
        z0,la0,lo0,vsv0= loadModelHJ('s')
        resDir = self.runPath+'/'+'plot/'
        if not os.path.exists(resDir):
            os.mkdir(resDir)
        for i in range(nxyz[-1]):
            index = np.abs(z[i]-z0).argmin()
            v = interpolate.interp2d(lo0,la0,vsv0[index],kind='linear')(lo,la) 
            print(la)
            plt.close()
            plt.pcolormesh(lo,la,-v,cmap=cmap,rasterized=True)
            plt.colorbar()
            plt.title('%f.jpg'%z[i])
            plt.savefig('%s/HJ_%f.jpg'%(resDir,z[i]),dpi=200)
            plt.ylim([35,55])
            plt.close()
    def loadRes(self):
        vR = ''#self.config.para['vR']
        if self.mode == 'syn':
            self.modelTrue = Model(self.config,mode='DSFile',runPath=self.runPath,file='MODVs.true',vR =vR)
            if self.config.para['iso']=='F':
                self.fastTrue      = Model(self.config,mode='fast',runPath=self.runPath,file='Gc_Gs_model.real',vR =vR)
        self.model0 = Model(self.config,mode='DSFile',runPath=self.runPath,file='MOD',vR =vR)
        self.modelPeriod = Model(self.config,mode='DSP',runPath=self.runPath,file='period_Azm_tomo.inv',vR =vR)
        #self.GsTrue = Model(self.config,mode='GSO',runPath=self.runPath,file='MODGs.true')
        #self.GcTrue = Model(self.config,mode='GCO',runPath=self.runPath,file='MODGc.true')
        #self.modelInit = Model(self.config,mode='DSFile',runPath=self.runPath,file='MOD')
        self.modelRes = Model(self.config,mode='DS',runPath=self.runPath,file='Gc_Gs_model.inv',vR =vR)
        if self.config.para['iso']=='F':
            self.GsRes = Model(self.config,mode='GS',runPath=self.runPath,file='Gc_Gs_model.inv',vR =vR)
            self.GcRes = Model(self.config,mode='GC',runPath=self.runPath,file='Gc_Gs_model.inv',vR =vR)
            self.fast = Model(self.config,mode='fast',runPath=self.runPath,file='Gc_Gs_model.inv',vR =vR)
            self.fastP = Model(self.config,mode='fastP',runPath=self.runPath,file='period_Azm_tomo.inv',vR =vR)
    def plotHJ(self,p2L=[],R=[],isCompare=False):
        vR = ''
        self.HJ = Model(self.config,mode='HJ',runPath=self.runPath,vR =vR,self1=self.modelRes)
        self.HJ.plotByZ(self.runPath,vR=self.config.para['vR'],R=R,head='depth(USTC)')#,selfRef=self.modelRes
        if isCompare:
           self.modelResHJDiff = self.HJ.copy()
           self.modelResHJDiff.v = self.modelResHJDiff.v/self.modelRes.v-1
           self.modelResHJDiff.plotByZ(self.runPath,vR=self.config.para['vR'],R=R,head='depth(USTC)',isDiff=True)
    def plotWS(self,p2L=[],R=[],isCompare=False):
        vR = ''
        self.WS = Model(self.config,mode='WS',runPath=self.runPath,vR =vR,self1=self.modelRes)
        self.WS.plotByZ(self.runPath,vR=self.config.para['vR'],R=R,head='depth(WS)')#,selfRef=self.modelRes
        if isCompare:
           self.modelResWSDiff = self.WS.copy()
           self.modelResWSDiff.v = self.modelResWSDiff.v/self.modelRes.v-1
           self.modelResWSDiff.plotByZ(self.runPath,vR=self.config.para['vR'],R=R,head='depth(WS)',isDiff=True)
    def outputRef(self):
        self.modelRes.outputRef(self.config.para['modelPara']['file'])

class Model(Model0):
    def __init__(self,config=None,mode='DS',runPath='',file='',la='',lo='',z='',self1='',Gs='',Gc='',vR=''):
        self.mode = mode
        self.config=config
        if mode =='DS':
            data = np.loadtxt(runPath+file)
            nxyz,la,lo,z = config.output()
            z    = np.array(z)
            v = np.zeros(nxyz)*0+np.nan
            for i in range(data.shape[0]):
                Lo = data[i,0]
                La = data[i,1]
                Z  = data[i,2]
                V  = data[i,3]
                i0 = np.abs(la-La).argmin()
                i1 = np.abs(lo-Lo).argmin()
                i2 = np.abs(z-Z).argmin()
                v[i0,i1,i2]=V
        if mode =='DSP':
            data = np.loadtxt(runPath+file)
            nxyz,la,lo,z = config.outputP()
            z    = np.array(z)
            v = np.zeros(nxyz)*0+np.nan
            for i in range(data.shape[0]):
                Lo = data[i,0]
                La = data[i,1]
                Z  = data[i,2]
                V  = data[i,3]
                i0 = np.abs(la-La).argmin()
                i1 = np.abs(lo-Lo).argmin()
                i2 = np.abs(z-Z).argmin()
                v[i0,i1,i2]=V
        if mode == 'DSFile':
            nxyz,la,lo,z = config.output()
            v = np.loadtxt(runPath+file,skiprows=1)
            v=v.reshape(nxyz[2],nxyz[1],nxyz[0]).transpose([2,1,0])
        if mode =='TK':
            z,la,lo,v=loadModelTK()
            #.reshape([-1])
            #shape = self.vsv
            #self.vsv=self.vsv.reshape([-1])
            v=v.transpose([1,2,0])
        if mode =='HJ':
            z,la,lo,v=loadModelHJ()
            #.reshape([-1])
            #shape = self.vsv
            #self.vsv=self.vsv.reshape([-1])
        if mode =='WS':
            z,la,lo,v=loadModelWS()
        if mode=='byModel':
            v = self1.output(la,lo,z)
        if mode=='prem':
            data = np.loadtxt(file)
            la = np.array([-90,90])
            lo = np.array([-180,180])
            z = data[:,0]
            V = data[:,2]
            v = np.zeros([2,2,len(V)])
            for i in range(2):
                for j in range(2):
                    v[i,j] = V
        if mode=='GC' or mode=='GS' or mode=='fast':
            nxyz,la,lo,z = config.output()
            v = np.zeros(nxyz)*0
            if mode =='fast':
                v = v+1j*v
            if file!='':
                data = np.loadtxt(runPath+file)
                for i in range(data.shape[0]):
                    Lo = data[i,0]
                    La = data[i,1]
                    Z  = data[i,2]
                    Phi= data[i,4] 
                    A  = data[i,5] 
                    gc = data[i,6] 
                    gs = data[i,7] 
                    i0 = np.abs(la-La).argmin()
                    i1 = np.abs(lo-Lo).argmin()
                    i2 = np.abs(z-Z).argmin()
                    if mode == 'GC':
                        v[i0,i1,i2]=gc/100
                    elif mode=='GS':
                        v[i0,i1,i2]=gs/100
                    elif mode=='fast':
                        phi = Phi/180*np.pi
                        v[i0,i1,i2]=A*np.cos(phi)+1j*A*np.sin(phi)
        if mode=='fastP':
            nxyz,la,lo,z = config.outputP()
            v = np.zeros(nxyz)*0
            if mode =='fastP':
                v = v+1j*v
            if file!='':
                data = np.loadtxt(runPath+file)
                for i in range(data.shape[0]):
                    Lo = data[i,0]
                    La = data[i,1]
                    Z  = data[i,2]
                    Phi= data[i,4] 
                    A  = data[i,5] 
                    gc = data[i,6] 
                    gs = data[i,7] 
                    i0 = np.abs(la-La).argmin()
                    i1 = np.abs(lo-Lo).argmin()
                    i2 = np.abs(z-Z).argmin()
                    if mode == 'GC':
                        v[i0,i1,i2]=gc/100
                    elif mode=='GS':
                        v[i0,i1,i2]=gs/100
                    elif mode=='fastP':
                        phi = Phi/180*np.pi
                        v[i0,i1,i2]=A*np.cos(phi)+1j*A*np.sin(phi)
        if mode=='GCO' or mode=='GSO':
            nxyz,la,lo,z = config.output()
            v = np.zeros(nxyz)*0+0.01
            if file!='':
                v = np.loadtxt(runPath+file)
                v=v.reshape(nxyz[2],nxyz[1],nxyz[0]).transpose([2,1,0])
        if mode=='fast_':
            self.config=Gs.config
            z      = Gs.z
            la     = Gs.la
            lo     = Gs.lo
            A,phi=[Gs.v,Gc.v]
            '''
            for i in range(Gs.v.shape[0]):
                for j in range(Gs.v.shape[1]):
                    for k in range(Gs.v.shape[2]):
                        A[i,j,k],phi[i,j,k]  = cmath.polar(Gc.v[i,j,k]+1j*Gs.v[i,j,k])
            '''
            phi = 1/2*np.arctan(Gs.v/Gc.v)
            A = 1/2*(Gs.v**2+Gc.v**2)**0.5
            v      = A*np.cos(phi)+1j*A*np.sin(phi)
        self.nxyz = [len(la),len(lo),len(z)]
        self.z  =  z#.reshape([-1,1,1])
        self.la = la#.reshape([1,-1,1])
        self.lo = lo#.reshape([1,1,-1])
        self.v  = v
        if len(vR)!=0:
            out  = outR(vR,self.la,self.lo)
            #print(out)
            self.v[out]=np.nan
        if self1!='' and mode!='byModel':
            nxyz0,la0,lo0,z0 = self1.config.output()
            self.v = self.output(la0,lo0,z0)
            for i in range(len(z0)):
                VTmp = self.v[:,:,i]
                VTmp[VTmp<1]=VTmp[VTmp>2].mean()
                self.v[:,:,i]=VTmp
            self.nxyz = [len(la0),len(lo0),len(z0)]
            self.z  =  z0#.reshape([-1,1,1])
            self.la = la0#.reshape([1,-1,1])
            self.lo = lo0#.reshape([1,1,-1])
    def copy(self):
        self1 = Model(self.config,'byModel','','',self.la,self.lo,self.z,self,Gs='',Gc='',vR='')
        self1.mode = self.mode
        return self1
    def __call__(self,la,lo,z):
        i0 = np.abs(self.la - la).argmin()
        i1 = np.abs(self.lo - lo).argmin()
        i2 = np.abs(self.z  - z).argmin()
        v = self.v[i0,i1,i2]
        return v 
    
    def plotByZ(self,runPath='DS',head='res',self1='',vR='',maxA=0.02,R=[],isDiff=False,selfRef='',isVol=False,maxDeep=20,isFault=True,FORMAT='eps'):
        R0=R
        resDir = runPath+'/'+'plot/'
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        nxyz,la,lo,z = self.config.output()
        if self.mode=='DSP':
            nxyz,la,lo,z = self.config.outputP()
        #if self.mode=='HJ':
        #    nxyz,la,lo,z = self.nxyz,self.la,self.lo,self.z
        Lo,La,Z    =  np.meshgrid(lo,la,z)
        laO,loO = [la,lo]
        #V            =  self.output(la,lo,z)
        V = self.v
        if vR !='':
            out  = outR(vR,la,lo)
        #out=out.reshape([out.shape[1],out.shape[0]]).transpose()
        if self1!='':
            nxyz1,la1,lo1,z1       =  self1.config.output()
            if self1.mode=='fastP':
                nxyz1,la1,lo1,z1       =  self1.config.outputP()
                print(z1)
            V1  = self1.v
            if vR!='':
                out1  = outR(vR,la1,lo1)
        if selfRef!='':
            nxyzRef,laRef,loRef,zRef       =  selfRef.config.output()
            VRef  = selfRef.v
            if vR!='':
                outRef  = outR(vR,laRef,loRef)
        for i in range(self.nxyz[-1]):
            plt.close()
            if len(R0)==0:
                fig=plt.figure(figsize=[7.2,2.5*8/6])
            else:
                fig=plt.figure(figsize=[6,4])
            v = V[:,:,i]
            if vR!='':
                #pass
                v[out]=np.nan
            if selfRef!='':
                vRef = VRef[:,:,i]
                vRef[outRef]=np.nan
                if len(vRef[np.isnan(vRef)==False])==0:
                    continue
                mean = vRef[np.isnan(vRef)==False].mean()
                print('###by ref mean###',mean)
            else:
                mean = v[np.isnan(v)==False].mean()
            dLaO = laO.max()-la.min()
            dLoO = loO.max()-lo.min()
            if len(R)==0:
                R = [laO.min()-dLaO*0.1,la.max()+dLaO*0.1,lo.min()-dLoO*0.1,lo.max()+dLoO*0.1]
            la,lo,per=self.denseLaLoGrid(v.copy(),doDense=True,N=100,dIndex=1)
            #print(per)
            if isinstance(per,type(None)):
                plt.close()
                print('no enough data in depth:',z)
                continue
            if (np.isnan(per)==False).sum()==0:
                plt.close()
                print('no enough data in depth:',z)
                continue
            m = mt.genBaseMap(R)
            x,y= m(lo,la)
            #X,Y=m(vR[:,1],vR[:,0])
            #m.plot(X,Y,'r')
            #if vR !='':
            #    OUT  = outR(vR,la,lo)
            #    per[OUT]=np.nan
            
            if isDiff:
                vmin=-0.05
                vmax=+0.05
                mean = np.abs(v[np.isnan(v)==False]).mean()
            else:
                if maxDeep>10000:
                    per/=mean
                    per-=1
                    if len(per[np.isnan(per)==False])>0:
                        perA=np.abs(per[np.isnan(per)==False]).max()
                    else:
                        perA=0.05
                    print(perA)
                    vmin=-perA
                    vmax=+perA
                else:
                    per/=mean
                    per-=1
                    perA=0.05
                    if z[i]> maxDeep:
                        perA=0.03
                    vmin=-perA
                    vmax=+perA
            ac =plt.gca()
            if self1 !='':
                x0,y0= m(R[2],R[1])
                x1,y1= m(R[3],R[0])
                dx1 = (x1-x0)/maxA/nxyz1[1]
            if self1 !='':
                x1,y1= m(lo1,la1)
            
            if isDiff:
                plotPlane(m,x,y,per*100,R,z[i],mean*100,vmin*100,vmax*100,isFault=isFault,head=head,isVol=False,cLabel='V. D.(%)',meanLabel='%',midName='abs mean',cmap=diffMap)
            else:
                plotPlane(m,x,y,per*100,R,z[i],mean,vmin*100,vmax*100,isFault=isFault,head=head,isVol=isVol,cLabel='V. A.(%)',midName='$v_0$',cmap=cmap)#cmap
            #plotPlane(m,x,y,out,R,z[i],mean,vmin,vmax,isFault=True,head=head,isVol=False)
            plt.gca().set(facecolor='w')
            if self1 !='':
                color='k'
                v1 = V1[:,:,i]
                #printplt.arrow(x0+0.01*dx1,y0-0.05*dx1,0,0.03*dx1,color='b')
                ac.plot([x0+0.01*dx1,x0+0.01*dx1],\
                    [y0-0.05*dx1,y0-0.05*dx1+0.03*dx1],color=color,linewidth=0.75)
                ac.text(x0+0.01*dx1,y0-0.05*dx1,'0.03',ha='left',va='top',color='k',size=10)
                for ii in range(v1.shape[0]):
                    for jj in range(v1.shape[1]):
                        if vR!='':
                            if out1[ii,jj]:
                                continue
                        dX,dY=[np.imag(v1[ii,jj])*dx1,np.real(v1[ii,jj])*dx1]
                        #plt.arrow(x1[jj]-0.5*dX,y1[ii]-0.5*dY,dX,dY,color='b',)
                        ac.plot([x1[jj]-0.5*dX,x1[jj]+0.5*dX],\
                            [y1[ii]-0.5*dY,y1[ii]+0.5*dY],color=color,linewidth=0.75)
            #fig.tight_layout()
            headNew = head
            if isDiff:
                headNew=head+'Diff'
            plt.savefig('%s/%s_%f.%s'%(resDir,headNew,self.z[i],FORMAT),dpi=300)
            print('###################################',self.z[i])
            plt.close()
    def plotByP2(self,runPath='DS',head='res',self1='',vR='',maxA=0.02,P2=[],N=300):
        resDir = runPath+'/'+'plot/'
        nxyz,la0,lo0,Z = self.config.output()
        La = P2[0][0]+(P2[1][0]-P2[0][0])/N*np.arange(N)
        Lo = P2[0][1]+(P2[1][1]-P2[0][1])/N*np.arange(N)
        dist= DistAz(P2[0][0],P2[0][1],P2[1][0],P2[1][1]).getDelta()* 111.19
        Dist = np.arange(N)/N*dist
        Z = P2[0][2]+(P2[1][2]-P2[0][2])/N*np.arange(N)
        la = La.reshape([1,-1])+Z.reshape([-1,1])*0
        lo = Lo.reshape([1,-1])+Z.reshape([-1,1])*0
        z  =  La.reshape([1,-1])*0+Z.reshape([-1,1])
        print('la',la)
        print('lo',lo)
        print('z',z)
        for TF in [True, False]:
            V= self.Output(la,lo,z,isPer=TF)
            if TF:
                HEAD = head+'RELA'
            else:
                HEAD = head+'ABS'
            plt.close()
            plt.figure()
            if np.abs(la.min()-la.max())>np.abs(lo.min()-lo.max()):
                ax=plt.pcolor(La,Z,V,cmap=cmap,shading='auto')
                plt.axes().set_aspect(1/111.19)
                #plt.xlabel('la')
            else:
                ax=plt.pcolor(Lo,Z,V,cmap=cmap,shading='auto')
                plt.axes().set_aspect(1/(111.19*np.cos(La.mean()/180*np.pi)))
                #plt.xlabel('lag')
            plt.ylim([Z[-1],Z[0]])
            plt.colorbar()
            plt.title('%s %.2f %.2f %.2f %.2f'%(HEAD,P2[0][0],P2[0][1],P2[1][0],P2[1][1]))
            plt.savefig('%s/%s_ %.2f_%.2f+%.2f_%.2f.jpg'%(resDir,HEAD,\
                P2[0][0],P2[0][1],P2[1][0],P2[1][1]),dpi=500)
            plt.close()
            R = [la0.min(),la0.max(),lo0.min(),lo0.max()]
            m = mt.genBaseMap(R)
            m.etopo()
            for fault in faultL:
                if fault.inR(R):
                    fault.plot(m,markersize=0.3)
            vX,vY=m(mt.volcano[:,0],mt.volcano[:,1])
            m.plot(vX, vY,'^r')
            pX,pV=m(Lo,La)
            m.plot(Lo,La,'-b')
            plotLaLoLine(m,dLa=5,dLo=5)
            plt.title('%s %.2f %.2f %.2f %.2f'%(HEAD,P2[0][0],P2[0][1],P2[1][0],P2[1][1]))
            plt.savefig('%s/%s_ %.2f_%.2f+%.2f_%.2f_map.jpg'%(resDir,HEAD,\
                P2[0][0],P2[0][1],P2[1][0],P2[1][1]),dpi=500)
            plt.close()
            
        plt.close()
    def plotArrByZ(self,runPath='DS',head='res',maxA=0.02):
        resDir = runPath+'/'+'plot/'
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        nxyz,la,lo,z = self.config.output()
        for i in range(self.nxyz[-1]):
            plt.close()
            plt.figure(figsize=[12,8])
            R = [la.min(),la.max(),lo.min(),lo.max()]
            m = mt.genBaseMap(R)
            v = self.v[:,:,i]
            #print(v[:10,:10])
            x,y= m(lo,la)
            dx = (x.max()-x.min())/maxA/nxyz[1]
            for ii in range(v.shape[0]):
                for jj in range(v.shape[1]):
                    plt.arrow(x[jj],y[ii],np.imag(v[ii,jj])*dx,np.real(v[ii,jj])*dx,color='b',\
                        )
            for fault in faultL:
                if fault.inR(R):
                    fault.plot(m,markersize=0.3)
            plt.savefig('%s/%s_%f.jpg'%(resDir,head,self.z[i]),dpi=500)
            plt.close()
    def write(self,filename,la,lo,z,isDiff=False,N=[2,2,2],A=0.05,base=0):
        with open(filename,'w') as fp:
            if self.mode != 'GS' and self.mode!='GC':
                for i in range(len(z)):
                    fp.write('%9.1f' % (z[i]))
                fp.write('\n')
            v = self.output(la,lo,z)
            if self.mode != 'GS' and self.mode!='GC':
                for k in range(len(z)):
                    for j in range(len(lo)):
                        for i in range(len(la)):
                            per=self.diff(isDiff,N,i,j,k,A,base=base)
                            #print(N,i,j,k,per)
                            fp.write('%9.3f' % (v[i,j,k]*(1+per)))
                        fp.write('\n')
            else:
                for k in range(1,len(z)):
                    for j in range(1,len(lo)-1):
                        for i in range(1,len(la)-1):
                            per=self.diff(isDiff,N,i,j,k,A,base=base)
                            fp.write('%9.3f' % (per))
                        fp.write('\n')
    def diff(self,isDiff,N,i,j,k,A,base=0):
        if isDiff:
            I = int(float(i)/float(N[0]))
            J = int(float(j)/float(N[1]))
            K = int(float(k)/float(N[2]))
            #print(i,j,k,I,J,K,((I+J+K+base)%2-0.5)*2*A)
            return ((I+J+K+base)%2-0.5)*2*A
        else:
            return 0
    def compare(self,self1,runPath='DS',head='compare'):
        resDir = runPath+'/'+'plot/'
        if not os.path.exists(resDir):
            os.mkdir(resDir)
        z = self.z.tolist()
        z.append(0)
        nxyz,la,lo,z = self.config.output()
        V            =  self.output(la,lo,z)
        V1           =  self1.output(la,lo,z)
        for i in range(self.nxyz[-1]):
            plt.close()
            plt.figure(figsize=[26,8])
            R = [la.min(),la.max(),lo.min(),lo.max()]
            v  = V[:,:,i]
            v1 = V1[:,:,i]
            laN, loN = v.shape
            midLaN = int(1.2/4*laN)
            midLoN = int(1.2/4*loN)
            mean = v[midLaN:-midLaN,midLoN:-midLoN].mean()
            #print(mean)
            dLa,dLo=getDlaDlo(R)
            v[v<0]=mean
            v1[v1<0]=mean
            #mean = v.mean()
            Per   = (v-mean)/mean
            Per1  = (v1-mean)/mean
            la,lo,per=denseLaLo(la,lo,Per[:,:])
            la,lo,per1=denseLaLo(la,lo,Per1[:,:])
            vmax=max(np.abs(per[midLaN:-midLaN,midLoN:-midLoN]).max(),\
                np.abs(per1[midLaN:-midLaN,midLoN:-midLoN]).max())
            vmin=-vmax
            dLa,dLo=getDlaDlo(R)
            plt.subplot(1,3,1)
            m = mt.genBaseMap(R)
            x,y= m(lo,la)
            plotPlane(m,x,y,per,R,z,mean,vmin,vmax,isFault=True,head=head+'o')
            plt.subplot(1,3,2)
            m = mt.genBaseMap(R)
            x,y= m(lo,la)
            plotPlane(m,x,y,per1,R,z,mean,vmin,vmax,isFault=True,head=head+'1')
            plt.subplot(1,3,3)
            dper = per - per1
            m = mt.genBaseMap(R)
            x,y= m(lo,la)
            plotPlane(m,x,y,dper,R,z,0,-0.02,0.02,isFault=True,head=head+'diff')
            plt.subtitle('depth: %.2f km mean: %.3f'%(self.z[i],mean))
            plt.savefig('%s/%s_%f.jpg'%(resDir,head,self.z[i]),dpi=500)
            #plt.ylim([35,55])
            plt.close()
    def outputRef(self,filename):
        NLa,NLo,NZ=self.v.shape
        midLa= int(NLa/2)
        midLo= int(NLa/2)
        with open(filename,'w+') as f:
            for i in range(NZ):
                v = self.v[midLa,midLo,i]
                z = self.z[i]
                if np.isnan(v):
                    v = 2
                f.write('%.2f       %.3f     %.3f     2.800    1400.0     600.0\n'%(z,v*1.7,v))

def plotPlane(m,x,y,per,R,z,mean,vmin=-0.05,vmax=0.05,isFault=True,head='res'\
    ,isVol=False,cLabel='velocity anomal',meanLabel='(km/s)',midName='mean',cmap=cmap):
    dLa,dLo=getDlaDlo(R)
    if isFault:
        for fault in faultL:
            if fault.inR(R):
                fault.plot(m,markersize=0.3,color='dimgray',linewidth=0.5)
    if isVol:
        vX,vY=m(mt.volcano[:,0],mt.volcano[:,1])
        m.plot(vX, vY,'^r')
    pc=m.pcolormesh(x,y,per,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto',rasterized=True)
    #m.drawcoastlines(linewidth=0.8, linestyle='dashdot', color='k')
    print(vmin,vmax)
    plt.title('%s %.2f km %s: %.3f %s'%(head,z,midName,mean,meanLabel))
    if 'period' in head:
        plt.title('%s %.2f s %s: %.3f %s'%(head,z,midName,mean,meanLabel))
    dLa,dLo=mt.getDlaDlo(R)
    mt.plotLaLoLine(m,dLa,dLo,dashes=[3,3],color='dimgrey',linewidth=0.5)
    plt.gca().set_position([0.15,0.2,0.6,0.7])
    figureSet.setColorbar(pc,cLabel,pos='right')
    #cbar=plt.colorbar(fraction=0.035)
    #cbar.set_label(cLabel)
    



def loadModel(file='models/prem'):
    data = np.loadtxt(file)
    z = data[:,0]
    vs = data[:,2]
    return interpolate.interp1d(z,vs)

def loadModelTK(file = 'models/tk.nc'):
    nc  = Dataset(file,'r')
    z   =  nc.variables['depth'][:]
    la  =  nc.variables['latitude'][:]
    lo  =  nc.variables['longitude'][:]
    vsv =  nc.variables['vsv'][:]
    return z,la,lo,vsv

def loadModelHJ_(phase='p',fileDir='../models/SRL_2018209_esupp_Velocity/'):
    ref = np.loadtxt('%s/Z_v%s0.txt'%(fileDir,phase))
    lo  = np.unique(ref[:,0])
    la  = np.unique(ref[:,1])
    lo.sort()
    la.sort()
    z   = np.array([0,5,10,15,20,30,60,80,100,120,150])
    v = np.zeros([len(z),len(la),len(lo),])
    for i in range(len(z)):
        Z = z[i]
        v[i,:,:]=np.loadtxt('%s/Z_v%s%d.txt'%(fileDir,phase,Z))[:,2].reshape([len(lo),len(la)]).transpose()
    return z,la,lo,v
def loadModelHJ(phase='p',file='models/USTClith2.0/USTClitho2.0.wrst.sea_level.txt'):#models/USTClith2.0/USTClitho2.0.wrst.sea_level.txt'
    data = np.loadtxt(file)
    lo  = np.unique(data[:,0])
    la  = np.unique(data[:,1])
    z  = np.unique(data[:,2])
    v = data[:,4].reshape([len(z),len(la),len(lo)]).transpose([1,2,0])
    return z,la,lo,v
def loadModelWS(phase='p',file='data/China_2015_Vs_v1.0/'):#models/USTClith2.0/USTClitho2.0.wrst.sea_level.txt'
    lo=np.arange(75,143.5,0.5)
    la=np.arange(19.5,55,0.5)
    z = np.arange(0,200,0.5)[::10]
    data = np.zeros([len(la),len(lo),len(z)])
    NPFILE= file+'model6.npy'
    if not os.path.exists(NPFILE):
        for i in range(len(la)):
            for j in range(len(lo)):
                La = la[i]
                Lo = lo[j]
                if La%1==0:
                    La = str(int(La))
                else:
                    La = '%.1f'%La
                if Lo%1==0:
                    Lo = str(int(Lo))
                else:
                    Lo = '%.1f'%Lo
                File = '%s/%s_%s.mod'%(file,Lo,La)
                if os.path.exists(File):
                    DATA = np.loadtxt(File)
                    if len(DATA.shape)==2:
                        print(File)
                        data[i,j] = DATA[::10,1]
        np.save(NPFILE,data)
    else:
        data = np.load(NPFILE)
    return z,la,lo,data

def denseLaLo(La,Lo,Per,N=500):
    Per = Per+0
    Per[np.isnan(Per)]=-1e9
    dLa = (La[-1]-La[0])/N
    dLo = (Lo[-1]-Lo[0])/N
    la  = np.arange(La[0],La[-1],dLa)[-1::-1]
    lo  = np.arange(Lo[0],Lo[-1],dLo)
    per = interpolate.interp2d(Lo, La, Per,kind='cubic')(lo,la)
    per[per<-50]=np.nan
    return la, lo, per 


def nanV(v,v0):
    v=v+0
    v0=v0+0
    v[np.isnan(v)]=-999
    v0[np.isnan(v)]=-999
    dv0 = v-v0
    dv1 = dv0*0
    dv1[:-2,:] = v[2:,:]-v[:-2,:]
    dv2 = dv0*0
    dv2[:,:-2] = v[:,2:]-v[:,:-2]
    return (dv1**2+dv2**2)**0.5*(np.abs(dv0)+0.01)<1e-5
'''
def outR(vR,la,lo):
    lo,la=np.meshgrid(lo,la)
    print(lo,la)
    #print(np.conca(la,lo).shape)
    lalo=np.concatenate((la.reshape([1,la.shape[0],la.shape[1]]),\
            lo.reshape([1,la.shape[0],la.shape[1]])),axis=0).transpose([1,2,0])
    dlalo = lalo.reshape([lalo.shape[0],lalo.shape[1],1,2])- vR.reshape([1,1,-1,2])
    dlalo2 = np.concatenate((dlalo[:,:,:-1].reshape([1,dlalo.shape[0],dlalo.shape[1],\
            dlalo.shape[2]-1,dlalo.shape[3]])\
            ,dlalo[:,:,1:].reshape([1,dlalo.shape[0],dlalo.shape[1],\
            dlalo.shape[2]-1,dlalo.shape[3]]))\
        ,axis=0).transpose([0,4,1,2,3])
    print(dlalo2.shape,dlalo.shape)
    theta = np.arcsin((dlalo2[0,0]*dlalo2[1,1]-dlalo2[0,1]*dlalo2[1,0])\
    /((dlalo2[0]**2).sum(axis=0)*(dlalo2[1]**2).sum(axis=0))**0.5)
    sumTheta=theta.sum(axis=2)
    print(sumTheta)
    return np.abs(sumTheta)<np.pi/3
'''


'''
def output(self,la,lo,z,interp=True):
        nxyz    = [len(la),len(lo),len(z)]
        nxyzTmp = [len(la),len(lo),len(self.z)]
        if interp == False:
            return self.v
        v       = np.zeros(nxyz)
        vTmp    = np.zeros(nxyzTmp)
        #print(z)
        V= self.v
        V[np.isnan(V)]=-1e9
        for i in range(nxyzTmp[-1]):
            vTmp[:,:,i] = interpolate.interp2d(self.lo, self.la, V[:,:,i],bounds_error=False,fill_value=1e-8,kind='linear')(lo,la)
        if la[-1]<la[0]:
            vTmp = vTmp[::-1]
        if lo[-1]<lo[0]:
            vTmp = vTmp[:,::-1]
        for i in range(nxyz[0]):
            for j in range(nxyz[1]): 
                v[i,j,:] = interpolate.interp1d(self.z,vTmp[i,j])(z)
        v[v<0]=np.nan
        return v
    def OutputGriddata(self,la,lo,z):
        Lo,La,Z = np.meshgrid(self.lo,self.la,self.z)
        V = self.v.reshape([-1])
        V[np.isnan(V)]=-1e9
        points = np.concatenate((Lo.reshape([-1,1]),La.reshape([-1,1]),\
            Z.reshape([-1,1])),axis=1)
        v=interpolate.griddata(points,V,(lo,la,z),method='cubic')
        v[v<0]=np.nan
        return v
    def Output(self,la,lo,z,isPer=False,vR=''):
        #Lo,La,Z = np.meshgrid(self.lo,self.la,self.z)
        V = self.v.copy()
        if vR !='':
            out  = outR(vR,self.la,self.lo)
        if isPer:
            for i in range(V.shape[-1]):
                v= V[:,:,i]
                if vR!='':
                    v[out]=np.nan
                V[:,:,i]/=v[np.isnan(v)==False].mean()
            V-=1
        V[np.isnan(V)]=-1e9
        shape = list(la.shape)
        shape.append(1)
        points = np.concatenate((la.reshape(shape),\
            lo.reshape(shape),z.reshape(shape)),axis=-1)
        laIndex = self.la.argsort()
        v=interpolate.interpn((self.la[laIndex],self.lo,self.z),V[laIndex],points,method='linear')
        v[v<-10]=np.nan
        return v

    def denseLaLo(self,per,N=500):
        dLa = (self.la[-1]-self.la[0])/N
        dLo = (self.lo[-1]-self.lo[0])/N
        la  = np.arange(self.la[0],self.la[-1]+1e-5*dLa,dLa)[-1::-1]
        lo  = np.arange(self.lo[0],self.lo[-1]+1e-5*dLa,dLo)
        per = interpolate.interp2d(self.lo, self.la, per,kind='cubic')(lo,la)
        return la, lo, per
'''