isCheckSignal=False
isNew = True
isRun=True
isSave=False
isPlot=True
isSurfNet = True
isConv=False
isPredict=True
isCheck=True
isStaCheck = False
N=1

#from SurfDisp.dispersion import model
import obspy
from glob import glob
from SeismTool.io import seism
from matplotlib import pyplot as plt
import numpy as np
import os
tmpDir = '/NET/common/CEA/CEA1s/'
staInfoDir='../stations/CEAINFO/'
staFile = '../stations/CEA.sta_labeled_sort'
time0 = obspy.UTCDateTime(2007,1,1)
time1 = obspy.UTCDateTime(2012,12,31)
if isStaCheck:
    stations = seism.StationList(staFile)
    if not os.path.exists(staInfoDir):
        os.makedirs(staInfoDir)
    for sta in stations:
        plt.close()
        plt.figure(figsize=[5,5])
        print(sta)
        with open('%s/%s.%s.loc'%(staInfoDir,sta['net'],sta['sta']),'w+') as f:
            for time in np.arange(time0.timestamp,time1.timestamp,86400*30):
                sacFiles = sta.getFileNames(time)[-1]
                if len(sacFiles)>0:
                    sacFile = sacFiles[0]
                    if os.path.exists(sacFile):
                        try:
                            trace = obspy.read(sacFile,headonly=True,check_compression=False)[0]
                        except:
                            continue
                        else:
                            pass
                        la = trace.stats['sac']['stla']
                        lo = trace.stats['sac']['stla']
                        el = trace.stats['sac']['stel']
                        print('%s %.6f %.6f %.6f'%(obspy.UTCDateTime(time).strftime('%Y:%m:%d-%H%M%S'),la,lo,el))
                        f.write('%s %.6f %.6f %.6f\n'%(obspy.UTCDateTime(time).strftime('%Y:%m:%d-%H%M%S'),la,lo,el))
                        plt.plot(lo,la,'.k')
        plt.savefig('%s/%s.%s.jpg'%(staInfoDir,sta['net'],sta['sta']),dpi=300)
        plt.close()
    exit()
    


import os
import sys
from imp import reload
from tkinter.tix import Tree
from SeismTool.SurfDisp import run
from SeismTool.mathTool import mathFunc
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto
import h5py

if isCheckSignal:
    R = run.run(run.runConfig(run.paraTrainTest))
    para = R.config.para
    para['isIqual']=False
    para['minSNRL']=[0]
    para['time0']  = 0
    if not os.path.exists(R.config.para['matH5']):
        R.calCorrOneByOne()

    sta     = run.seism.StationList(para['stationFileL'][0])
    q  = run.seism.QuakeL(para['quakeFileL'][0])
    dRatioD={}
    fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][0],quakeD=q,dRatioD=dRatioD)
    corrL = run.d.corrL()
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
    

    '''
    i=0
    para['sacPara']['delta0']=1.
    para['sacPara']['freq']=[1/200,1/6]
    para['sacPara']['gaussianTail']=800
    para['sacPara']['corners']=4
    para['maxCount']=1024
    corrL = para['dConfig'].quakeCorr(q[-20:],R.stations,\
            byRecord=True,remove_resp=para['remove_respL'][0],\
            minSNR=0,isLoadFv=False,\
            fvD={},isByQuake=True,para=para['sacPara'],resDir=para['eventDir'],maxCount=para['maxCount'],up=para['up'],isIqual=False)
    '''
    fvDGet ={}
    allCount=0
    validCount = 0
    getCount =0
    fvCount=0
    for corr in corrL:
        modelFile = corr.modelFile
        if modelFile in fvd:
            fvCount+=1
            fvRef = fvd[modelFile]
            f = fvRef.f
            f =f[f>1/170]
            f =f[f<1/8]
            if len(f)==0:
                continue
            allCount+=len(f)
            if len(corr.xx)==0:
                continue
            fvGet=run.d.corr.getFV(corr,f,fvRef,-100+f*0,100+f*0,1000+f*0,-0.02+f*0,0.02+f*0,minSNR=0+para['T']*0,N=50,v0=1.5,v1=5.,k=0,isControl=False,isByLoop=False,isStr=True)
            validCount+=len(fvGet.f)
            fvDGet[modelFile]=fvGet
            if len(f)/len(fvGet.f)-1>0.1:
                print(modelFile,corr.dDis,corr.dis,'wrong')
                continue
            getCount+=1
            print(validCount/allCount,getCount/fvCount,fvCount/len(fvd))

    saveDir='../models/New/Pairs_pvtsel/'
    run.d.saveFVD(fvDGet,sta,q,saveDir,'pair',isOverwrite=True)
    exit()

isAll    = False
run.d.Vav=-1
isDisQC =True
isCoverQC = True
R = run.run(run.runConfig(run.paraTrainTest))
resDir0 = R.config.para['resDir']
resDir = R.config.para['resDir']
R.config.para['resDir']=resDir

if isNew:
    R.config.para['pairDirL'][0]='../models/New/Pairs_pvtsel/'

if not os.path.exists(R.config.para['matH5']):
    R.calCorrOneByOne()

#R.calCorrOneByOne()
R.loadCorr(isLoad=True,isLoadFromMat=True,isGetAverage=True,isDisQC=isDisQC,isAll=True,isSave=(isSave and isSurfNet),isAllTrain=False,isControl=False,isCheck=isCheck)#True
R.getDisCover()

if isSurfNet:
    for i in range(N):
        resDir=resDir0[:-1]+('_%d/'%i)
        R.config.para['resDir']=resDir
        tmpDir='predict/'+R.config.para['resDir'].split('/')[-2]+'/'
        if os.path.exists(tmpDir+'resOnTrainTestValid'):
            with open(tmpDir+'resOnTrainTestValid') as f:
                R.config.para['modelFile']=f.readline()[:-1]
        #R.config.para['randA'] = 0.0
        if isRun:
            R.model=None
            R.loadModelUp()
        elif isPredict:
            R.model=None
            R.loadModelUp(file=R.config.para['modelFile'])
        run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True,isRun=isRun,isAll=isAll)
        if isRun:
            R.calFromCorrL()
        run.run.loadRes(R)
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        run.run.analyRes(R,format='eps')
        if i==0:
            run.run.plotGetAvDis(R)
            run.run.plotGetDis(R)
            R.plotTrainDis()
            R.plotStaDis()
            #R.plotStaDis(isAll=True)
            if isPredict:
                R.showTest()
            R.preDS(isByTrain=True,do=isRun)
            R.preDSTrain(do=isRun)
            R.preDSSyn(isByTrain=True,do=isRun)
        R.config.para['resDir']=resDir[:-1]+'_rand2/'
        #R.config.para['randA'] = 0.05
        if isRun:
            R.calFromCorrL(isRand=True)
        run.run.loadRes(R)
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        run.run.analyRes(R,format='eps')
        if i==0:
            run.run.plotGetAvDis(R)
            run.run.plotGetDis(R)
else:
    run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True,isRun=False,isAll=isAll)

import time
resDir = resDir0[:-1]+('_%d/'%0)
R.config.para['resDir']=resDir
if isSurfNet:
    if isPlot:
        if N==1 and isRun:
            time.sleep(60*25)
        R.loadAndPlot(R.DS,isPlot=False)
        R.loadAndPlot(R.DSTrain,isPlot=False)
        R.compare(R.DS,R.DSTrain,isCompare=True)
        R.loadAndPlot(R.DSSyn,isPlot=True)

if isConv:
    resDirAllSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAllTra/'
    resDirSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvTra/'
    resDirAvSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAvTra/'
    if isRun :
        run.run.plotDVK(R,R.fvD0,isPlot=False)
        fvDAll = run.run.calByDKV(R,R.corrL1,fvD0=R.fvD0,isControl=False,isByLoop=False)
        run.d.saveFVD(fvDAll,R.stations,R.quakes,resDirAllSave,'pair',isOverwrite=True)
        run.run.plotDVK(R,fvDAll,fvD0=R.fvD0,isRight=True,format='eps')
        run.run.plotDVK(R,fvDAll,fvD0=R.fvD0,isRight=False,format='eps')
        fvD = run.run.calByDKV(R,R.corrL,isControl=True,isByLoop=False)
        R.fvDGet = fvD
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        run.d.saveFVD(fvD,R.stations,R.quakes,resDirSave,'pair',isOverwrite=True)
        run.d.saveFVD(R.fvAvGet,R.stations,R.quakes,resDirAvSave,'NEFile',isOverwrite=True)
    else:
        R.config.para['resDir']=resDirAllSave
        R.loadRes()
        fvDAll =R.fvDGet
        R.config.para['resDir']=resDir
        run.run.plotDVK(R,fvDAll,fvD0=R.fvD0,isRight=True,format='eps')
        run.run.plotDVK(R,fvDAll,fvD0=R.fvD0,isRight=False,format='eps')
        R.config.para['resDir']=resDirSave
        R.loadRes()
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
    R.config.para['resDir']=resDir[:-1]+'_tra/'
    run.run.analyRes(R,format='eps')
    run.run.plotGetAvDis(R)

from glob import glob
saveDir='predict/'+resDir0.split('/')[-2]
if isSurfNet:
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?/resOnTrainTestValid')] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet',method='Surf-Net',isStd=True,isRand='F')
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_rand2/resOnTrainTestValid')] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_rand',method='Surf-Net',isStd=True,isRand='T')

if isConv:
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_tra/resOnTrainTestValid')] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_tra',method='conventional',isStd=False,isRand='F')

import obspy
from glob import glob
from SeismTool.io import seism
from matplotlib import pyplot as plt
from SeismTool.SurfDisp import run
import numpy as np
import os
if False:
    pass

    staD = {}
    R = run.run(run.runConfig(run.paraTrainTest))
    R.loadCorr(isLoad=True,isLoadFromMat=True,isGetAverage=True,isDisQC=isDisQC,isAll=True,isSave=(isSave and isSurfNet),isAllTrain=False,isControl=False,isCheck=isCheck,staD=staD) 
    resDir0 = R.config.para['resDir']
    resDir = R.config.para['resDir']
    R.config.para['resDir']=resDir
    resDir = resDir0[:-1]+('_%d/'%0)
    resDirLocSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/loc2/'
    time0 = obspy.UTCDateTime(2009,1,1).timestamp/86400
    if not os.path.exists(resDirLocSave):
        os.makedirs(resDirLocSave)
    for key in staD:
        plt.close()
        plt.figure(figsize=[5,5])
        staInfo = np.array(staD[key])
        plt.subplot(2,1,1)
        plt.plot(staInfo[:,0]/86400-time0,staInfo[:,1],'.k')
        plt.xlabel('t/d')
        plt.ylabel('la')
        plt.subplot(2,1,2)
        plt.plot(staInfo[:,0]/86400-time0,staInfo[:,2],'.k')
        plt.xlabel('t/d')
        plt.ylabel('lo')
        plt.savefig('%s/%s.jpg'%(resDirLocSave,key),dpi=300)