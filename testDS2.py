import os
import sys
from imp import reload
from tkinter.tix import Tree
from SeismTool.SurfDisp import run
from SeismTool.mathTool import mathFunc
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto
isRun=True
isSave=False
isSurfNet = True
isConv=False
isPredict=True


isAll    = False
run.d.Vav=-1
isDisQC =True
isCoverQC = True
R = run.run(run.runConfig(run.paraTrainTest))
resDir0 = R.config.para['resDir']
resDir = R.config.para['resDir']
R.config.para['resDir']=resDir

#R.calCorrOneByOne()
R.loadCorr(isLoad=True,isLoadFromMat=True,isGetAverage=True,isDisQC=isDisQC,isAll=True,isSave=(isSave and isSurfNet),isAllTrain=False,isControl=False)#True
R.getDisCover()

if isSurfNet:
    for i in range(5):
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
            R.plotStaDis(isAll=True)
            if isPredict:
                R.showTest()
            if isRun:
                R.preDS(isByTrain=True)
                R.preDSTrain()
                R.preDSSyn(isByTrain=True)
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


resDir = resDir0[:-1]+('_%d/'%0)
R.config.para['resDir']=resDir
if isSurfNet:
    if isRun:
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