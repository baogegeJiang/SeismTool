from tkinter import TRUE
from numpy import True_


isCheckSignal=False
isNew = True
isRun= True
isDS = True
isSave= True
isPlot=True
isAn= True
isSurfNet = True
isConv=  False
doL=[1]
isPredict= True
isCheck=True
isStaCheck = False
N=5
isHalf=2
isRePick=False
doRePick=False
threshold=0.015
#from SurfDisp.dispersion import model
from tkinter.ttk import Label
from trace import Trace
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
    print('checking')
    R = run.run(run.runConfig(run.paraTrainTest))
    para = R.config.para
    para['isIqual']=False
    para['minSNRL']=[0]
    para['time0']  = 0
    R.config.para['matH5']='/media/jiangyr/1TSSD/trainTest1Hz_20220402_snrSTD3V15.h5'
    R.config.para['matH5']='/media/jiangyr/1TSSD/trainTest4Hz_check_snr0_from-384-1152V10.h5'
    if not os.path.exists(R.config.para['matH5']):
        R.calCorrOneByOne()
    
    sta     = run.seism.StationList(para['stationFileL'][0])
    q  = run.seism.QuakeL(para['quakeFileL'][0])
    dRatioD={}
    fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][0],quakeD=q,dRatioD=dRatioD,time0=R.config.para['time0'],isCheck=isCheck)
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
reload(run.d)
if isCheckSignal:
    fvDGet ={}
    allCount=0
    validCount = 0
    getCount =0
    fvCount=0
    for corr in corrL:
        xx = corr.xx
        modelFile = corr.modelFile
        if modelFile in fvd:
            fvCount+=1
            fvRef = fvd[modelFile]
            f = fvRef.f
            travel = corr.dDis/4
            MAXT  = travel*2
            f =f[f>1/MAXT]
            f =f[f>1/170]
            f =f[f<1/8]
            if len(f)==0:
                continue
            allCount+=len(f)
            if len(corr.xx)==0:
                continue
            fvGet=run.d.corr.getFV(corr,f,fvRef,-100+f*0,100+f*0,1000+f*0,-0.03+f*0,0.03+f*0,minSNR=0+para['T']*0,N=100,v0=1.5,v1=5.,k=0,isControl=False,isByLoop=False,isStr=True)
            #if 1/fvGet.f.min()-1/fvGet.f.max()<5:
            #    continue
            validCount+=len(fvGet.f)
            fvDGet[modelFile]=fvGet
            if len(f)/len(fvGet.f)-1>0.1:
                #print(modelFile,corr.dDis,corr.dis,'wrong')
                continue
            getCount+=1
            print(validCount/allCount,getCount/fvCount,fvCount/len(fvd))
    saveDir='../models/New/Pairs_pvtsel_4Hz_-384-1152V10/'
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
    R.config.para['pairDirL'][0]='../models/New/Pairs_pvtsel_4Hz_-384-1152V10/'

if not os.path.exists(R.config.para['matH5']):
    R.calCorrOneByOne()

#R.calCorrOneByOne()
R.loadCorr(isLoad=True,isLoadFromMat=True,isGetAverage=True,isDisQC=isDisQC,isAll=True,isSave=(isSave and isSurfNet),isAllTrain=False,isControl=False,isCheck=isCheck)#True
R.getDisCover()

if isSurfNet:
    if 1 in doL:
        for i in range(N):
            resDir=resDir0[:-1]+('_%d/'%i)
            R.config.para['resDir']=resDir
            R.config.para['disAmp']=1
            tmpDir='predict/'+R.config.para['resDir'].split('/')[-2]+'/'
            if len(glob(tmpDir+'resOnTrainTestValid_0.0??'))>0:
                with open(glob(tmpDir+'resOnTrainTestValid_0.0??')[0]) as f:
                    R.config.para['modelFile']=f.readline()[:-1]
            #R.config.para['randA'] = 0.0
            if isRun:
                #R.model=None
                R.loadModelUp()
            elif isPredict:
                #R.model=None
                R.loadModelUp(file=R.config.para['modelFile'])
            run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True,isRun=isRun,isAll=isAll)
            if isRun:
                R.calFromCorrL(isRand=True)
            run.run.loadRes(R,isCoverQC=isCoverQC,isDisQC=isDisQC)
            run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
            run.run.getAV(R)
            run.run.limit(R)
            if isAn:
                run.run.analyRes(R,threshold=threshold,format='eps')
            if i==0:
                if isAn:
                    run.run.plotGetAvDis(R)
                    run.run.plotGetDis(R)
                    R.plotTrainDis()
                    R.plotStaDis()
                #R.plotStaDis(isAll=True)
                if isPredict:
                    R.showTest()
                R.preDS(isByTrain=True,do=isDS,isRun=isDS)
                R.preDSTrain(do=isDS,isRun=isDS)
                R.preDSSyn(isByTrain=True,do=isDS,isRun=isDS)
            if isHalf!=False:
                R.config.para['resDir']=resDir[:-1]+'_half/'
                run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob',isHalf=isHalf)
                run.run.getAV(R)
                run.run.limit(R)
                if isAn:
                    run.run.analyRes(R,threshold=threshold,format='eps',isHalf=True)
            if isRePick:
                R.config.para['resDir']=resDir[:-1]+'_re/'
                if doRePick:
                    R.rePick()
                run.run.loadRes(R,isCoverQC=isCoverQC,isDisQC=isDisQC)
            if False:
                R.config.para['resDir']=resDir[:-1]+'_rand2/'
                #R.config.para['randA'] = 0.05
                if isRun:
                    R.calFromCorrL(isRand=True)
                run.run.loadRes(R,isCoverQC=isCoverQC,isDisQC=isDisQC)
                run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
                run.run.getAV(R)
                run.run.limit(R)
                run.run.analyRes(R,format='eps')
                if i==0:
                    run.run.plotGetAvDis(R)
                    run.run.plotGetDis(R)
    if 2 in doL:
        for i in range(N):
            resDir=resDir0[:-1]+('_%d_noDis/'%i)
            R.config.para['resDir']=resDir
            R.config.para['disAmp']=0
            tmpDir='predict/'+R.config.para['resDir'].split('/')[-2]+'/'
            if os.path.exists(tmpDir+'resOnTrainTestValid'):
                with open(tmpDir+'resOnTrainTestValid') as f:
                    R.config.para['modelFile']=f.readline()[:-1]
            #R.config.para['randA'] = 0.0
            if isRun:
                #R.model=None
                R.loadModelUp()
            elif isPredict:
                #R.model=None
                R.loadModelUp(file=R.config.para['modelFile'])
            run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True,isRun=isRun,isAll=isAll)
            if isRun:
                R.calFromCorrL(isRand=True)
            run.run.loadRes(R,isCoverQC=isCoverQC,isDisQC=isDisQC)
            run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
            run.run.getAV(R)
            run.run.limit(R)
            if isAn:
                run.run.analyRes(R,threshold=threshold,format='eps')
            if i==-1:
                if isAn:
                    run.run.plotGetAvDis(R)
                    run.run.plotGetDis(R)
                    R.plotTrainDis()
                    R.plotStaDis()
                    R.plotStaDis(isAll=True,isAllQuake=True)
                if isPredict:
                    R.showTest()
                R.preDS(isByTrain=True,do=isRun)
                R.preDSTrain(do=isRun)
                R.preDSSyn(isByTrain=True,do=isRun)
            if isHalf!=False:
                R.config.para['resDir']=resDir[:-1]+'_half/'
                run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob',isHalf=isHalf)
                run.run.getAV(R)
                run.run.limit(R)
                if isAn:
                    run.run.analyRes(R,threshold=threshold,format='eps',isHalf=True)
            if False:
                R.config.para['resDir']=resDir[:-1]+'_rand2__noDis/'
                #R.config.para['randA'] = 0.05
                if isRun:
                    R.calFromCorrL(isRand=True)
                run.run.loadRes(R,isCoverQC=isCoverQC,isDisQC=isDisQC)
                run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
                run.run.getAV(R)
                run.run.limit(R)
                run.run.analyRes(R,format='eps')
                if i==0:
                    run.run.plotGetAvDis(R)
                    run.run.plotGetDis(R)
else:
    run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True,isRun=False,isAll=isAll)

R.config.para['disAmp']=1
import time
resDir = resDir0[:-1]+('_%d/'%0)
R.config.para['resDir']=resDir
if isSurfNet and (1 in doL):
    if isPlot:
        if N==1 and isDS:
            time.sleep(60*50)
        R.loadAndPlot(R.DS,isPlot=False)
        R.loadAndPlot(R.DSTrain,isPlot=False)
        R.compare(R.DS,R.DSTrain,isCompare=True)
        R.loadAndPlot(R.DSSyn,isPlot=True)

fvRef = run.d.averageFVL([R.fvD0[key] for key in R.fvD0],fL=1/R.config.para['T'])
if isConv:
    resDirAllSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAllTra/'
    resDirSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvTra/'
    resDirAvSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAvTra/'
    resDirSaveRand = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvTraRand/'
    resDirAvSaveRand = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAvTraRand/'
    if isRun :
        if os.path.exists(resDirAllSave):
            R.config.para['resDir']=resDirAllSave
            R.loadRes()
            fvDAll =R.fvDGet
        else:
            run.run.plotDVK(R,R.fvD0,fvRef,isPlot=False)
            fvDAll = run.run.calByDKV(R,R.corrL1,fvRef,fvD0=R.fvD0,isControl=False,isByLoop=False)
            run.d.saveFVD(fvDAll,R.stations,R.quakes,resDirAllSave,'pair',isOverwrite=True)
        R.config.para['resDir']=resDir
        R.config.para['vPer']=0.08
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=False,format='eps',isRand=True)
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=True,format='eps',isRand=True)
        
        fvD = run.run.calByDKV(R,R.corrL,fvRef,isControl=True,isByLoop=False,isRand=True,randA=R.config.para['randA'])
        R.fvDGet = fvD
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        run.d.saveFVD(fvD,R.stations,R.quakes,resDirSaveRand,'pair',isOverwrite=True)
        run.d.saveFVD(R.fvAvGet,R.stations,R.quakes,resDirAvSaveRand,'NEFile',isOverwrite=True)
        R.config.para['resDir']=resDir[:-1]+'_tra_rand/'
        run.run.analyRes(R,threshold=threshold,format='eps')
        run.run.plotGetAvDis(R)
        if isHalf:
            R.config.para['resDir']=resDir[:-1]+'_tra_rand_half/'
            run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob',isHalf=isHalf)
            run.run.getAV(R)
            run.run.limit(R)
            run.run.analyRes(R,format='eps',isHalf=True)
            
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=False,format='eps')
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=True,format='eps')
        R.config.para['vPer']=0.08
        fvD = run.run.calByDKV(R,R.corrL,fvRef,isControl=True,isByLoop=False)
        R.fvDGet = fvD
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        run.d.saveFVD(fvD,R.stations,R.quakes,resDirSave,'pair',isOverwrite=True)
        run.d.saveFVD(R.fvAvGet,R.stations,R.quakes,resDirAvSave,'NEFile',isOverwrite=True)
        R.config.para['resDir']=resDir[:-1]+'_tra/'
        run.run.analyRes(R,threshold=threshold,format='eps')
        run.run.plotGetAvDis(R)
        if isHalf:
            R.config.para['resDir']=resDir[:-1]+'_tra_half/'
            run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob',isHalf=isHalf)
            run.run.getAV(R)
            run.run.limit(R)
            run.run.analyRes(R,threshold=threshold,format='eps',isHalf=True)
    else:
        R.config.para['resDir']=resDirAllSave
        R.loadRes()
        fvDAll =R.fvDGet
        R.config.para['resDir']=resDir
        R.config.para['vPer']=0.10
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=True,format='eps')
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=False,format='eps')
        R.config.para['resDir']=resDirSave
        R.loadRes()
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        R.config.para['resDir']=resDir[:-1]+'_tra/'
        run.run.analyRes(R,threshold=threshold,format='eps')
        run.run.plotGetAvDis(R)
        if isHalf:
            R.config.para['resDir']=resDir[:-1]+'_tra_half/'
            run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob',isHalf=isHalf)
            run.run.getAV(R)
            run.run.limit(R)
            run.run.analyRes(R,threshold=threshold,format='eps',isHalf=True)
        R.config.para['resDir']=resDir
        R.config.para['vPer']=0.10
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=False,format='eps',isRand=True)
        run.run.plotDVK(R,fvDAll,fvRef,fvD0=R.fvD0,isRight=True,format='eps',isRand=True)
        R.config.para['resDir']=resDirSaveRand
        R.loadRes()
        run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
        run.run.getAV(R)
        run.run.limit(R)
        R.config.para['resDir']=resDir[:-1]+'_tra_rand/'
        run.run.analyRes(R,threshold=threshold,format='eps')
        run.run.plotGetAvDis(R)
        if isHalf:
            R.config.para['resDir']=resDir[:-1]+'_tra_rand_half/'
            run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob',isHalf=isHalf)
            run.run.getAV(R)
            run.run.limit(R)
            run.run.analyRes(R,threshold=threshold,format='eps',isHalf=True)
    
from glob import glob
saveDir='predict/'+resDir0.split('/')[-2]
if isSurfNet:
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?/resOnTrainTestValid_%.3f'%threshold)] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet',method='Surf-Net',isStd=True,isRand='T')
    if isHalf!=False:
        M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_half/resOnTrainTestValid_%.3f'%threshold)] )
        run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_half',method='Surf-Net',isStd=True,isRand='T')
        M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_noDis_half/resOnTrainTestValid_%.3f'%threshold)] )
        run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_noDis_half',method='Surf-Net_noDis_half',isStd=True,isRand='T')
    #M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_rand2/resOnTrainTestValid')] )
    #run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_rand',method='Surf-Net',isStd=True,isRand='T')
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_noDis/resOnTrainTestValid')] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_noDis',method='Surf-Net_noDis',isStd=True,isRand='T')

if isConv:
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_tra/resOnTrainTestValid_%.3f'%threshold)] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_tra',method='conventional',isStd=False,isRand='F')
    if isHalf!=False:
        M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_tra_half/resOnTrainTestValid_%.3f'%threshold)] )
        run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_tra_half',method='conventional_half',isStd=False,isRand='F')
    M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_tra_rand/resOnTrainTestValid_%.3f'%threshold)] )
    run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_tra_rand',method='conventional_rand',isStd=False,isRand='F')
    if isHalf!=False:
        M,S=run.calData([run.loadResData(file) for file in glob(saveDir+'_?_tra_rand_half/resOnTrainTestValid_%.3f'%threshold)] )
        run.output(M,S,saveDir+'_0/resOnTrainTestValid_surfNet_tra_rand_half',method='conventional_rand_half',isStd=False,isRand='F')

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

    from obspy import read
    from SeismTool.SurfDisp import dispersion as d
    from imp import reload
    import numpy as np
    from matplotlib import pyplot as plt

    file='/media/jiangyr/1TSSD/eventSac/1288975239.80000_12.92000_122.96999/FJ.DSXP._remove_resp_DISP.BHZ'
    trace = read(file)[0]
    trace.filter('bandpass',freqmin=1/160, freqmax=1/6, corners=4, zerophase=True)
    dist=trace.stats.sac['gcarc']*111.19
    timeL = np.arange(len(trace.data))/trace.stats.sampling_rate+trace.stats.sac['b']
    time0=0
    time1=dist/1.25
    timeB=dist/5
    timeE=dist/2.5
    tRef = dist/3.5
    T = np.arange(160,9,-1)
    f = 1/T
    data=trace.data[(timeL>time0)*(timeL<time1)]
    timeLNew= timeL[(timeL>time0)*(timeL<time1)]
    Phi,std,S0,S1 = d.calPhi(data,timeLNew,f,tRef=f*0+tRef,gamma=20,isS=True)
    d.showS(timeLNew,data,f,S0,'predict/S.jpg')

    modelFile = '../models/iasp'
    zMax=900
    Tmax=400
    z= np.arange(0,zMax,10)
    TL =np.arange(1,Tmax,8)
    f = 1/TL
    z0=80
    z02=300
    dz=20
    dz2=10
    A=0.15
    model = np.loadtxt(modelFile)
    index= np.abs(model[:,0]-zMax).argmin()
    Z,Vp,Vs,Rho=model[:index+1,:4].transpose()
    vp=d.scipy.interpolate.interp1d(Z,Vp)(z)
    vs=d.scipy.interpolate.interp1d(Z,Vs)(z)
    rho=d.scipy.interpolate.interp1d(Z,Rho)(z)

    v0,GP0,GS0,GRho0=d.model2kernel(vp,vs,z,f,rho,wave='rayleigh',mode=1,velocity='phase',flat_earth=True)
    v1,GP1,GS1,GRho1=d.model2kernel(vp,vs,z,f,rho,wave='rayleigh',mode=2,velocity='phase',flat_earth=True)

    plt.close()
    linewidth=1
    plt.figure(figsize=(2,4))
    plt.subplot(2,1,1)
    plt.plot(vp,z,'k',label='$v_p$',linewidth=linewidth)
    plt.plot(vs,z,'r',label='$v_p$',linewidth=linewidth)
    plt.ylim([zMax,0])
    plt.xlabel('$v$(km/s)')
    plt.ylabel('depth(km)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(v0,TL,'k',label='$v_0$',linewidth=linewidth)
    plt.plot(v1,TL,'r',label='$v_1$',linewidth=linewidth)
    plt.ylim([T.max(),1])
    plt.legend()
    plt.xlabel('$v$(km/s)')
    plt.ylabel('$T$(s)')
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig('predict/structrue.jpg',dpi=300)
    plt.close()

    plt.close()
    Gmax=None
    Gmin=None
    plt.figure(figsize=(2,4))
    plt.subplot(2,1,1)
    #plt.pcolor(TL,z,GP0,cmap='hot',vmax=Gmax,vmin=Gmin)
    #plt.colorbar(label='$G_{p0}(km^{-1})$')
    colorL='krgbm'
    tL=[30,60,90,120,200]
    for t in tL:
        index = np.abs(TL-t).argmin()
        print(index)
        plt.plot(GP0[:,index],z,colorL[tL.index(t)],label=str(t),linewidth=linewidth)

    plt.legend()
    plt.xlabel('$G_{P0}(km^{-1})$')
    plt.ylabel('depth(s)')
    plt.ylim([zMax-200,1])
    #plt.xlim([0,0.02])
    #plt.xlim([T.max(),1])
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    plt.subplot(2,1,2)
    #plt.pcolor(TL,z,GS0,cmap='hot',vmax=Gmax,vmin=Gmin)
    #plt.colorbar(label='$G_{s0}(km^{-1})$')
    for t in tL:
        index = np.abs(TL-t).argmin()
        plt.plot(GS0[:,index],z,colorL[tL.index(t)],label=str(t),linewidth=linewidth)

    plt.xlabel('$G_{s0}(km^{-1})$')
    plt.ylabel('depth(s)')
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    plt.ylim([zMax-200,1])
    #plt.xlim([0,0.1])
    plt.tight_layout()
    plt.legend()
    plt.savefig('predict/structrueG.jpg',dpi=300)



    plt.figure(figsize=(2,4))
    plt.subplot(2,1,1)
    #plt.pcolor(TL,z,GP0,cmap='hot',vmax=Gmax,vmin=Gmin)
    #plt.colorbar(label='$G_{p0}(km^{-1})$')
    colorL='krgbm'
    tL=[30,60,90,120,200]
    for t in tL:
        index = np.abs(TL-t).argmin()
        print(index)
        plt.plot(GP1[:,index],z,colorL[tL.index(t)],label=str(t),linewidth=linewidth)

    plt.legend()
    plt.xlabel('$G_{P0}(km^{-1})$')
    plt.ylabel('depth(s)')
    plt.ylim([zMax-200,1])
    #plt.xlim([0,0.02])
    #plt.xlim([T.max(),1])
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    plt.subplot(2,1,2)
    #plt.pcolor(TL,z,GS0,cmap='hot',vmax=Gmax,vmin=Gmin)
    #plt.colorbar(label='$G_{s0}(km^{-1})$')
    for t in tL:
        index = np.abs(TL-t).argmin()
        plt.plot(GS1[:,index],z,colorL[tL.index(t)],label=str(t),linewidth=linewidth)

    plt.xlabel('$G_{s0}(km^{-1})$')
    plt.ylabel('depth(s)')
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    plt.ylim([zMax-200,1])
    #plt.xlim([0,0.1])
    plt.tight_layout()
    plt.legend()
    plt.savefig('predict/structrueG1.jpg',dpi=300)

    from SeismTool.SurfDisp import run
    RNorth = run.run(run.runConfig(run.paraNorth))
    RNorth.plotStaDis(True,True,'predict/')
    RNorth.plotStaDis(False,True,'predict/')

    R = run.run(run.runConfig(run.paraTrainTest))
    resDir0 = R.config.para['resDir']
    resDir=resDir0[:-1]+('_%d/'%0)
    R.config.para['resDir']=resDir
    R.plotStaDis(True,True)
    R.plotStaDis(False,True)
    '''
    plt.subplot(3,1,3)
    plt.pcolor(TL,z,GRho0,cmap='hot',vmax=Gmax,vmin=Gmin)
    plt.colorbar(label='$G_{rho0}(cm/g)$')
    plt.xlabel('$T$/s')
    plt.ylabel('depth(s)')
    plt.gca().set_yscale('log')

    '''
    T=np.array([30])
    f=1/T
    omega=np.pi*2*f
    A=0.3
    DOmega = omega*0.5
    timeL=np.arange(0,200,0.1)
    x = 300
    phi=0

    v0=d.model2disp(vp,vs,z,f,rho,wave='rayleigh',mode=1,velocity='phase',flat_earth=True)
    u0=d.model2disp(vp,vs,z,f,rho,wave='rayleigh',mode=1,velocity='group',flat_earth=True)
    v1=d.model2disp(vp,vs,z,f,rho,wave='rayleigh',mode=2,velocity='phase',flat_earth=True)
    u1=d.model2disp(vp,vs,z,f,rho,wave='rayleigh',mode=2,velocity='group',flat_earth=True)
    for DOmega in [omega*0.5,omega*0.3,omega*0.1]:
        data0,oData0,sData0 = d.genSharpW(omega,v0,u0,DOmega,x,phi,timeL) 
        data1,oData1,sData1= d.genSharpW(omega,v1,u1,DOmega,x,phi,timeL)
        plt.figure(figsize=[4,6])
        #plt.plot(timeL,oData0,'k',linewidth=linewidth)
        plt.subplot(3,1,1)
        plt.plot(timeL,data0,'k',linewidth=linewidth,Label='0')
        plt.plot(timeL,sData0,'g',linewidth=linewidth,Label='$S$')
        plt.plot([x/v0,x/v0],[-1.5,1.5],'r',linewidth=linewidth,label='$t_0$')
        plt.legend()
        plt.xlim([timeL[0],timeL[-1]])
        plt.subplot(3,1,2)
        plt.plot(timeL,data1*A,'k',linewidth=linewidth,label='1')
        plt.plot(timeL,sData1,'g',linewidth=linewidth,Label='$S$')
        plt.plot([x/v1,x/v1],[-1.5,1.5],'r',linewidth=linewidth,label='$t_1$')
        plt.legend()
        plt.xlim([timeL[0],timeL[-1]])
        plt.subplot(3,1,3)
        plt.plot(timeL,data0+data1*A,'k',linewidth=linewidth,label='0+1')
        plt.plot([x/v0,x/v0],[-1.5,1.5],'r',linewidth=linewidth,label='$t_0$')
        plt.xlabel('t/s')
        plt.legend()
        plt.xlim([timeL[0],timeL[-1]])
        plt.suptitle('$\omega$: %.1f mHz, D$\omega$: %.1fmHz'%(omega*1000,DOmega*1000))
        plt.savefig('predict/DW_%.1f_%.1f.jpg'%(omega*1000,DOmega*1000),dpi=300)
        plt.close()




if False:
    vpNew = (1+np.exp(-(z-z0)**2/dz**2)*A)*vp
    vsNew = (1+np.exp(-(z-z0)**2/dz**2)*A)*vs
    rhoNew = (1+np.exp(-(z-z0)**2/dz**2)*A)*rho

    vpNew2 = (1+np.exp(-(z-z02)**2/dz2**2)*A)*vp
    vsNew2 = (1+np.exp(-(z-z02)**2/dz2**2)*A)*vs
    rhoNew2 = (1+np.exp(-(z-z02)**2/dz2**2)*A)*rho

    TL =np.arange(1,Tmax,8)
    f = 1/TL
    RPL =np.array([d.model2disp(vp,vs,z,f,rho,wave='rayleigh',mode=i,velocity='phase',flat_earth=True)for i in range(1,2)])
    RPLNew =np.array([d.model2disp(vpNew,vsNew,z,f,rhoNew,wave='rayleigh',mode=i,velocity='phase',flat_earth=True)for i in range(1,2)])
    RPLNew2 =np.array([d.model2disp(vpNew2,vsNew2,z,f,rhoNew2,wave='rayleigh',mode=i,velocity='phase',flat_earth=True)for i in range(1,2)])

    plt.close()
    #plt.subplots_adjust(wspace=0.2)
    plt.figure(figsize=(2,4))
    plt.subplot(3,1,1)
    #plt.plot(vp,z,'k',label='$v_p$',linewidth=0.5)
    plt.plot(vs,z,'k',label='$v_s$',linewidth=1)
    #plt.plot(vpNew,z,'--r',label='$v_p$',linewidth=0.5)
    plt.plot(vsNew,z,'--r',label='$v_s^0$',linewidth=1)
    #plt.plot(vpNew2,z,'--r',label='$v_p$',linewidth=0.5)
    #plt.plot(vsNew2,z,'--g',label='$v_s^1$',linewidth=0.5)
    plt.ylim([500,-2])
    plt.xlabel('$v$(km/s)')
    plt.ylabel('depth(km)')
    plt.legend()
    plt.subplot(3,1,2)
    RPL[RPL==0]=np.nan
    RPLNew[RPLNew==0]=np.nan
    RPLNew2[RPLNew2==0]=np.nan
    plt.plot(RPL.transpose(),TL,'k',linewidth=1)
    plt.plot(RPLNew.transpose(),TL,'--r',linewidth=1)
    #plt.plot(RPLNew2.transpose(),TL,'--r',linewidth=1)
    plt.ylim([Tmax,1])
    plt.xlabel('$v_{phase}$(km/s)')
    #plt.gca().set_yscale('log')
    plt.ylabel('$period$(s)')
    plt.xlim([2.8,6.5])
    plt.subplot(3,1,3)
    plt.plot(RPLNew.transpose()-RPL.transpose(),TL,'k',linewidth=1)
    #plt.plot(RPLNew2.transpose(),TL,'--r',linewidth=1)
    plt.ylim([Tmax,1])
    plt.xlabel('$v_d$(km/s)')
    #plt.gca().set_yscale('log')
    plt.ylabel('$period$(s)')
    #plt.xlim([],6.5])
    #plt.legend()
    plt.tight_layout()
    plt.savefig('predict/structrue1.jpg',dpi=300)

    plt.close()
    plt.figure(figsize=(2,4))
    plt.subplot(3,1,1)
    #plt.plot(vp,z,'k',label='$v_p$',linewidth=0.5)
    plt.plot(vs,z,'k',label='$v_s$',linewidth=1)
    #plt.plot(vpNew,z,'--r',label='$v_p$',linewidth=0.5)
    #plt.plot(vsNew,z,'--r',label='$v_s^0$',linewidth=0.5)
    #plt.plot(vpNew2,z,'--r',label='$v_p$',linewidth=0.5)
    plt.plot(vsNew2,z,'--r',label='$v_s^1$',linewidth=1)
    plt.ylim([500,-2])
    plt.xlabel('depth(km)')
    plt.legend()
    plt.subplot(3,1,2)
    RPL[RPL==0]=np.nan
    RPLNew[RPLNew==0]=np.nan
    RPLNew2[RPLNew2==0]=np.nan
    plt.plot(RPL.transpose(),TL,'k',linewidth=1)
    #plt.plot(RPLNew.transpose(),TL,'--r',linewidth=0.5)
    plt.plot(RPLNew2.transpose(),TL,'--r',linewidth=1)
    plt.ylim([Tmax,1])
    plt.xlabel('$v_{phase}$(km/s)')
    #plt.gca().set_yscale('log')
    plt.ylabel('$period$(s)')
    plt.xlim([2.8,6.5])
    plt.subplot(3,1,3)
    plt.plot(RPLNew2.transpose()-RPL.transpose(),TL,'k',linewidth=1)
    #plt.plot(RPLNew2.transpose(),TL,'--r',linewidth=1)
    plt.ylim([Tmax,1])
    plt.xlabel('$v_d$(km/s)')
    #plt.gca().set_yscale('log')
    plt.ylabel('$period$(s)')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('predict/structrue2.jpg',dpi=300)