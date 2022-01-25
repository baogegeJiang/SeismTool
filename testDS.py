import os
import sys
from imp import reload
from SeismTool.SurfDisp import run
from SeismTool.mathTool import mathFunc
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto
R = run.run(run.runConfig(run.paraTrainTest))
run.d.Vav=-1
isDisQC =True
isCoverQC = True
R.calCorrOneByOne()
R.loadCorr(isLoad=True,isLoadFromMat=True,isGetAverage=True,isDisQC=isDisQC,isAll=True,isSave=True,isAllTrain=False)#True
R.getDisCover()
R.model=None
R.loadModelUp()
run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True)

R.calFromCorrL()
run.run.loadRes(R)
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.getAV(R)
run.run.limit(R)
run.run.analyRes(R,format='eps')
run.run.plotGetAvDis(R)
R.plotTrainDis()
R.plotStaDis()
R.showTest()
R.preDS()
R.preDSTrain()
R.preDSSyn()

resDir = R.config.para['resDir']

R.config.para['resDir']=resDir[:-1]+'_rand/'
R.calFromCorrL(isRand=True)
run.run.loadRes(R)
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.getAV(R)
run.run.limit(R)
run.run.analyRes(R,format='eps')
run.run.plotGetAvDis(R)

R.config.para['resDir']=resDir
R.plotDVK(R.fvD0)
fvDAll = R.calByDKV(R.corrL1,fvD0=R.fvD0,isControl=False)
run.run.plotDVK(R,fvDAll,fvD0=R.fvD0,isRight=True,format='eps')
run.run.plotDVK(R,fvDAll,fvD0=R.fvD0,isRight=False,format='eps')
fvD = run.run.calByDKV(R,R.corrL,isControl=True)
R.fvDGet = fvD
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.getAV(R)
run.run.limit(R)
resDirSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fv/'
resDirAvSave = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAv/'
run.d.saveFVD(fvD,R.stations,R.quakes,resDirSave,'pair',)
run.d.saveFVD(R.fvAvGet,R.stations,R.quakes,resDirAvSave,'NEFile')
R.config.para['resDir']=resDir[:-1]+'_tra/'
run.run.analyRes(R,format='eps')
run.run.plotGetAvDis(R)


run.run.plotGetAvDis(R)

R.config.para['resDir']=resDir[:-1]+'_up=1/'
R.config.para['up']=1
R.model=None
R.loadModelUp()
run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True)
R.calFromCorrL()
run.run.loadRes(R)
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.analyRes(R,format='eps')

R.config.para['resDir']=resDir[:-1]+'_up=3/'
R.config.para['up']=3
R.model=None
R.loadModelUp()
run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True)
R.calFromCorrL()
run.run.loadRes(R)
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.analyRes(R,format='eps')


run.run.analyRes(R,format='eps')
run.run.plotGetAvDis(R)
reload(run.d)
fvD=run.run.calByDKV(R,k=0,maxCount=len(R.corrL)*0+100)
run.np.abs(fvD[key](R.fvD0[key].f)/R.fvD0[key].v*100-100).mean()

R.loadModelUp(R.config.para['modelFile'])
R.config.para['resDir']='/media/jiangyr/MSSD/20220113V3_1_1_rand/'
run.run.calFromCorrL(R,isRand=True)
run.run.loadRes(R)
#R.getAv(isCoverQC=True,isDisQC=False)
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.getAV(R)
run.run.limit(R)

run.run.analyRes(R,format='eps')
run.run.plotGetAvDis(R)




R.config.para['resDir']='/media/jiangyr/MSSD/20220113V3_1_1/'
run.run.calFromCorrL(R)
run.run.loadRes(R)
#R.getAv(isCoverQC=True,isDisQC=False)
R.fvDGet = fvD
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.getAV(R)
run.run.limit(R)
run.run.analyRes(R,format='eps')
run.run.plotGetAvDis(R)


resDir = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fv_tra/'
resDirAv = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/fvAv_tra/'
run.d.saveFVD(fvD,R.stations,R.quakes,resDir,'pair',)
run.d.saveFVD(R.fvAvGet,R.stations,R.quakes,resDirAv,'NEFile')

sigma = run.np.ones(len(R.config.para['T']))
N =len(R.config.para['T'])
N_5=int(N/5)
sigma[:N_5]        = 1.25
sigma[N_5:2*N_5]   = 1.5
sigma[2*N_5:3*N_5] = 1.75
sigma[3*N_5:4*N_5] = 2.0
sigma[4*N_5:5*N_5] = 2.5
R.config.sigma=sigma
key='1275335509.90000_11.16000_93.70000_HE.KAB_HL.JGD'
fvD=run.run.calByDKV(R,k=0.5,maxCount=len(R.corrL)*0+100)
fvD[key](R.fvD0[key].f)/R.fvD0[key].v*100-100




#R.loadModelUp(R.config.para['modelFile'])
reload(run)
reload(run.fcn)
R.model=None
run.run.loadModelUp(R)
run.run.train(R,up=5,isRand=True,isShuffle=False,isAverage=False)
reload(run)
run.run.loadRes(R,isCoverQC=True)
R.config=run.runConfig(run.paraTrainTest)
run.run.preDS(R,isByTrain=False)
run.run.preDSTrain(R)
run.run.preDSSyn(R,isByTrain=False)
R.DS.plotHJ(R=R.config.para['R'])
R.corrL1.reSetUp(5)
run.run.calRes(R)
run.run.loadRes(R)

R.preDS(False)
R.loadAndPlot(R.DS,False)
run.run.compare(R,R.DS,R.DSTrain)

R.getDisCover()
R.loadRes()
run.run.getAv(R,isCoverQC=True)
run.run.preDS(R,isByTrain=True)
R.loadAndPlot(R.DS,isPlot=False)
R.loadAndPlot(R.DSTrain,isPlot=False)
R.compare(R.DS,R.DSTrain,isCompare=True)
R.loadAndPlot(R.DSSyn,isPlot=True)
R1.calFromCorr()
run.d.qcFvD(R.fvAvGet)
run.d.qcFvD(R.fvDGet)
run.d.compareFvD(R.fvAvGet,R.fvDAverage,1/R.config.para['T'],resDir='predict/compare300/',keyL=R.fvTest,stations=R.stations)
reload(run.d)
run.d.plotPair(R.fvAvGet,R.stations)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD0,R.stations,t=R.config.para['T'],keys=R.fvTest,)


R.loadAndPlot(R.DS,isPlot=False)
R.loadAndPlot(R.DSTrain,isPlot=False)
R.compare(R.DS,R.DSTrain,isCompare=True)


reload(run.fcn)
run.fcn.modelUp.show(R.model,R.corrLTest.x[::100],R.corrLTest.y[::100],outputDir='predict/raw/',delta=1,T=R.config.para['T'])
reload(run.d)
run.d.compareFvD(R.fvAvGet,R.fvDAverage,1/R.config.para['T'],resDir='predict/compareV10/')

run.d.plotFVM(R.fvMGet,R.fvAvGet,R.fvDAverage,resDir='predict/'+'pairsTrainTest220101V2/',isDouble=True,fL0=1/R.config.para['T'],stations=R.stations,keyL=R.fvTest)

run.d.plotFVM(R1.fvMGet,R1.fvAvGet,R1.fvAvGet,resDir='predict/'+'pairsTrainTestNorth220111V1/',isDouble=True,fL0=1/R.config.para['T'],stations=R.stations,keyL=R.fvTest)

M,V0,V1=run.d.compareInF(R.fvDAverage,R.fvAvGet,R.stations,1/R.config.para['T'],R=R.config.para['R'])
from SeismTool.mathTool import mathFunc
mathFunc.showQC('predict/QC.eps')
reload(run)
#reload(run.d)
#reload(run.seism)
R1=run.run(run.runConfig(run.paraNorth))
R1.model=R.model
#R1.calResOneByOne()

R1.loadRes(isGetQuake=False)

R2=run.run(run.runConfig(run.paraOrdos))
reload(run)
R1.config=run.runConfig(run.paraNorth)
from glob import glob
from SeismTool.io import seism
fvFileL = glob('%s/*.dat'%('../models/ayu/Pairs_avgpvt/'))

staL =[]
stations0 = seism.StationList('../stations/CEA.sta_know_few')
stations = seism.StationList()
for fvFile in fvFileL:
    key = os.path.basename(fvFile).split('-')[0]
    sta0,sta1=key.split('_')
    for sta in [sta0,sta1]:
        if sta not in staL:
            staL.append(sta)
            stations.append(stations0.Find(sta))
stations.write('../stations/CEA.sta_labeled')
stations=seism.StationList('../stations/CEA.sta_labeled')
stationsNew = seism.StationList()
staD = {}
keyL =[]
for station in stations:
    staD[station['net']+station['sta']]=station
    keyL.append(station['net']+station['sta'])

keyL.sort()
for key in keyL:
    stationsNew.append(staD[key])
stationsNew.write('../stations/CEA.sta_labeled_sort')

from SeismTool.io import seism
import sys
N= int(sys.argv[1])
n= int(sys.argv[2])
print(N,n)
para={\
'delta0' :1,
'freq'   :[-1,-1],#[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e19,
}
quakes = seism.QuakeL('CEA_quakes')
staions = seism.StationList('../stations/CEA.sta_labeled_sort')
quakes.cutSac(stations[n::N],bTime=-1500,eTime =12300,\
    para=para,byRecord=False,isSkip=True,resDir='/media/jiangyr/1TSSD/eventSac/')

for key in R.fvD:
    if '_' not in key:
        continue
    tmp0,tmp1=key.split('_')[-2:]
    KEY = tmp1+'_'+tmp0
    if KEY in R.fvL:
        print(KEY)
fvDAvarageNew ={'models/prem':fvDAverage['models/prem']}
for corr in R.corrL1:
    key = corr.modelFile
    if len(key.split('_'))>=2:
        name0 = key.split('_')[-2]
        name1 = key.split('_')[-1]
        modelName ='%s_%s'%(name0,name1)
        #print(modelName0)
        if modelName in R.fvDAverage:
            if modelName not in fvDAvarageNew:
                fvDAvarageNew[modelName] = fvDAverage[modelName]
'''
trainSetDir='/HOME/jiangyr/trainSet/'
R.fvL = run.loadListStr(trainSetDir+'fvL')
R.fvTrain = run.loadListStr(trainSetDir+'fvTrain')
R.fvTest = run.loadListStr(trainSetDir+'fvTest')
R.fvValid = run.loadListStr(trainSetDir+'fvValid')
#run.d.corrL.saveH5(cL,'/media/jiangyr/1TSSD/trainSet2.h5')
#run.run.train(R,isAverage=False,isRand=True,isShuffle=True)
#R.train(up=5,isRand=True,isShuffle=True)
#run.run.loadCorr(R,isLoad=False)
#R.calResOneByOne()
'''
R.config= run.runConfig(run.paraTrainTest)
R.calResOneByOne()
R.loadCorr()
R.model = None
R.train()
R.loadRes()
R.config.para['resDir']='/fastDir/results/20210412/'
R.config.para['modelFile']= '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_210411-205004_model.h5'
run.d.saveFvD(R.fvAvGet,'/fastDir/results/20210417V2_average/')
run.d.plotFVM(R.fvMGet,R.fvAvGet,resDir='resFig/DS/pait_plot0417/',isDouble=True)
run.d.plotFVL(R.fvAvGet,filename='resFig/DS/pait_plot0417/average.jpg')
R.config.para['minP']=0.7
R.loadRes()
run.d.plotFVM(R.fvMGet,R.fvAvGet,resDir='resFig/DS/pait_plot0417_0.7/',isDouble=True)
run.d.plotFVL(R.fvAvGet,filename='resFig/DS/pait_plot0417_0.7/average.jpg')
R.loadRes()
R.getAreas()
R.areasLimit()
R.config.para['runDir']='../DS/20210408_CEA160_all/'
R.preDS()
from tensorflow.keras.models import load_model
model=load_model('/home/jiangyr/Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_210407-031410_model',compile=False)
R.loadModel()
w = model.get_weights()
R.model.set_weights(w)
R.model.save('/home/jiangyr/Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_210407-031410_model.h5',format='h5')
import tensorflow as tf
tf.keras.backend.clear_session()
reload(run)
run.run.train(R)


'''
R = run.run(run.runConfig(run.paraNorthLagerNew2))
R.preDS(do=False)
R.loadAndPlot()

R = run.run(run.runConfig(run.paraNorthLagerNew2))
#print('aaaa')
R.loadAv(fvDir='/fastDir/results/1030_north/')
R.getAreas()
R.areasLimit()
R.preDS()

R = run.run(run.runConfig(run.paraYNSC))
#print('aaaa')
R.loadAv(fvDir='/home/jiangyr/Pairs_avgpvt_v1_resel/',mode='NEFile')
#R.getAreas()
#R.areasLimit()
R.preDS()
R.preDS(do=False)
R.loadAndPlot()


R2 = run.run(run.runConfig(run.paraYNSCV2))
#print('aaaa')
R2.loadAv(fvDir='/fastDir/results/all/',mode='NEFile')
R.getAreas()
R.areasLimit()
R2.preDS()
R.preDS(do=False)
R2.loadAndPlot()
'''



R3 = run.run(run.runConfig(run.paraAll2))
R3.loadAv(fvDir='/fastDir/results/all/',mode='NEFile')
#R3.preDS()
R3.loadRes()
R3.preDS(do=False)
R3.loadAndPlot()

R3.getAreas()
R3.config.para['areasLimit']=50
R3.areasLimit()

from netCDF4 import Dataset
kea20 = Dataset('models/KEA20.r0.0.nc')
run.d.analyModel(kea20.variables['depth'][:],kea20.variables['vsv'][:])
nameL = ['max','min','mean']
indexL= '123'
for i in range(3):
    name = nameL[i]
    indexStr = indexL[i]
    m=run.d.model('predict/KEA20/vRangeHigh', mode='PSV',getMode = 'fast',layerMode =indexStr,layerN=1000,isFlat=True,R=6371,flatM=-2,pog='p',gpdcExe='/home/jiangyr/program/geopsy/bin/gpdc',doFlat=True,QMul=1)
    f,v=m.calByGpdc(order=0,pog='p',T= run.np.arange(1,300,1).astype(run.np.float))
    f = run.d.fv([f,v],'num')
    f.save('predict/KEA20/'+name)



sigma = run.np.ones(len(R.config.para['T']))
N =len(R.config.para['T'])
N_5=int(N/5)
sigma[:N_5]        =1.5
sigma[N_5:2*N_5]   = 1.75
sigma[2*N_5:3*N_5] = 2.0
sigma[3*N_5:4*N_5] = 2.25
sigma[4*N_5:5*N_5] = 2.5
R.config.sigma=sigma


staL=[]
for key in R.fvDAverage:
    if '_' not in key:
        continue
    if len(R.fvDAverage[key].f)<3:
        continue
    sta0,sta1=key.split('_')
    if sta0 not in staL: 
        staL.append(sta0)
    if sta1 not in staL: 
        staL.append(sta1)



#############
for mul in [1,6,12,18]:
    up=2
    R.config.para['resDir']='%s/%d_%d/'%(resDir,mul,up)
    R.config.para['up']=up
    R.config.para['mul']=mul
    R.model=None
    run.run.loadModelUp(R)
    run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True)
    run.run.calFromCorrL(R)
    run.run.loadRes(R)
    #R.getAv(isCoverQC=True,isDisQC=False)
    R.getAv(isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
    run.run.analyRes(R,format='eps')

for up in [1,3,4]:
    mul=1
    R.config.para['resDir']='%s_%d_%d/'%(resDir[:-1],mul,up)
    R.config.para['up']=up
    R.config.para['mul']=mul
    R.model=None
    run.run.loadModelUp(R)
    run.run.trainMul(R,isAverage=False,isRand=True,isShuffle=True)
    run.run.calFromCorrL(R)
    run.run.loadRes(R)
    #R.getAv(isCoverQC=True,isDisQC=False)
    R.getAv(isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
    run.run.analyRes(R,format='eps')