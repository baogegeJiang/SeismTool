import os
import sys
from imp import reload
from SeismTool.SurfDisp import run
R = run.run(run.runConfig(run.paraTrainTest))
R.loadCorr()
R.saveTrainSet(isMat=True)
R.loadCorr(isLoadFromMat=True)
run.run.loadCorr(R,isLoad=False)
R.calResOneByOne()
#R.loadModelUp(R.config.para['modelFile'])
R.loadModelUp()
R.train(up=5,isRand=True,isShuffle=False)
R.model=None
R.loadModelUp()
run.run.train(R,up=5,isRand=False,isShuffle=False)
reload(run)
R.corrL1.reSetUp(5)
run.run.calRes(R)
run.run.loadRes(R)
run.run.getAv(R)
reload(run.d)
disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD,R.stations,t=R.config.para['T'],keys=R.fvTrain)
run.d.plotFvDist(disL,vL,fL,'predict/fvDistTrain.eps')

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD,R.stations,t=R.config.para['T'],keys=R.fvValid)
run.d.plotFvDist(disL,vL,fL,'predict/fvDistValid.eps')

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD,R.stations,t=R.config.para['T'],keys=R.fvTest)
run.d.plotFvDist(disL,vL,fL,'predict/fvDistTest.eps')

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvDAvarage,R.stations,t=R.config.para['T'],keys=R.fvTrain)
run.d.plotFV(vL,fL,'predict/FVTrain.eps',isAverage=True,fvAverage=fvAverage)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvDAvarage,R.stations,t=R.config.para['T'],keys=R.fvValid)
run.d.plotFV(vL,fL,'predict/FVValid.eps',isAverage=True,fvAverage=fvAverage)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvDAvarage,R.stations,t=R.config.para['T'],keys=R.fvTest)
run.d.plotFV(vL,fL,'predict/FVTest.eps',isAverage=True,fvAverage=fvAverage)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD0,R.stations,t=R.config.para['T'],keys=R.fvTrain)
run.d.plotFV(vL,fL,'predict/FVTrainSingle.eps',isAverage=True,fvAverage=fvAverage)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD0,R.stations,t=R.config.para['T'],keys=R.fvValid)
run.d.plotFV(vL,fL,'predict/FVValidSingle.eps',isAverage=True,fvAverage=fvAverage)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD0,R.stations,t=R.config.para['T'],keys=R.fvTest)
run.d.plotFV(vL,fL,'predict/FVTestSingle.eps',isAverage=True,fvAverage=fvAverage)

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
para={\
'delta0' :1,
'freq'   :[-1,-1],#[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e19,
}

R.quakes.cutSac(R.stations,bTime=-1500,eTime =12300,\
    para=para,byRecord=False,isSkip=True,resDir='/HOME/jiangyr/eventSac/')

for key in R.fvD:
    if '_' not in key:
        continue
    tmp0,tmp1=key.split('_')[-2:]
    KEY = tmp1+'_'+tmp0
    if KEY in R.fvL:
        print(KEY)
fvDAvarageNew ={'models/prem':fvDAvarage['models/prem']}
for corr in R.corrL1:
    key = corr.modelFile
    if len(key.split('_'))>=2:
        name0 = key.split('_')[-2]
        name1 = key.split('_')[-1]
        modelName ='%s_%s'%(name0,name1)
        #print(modelName0)
        if modelName in R.fvDAvarage:
            if modelName not in fvDAvarageNew:
                fvDAvarageNew[modelName] = fvDAvarage[modelName]
'''
trainSetDir='/HOME/jiangyr/trainSet/'
R.fvL = run.loadListStr(trainSetDir+'fvL')
R.fvTrain = run.loadListStr(trainSetDir+'fvTrain')
R.fvTest = run.loadListStr(trainSetDir+'fvTest')
R.fvValid = run.loadListStr(trainSetDir+'fvValid')
'''
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