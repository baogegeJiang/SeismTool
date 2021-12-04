import os
import sys
from imp import reload
from SeismTool.SurfDisp import run
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto
R = run.run(run.runConfig(run.paraTrainTest))
run.d.Vav=-1
#R.loadCorr()
#R.saveTrainSet(isMat=True)


#R.plotGetDis()
R.loadCorr(isLoad=True,isLoadFromMat=True,isGetAverage=False,isDisQC=False)#True
R.getDisCover()
#R.plotTrainDis()

R.loadModelUp()
run.run.train(R,isAverage=False,isRand=True,isShuffle=True)
#R.train(up=5,isRand=True,isShuffle=True)
#run.run.loadCorr(R,isLoad=False)
R.calResOneByOne()
#R.loadModelUp(R.config.para['modelFile'])
R.loadModelUp()
R.train(up=5,isRand=True,isShuffle=False)
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

run.d.qcFvD(R.fvAvGet)
run.d.qcFvD(R.fvDGet)
run.d.compareFvD(R.fvAvGet,R.fvDAvarage,1/R.config.para['T'],resDir='predict/compare160/')
reload(run.d)
run.d.plotPair(R.fvAvGet,R.stations)

disL,vL,fL,fvAverage = run.d.outputFvDist(R.fvD0,R.stations,t=R.config.para['T'],keys=R.fvTest,)


reload(run.fcn)
run.fcn.modelUp.show(R.model,R.corrLTest.x[::100],R.corrLTest.y[::100],outputDir='predict/raw/',delta=1,T=R.config.para['T'])
reload(run.d)
run.d.compareFvD(R.fvAvGet,R.fvDAvarage,1/R.config.para['T'],resDir='predict/compareV10/')
run.d.compareFVD(R.fvDAvarage,R.fvAvGet,R.stations,'predict/erroAll.eps',t=R.config.para['T'],keys=[],fStrike=1,title='error_distribution')
run.d.compareFVD(R.fvDAvarage,R.fvAvGet,R.stations,'predict/erroTest.eps',t=R.config.para['T'],keys=R.fvTest,fStrike=1,title='error_distribution',threshold=0.015)
run.d.compareFVD(R.fvD0,R.fvDGet,R.stations,'predict/erroTestSingle.eps',t=R.config.para['T'],keys=R.fvTest,fStrike=1,title='error_distribution')
run.d.compareFVD(R.fvAvGet,R.fvDGet,R.stations,'predict/stdTest.eps',t=R.config.para['T'],keys=R.fvTest,fStrike=1,title='error_distribution')
run.d.compareFVD(R.fvD,R.fvD0,R.stations,'predict/erro0.eps',t=R.config.para['T'],keys=R.fvTest,fStrike=2,title='error_distribution')
run.d.compareFVD(R.fvDGet,R.fvD,R.stations,'predict/erroSingle.eps',t=R.config.para['T'],keys=R.fvTest,fStrike=2,title='error_distribution')
run.d.plotFVM(R.fvMGet,R.fvAvGet,resDir=R.config.para['trainDir']+'pairsTrainTest5/',isDouble=True,fL0=1/R.config.para['T'],stations=R.stations)

M,V0,V1=run.d.compareInF(R.fvDAvarage,R.fvAvGet,R.stations,1/R.config.para['T'],R=R.config.para['R'])
from SeismTool.mathTool import mathFunc
mathFunc.showQC('predict/QC.eps')
reload(run)
#reload(run.d)
#reload(run.seism)
R1=run.run(run.runConfig(run.paraNorth))
R1.model=R.model
R1.calResOneByOne()
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