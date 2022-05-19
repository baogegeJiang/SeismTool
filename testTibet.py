from SeismTool.SurfDisp import run
import os
R = run.run(run.runConfig(run.paraTibet))
R.loadDisCover()
if False:
    if not os.path.exists(R.config.para['matH5']):
        R.calCorrOneByOne()
    R.loadModelUp(file=R.config.para['modelFile'])
    R.calFromCorr()

R.loadRes(isCoverQC=True)
run.run.getAv(R,isCoverQC=True,isWeight=False,weightType='prob')
R.getAV()
keyL = list(R.fvAvGet.keys())[::200]

run.run.limit(R)
resDir = 'predict/'+R.config.para['resDir'].split('/')[-2]+'/'
run.d.plotFVM(R.fvMGet,R.fvAvGet,R.fvAvGet,resDir=resDir+'/compare/',isDouble=True,fL0=1/R.config.para['T'],format='.jpg',stations=R.stations,keyL=keyL,threshold=0.01,fvMRef=R.fvAvGet)
    if False:
        R.preDS(isByTrain=False,do=True,isRun=False)
        R.preDSSyn(isByTrain=False,do=True,isRun=False)

R.preDS(isByTrain=False,do=False,isRun=False)
R.preDSSyn(isByTrain=False,do=False,isRun=False)
R.loadAndPlot(R.DS,isPlot=True,maxDeep=1e9)
R.DS.plotHJ(R=R.config.para['R'],maxDeep=1e9)

