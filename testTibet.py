from SeismTool.SurfDisp import run
import os
R = run.run(run.runConfig(run.paraTibet))
R.loadDisCover()
if not os.path.exists(R.config.para['matH5']):
    R.calCorrOneByOne()

R.loadModelUp(file=R.config.para['modelFile'])
R.calFromCorr()

R.loadRes(isCoverQC=True)
run.run.getAv(R,isCoverQC=True,isWeight=False,weightType='prob')
run.run.limit(R)
R.preDS(isByTrain=False,do=True,isRun=False,isSameFre=False)
R.preDSSyn(isByTrain=False,do=True,isRun=False,isSameFre=False)