from SeismTool.SurfDisp import run
from imp import reload

isAll    = False
run.d.Vav=-1
isDisQC =True
isCoverQC = True
R = run.run(run.runConfig(run.paraTibet))
R.loadDisCover()
R.loadRes()
run.run.getAv(R,isCoverQC=isCoverQC,isDisQC=isDisQC,isWeight=False,weightType='prob')
run.run.getAV(R)
run.run.limit(R)
#R.preDS()
reload(run)
R.config=run.runConfig(run.paraTibet)
R.plotGetAvDis()
run.run.preDS(R,isRun=False)
run.run.loadAndPlot(R,maxDeep=12000,FORMAT='jpg')
R.DS.plotHJ(R=R.config.para['R'])