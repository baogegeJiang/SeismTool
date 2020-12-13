import os
import sys
from imp import reload
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from SeismTool.SurfDisp import run

R = run.run(run.runConfig(run.paraNorthLagerNew2))
R.preDS(do=False)
R.loadAndPlot()

R = run.run(run.runConfig(run.paraNorthLagerNew2))
#print('aaaa')
R.loadAv(fvDir='/fastDir/results/1030_north/')
R.getAreas()
R.areasLimit()
R.preDS()