import os 
os.environ["CUDA_VISIBLE_DEVICES"] ='1'
from SeismTool.deepLearning import fcn,node_cell
from SeismTool.SurfDisp import run
from imp import reload
#reload(fcn)
#model = fcn.modelSq()

r = run.run(run.runConfig(run.paraAllSq))
r.loadCorr()
r.trainSq()

import os 
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
from SeismTool.deepLearning import fcn,node_cell
from SeismTool.SurfDisp import run
from imp import reload

r = run.run(run.runConfig(run.paraAllDt))
r.loadCorr()
r.loadModelDt()
r.trainDt()
reload(fcn);reload(run);run.fcn.K.clear_session();r.model=None


import os 
os.environ["CUDA_VISIBLE_DEVICES"] ='1'
from SeismTool.deepLearning import fcn,node_cell
from SeismTool.SurfDisp import run
from imp import reload

r = run.run(run.runConfig(run.paraAllO))
r.loadCorr()
r.train()