import os
import sys
from imp import reload
from SeismTool.SurfDisp import run
R = run.run(run.runConfig(run.paraAllONew))
R.calResOneByOne()
R.loadCorr()
R.model = None
R.train()
R.loadRes()
R.config.para['resDir']='/fastDir/results/20210412/'
R.config.para['modelFile']= '/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_210411-205004_model.h5'
run.d.saveFvD(R.fvAvGet,'/fastDir/results/20210417V2_average/')
run.d.plotFVM(R.fvMGet,R.fvAvGet,resDir='resFig/DS/pait_plot0417/',isDouble=True)
run.d.plotFVL(R.fvAvGet,filename=='resFig/DS/pait_plot0417/average.jpg')
R.config.para['minP']=0.7
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