from SeismTool.SurfDisp import run
from SeismTool.SurfDisp import dispersion as d
R = run.run(run.runConfig(run.paraTrainTest))
R.loadModelUp()