from imp import reload
from SeismTool.SurfDisp import run
R = run.run(run.runConfig(run.paraNorth))
R.calFromCorr()