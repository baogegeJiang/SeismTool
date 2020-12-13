import os
import sys
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from imp import reload
from SeismTool.programs import mineos
reload(mineos)
m = mineos.MINEOS()
m.loadCheck()
m.checking()
M = mineos.Mode(checkfile=m.checkFiles[-1],runPath=m.runPath)
M.reCal()
'''
reload(mineos)
m = mineos.MINEOS()
#m.getNLF()
m.plotNLF()
m.plotNLF(sphtor='tor')
#m.plotFun(0,3,'tor')
nL=[0,0,0,0,0,4,20,12,0,0,0,0,0,4,20,12,3,25,40]
lL=[3,25,40,120,250,2,24,30,3,25,40,120,250,2,24,30,0,0,0]
stL=['tor']*8+['sph']*11
for i in range(17,19):
    m.plotFun(nL[i],lL[i],stL[i])
'''