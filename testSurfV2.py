from SeismTool.SurfDisp.tmp import torchDispV3 as torchDisp
from pysurf96 import surf96
import torch
import numpy as np
from time import time
requires_grad=False
'''
thickness = np.array([2,4,8,10,12,13])
vp = np.array([3,4,5,5.5,6,6.1])
vs = vp/1.6
#vs[0]=0
rho = np.array([2,3,4,5,6,6.1])
'''

z0=np.array([0,2.5,7.5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,280,360,500])
vs0=np.array([3.352,3.352,3.374,3.419,3.489,3.569,3.632,3.651,4.125,4.563,4.483,4.416,4.373,4.362,4.374,4.389,4.400,4.427,4.530,4.719,5.058])
vp0=vs0*1.7
rho = vs0/2.0
thickness= z0[1:]-z0[:-1]
vp = (vp0[1:]+vp0[:-1])/2
vs = (vs0[1:]+vs0[:-1])/2
rho = (rho[1:]+rho[:-1])/2


f = 1/np.arange(5,100,2)
#c= f*0
omega = f*np.pi*2
sTime=time()
#v0=surf96(thickness, vp, vs,rho, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
vg0=surf96(thickness, vp, vs,rho, 1/f,mode=1, velocity='group', flat_earth=True,wave='rayleigh')
print('time',time()-sTime)
sTime=time()
#K = np.zeros([5,len(f),len(vp)])
#c=torchDisp.calDisp(thickness, vp, vs,rho,1/f,dc0=0.005,isR=True,isFlat=True,mode=1)
K=torchDisp.kernel(thickness, vp, vs,rho,1/f,dc0=0.001,isR=True,isFlat=True,mode=1)
vg=torchDisp.group(thickness, vp, vs,rho,1/f,dc0=0.001,isR=True,isFlat=True,mode=1)
print('time',time()-sTime)
#print(c-v0,)
print(vg-vg0)

#print(torchDisp.calDisp(thickness, vp, vs,rho,1/f,c,len(thickness),len(f),isFlat=True,mode=1)-v0,v0[0])

