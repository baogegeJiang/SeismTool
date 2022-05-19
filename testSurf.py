from SeismTool.SurfDisp import torchDisp as torchDisp
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


f = torch.tensor(1/np.arange(5,20,1.0),dtype=torch.float32)
omega = f*np.pi*2
model  = torchDisp.Model(thickness,vp,vs,rho,requires_grad=requires_grad)
sTime=time()
v0=surf96(thickness, vp, vs,rho, 1/f.cpu().numpy(),mode=2, velocity='phase', flat_earth=True,wave='rayleigh')
print('time',time()-sTime)
sTime=time()
v = torch.tensor(v0[0],requires_grad=requires_grad,dtype=torch.float32)

#v.requires_grad_(True)
waveNumber = 1/v*omega[0]
print(v)
#for r in np.arange(0.98,1.02,0.01):
#    print([model.dltarR(waveNumber[i]*r,omega[i]) for i in range(len(f))])
#print(model.getsol(1/f[-1],v-0.1,v-1,0.005,v-0.1,v+2,True,True))
i=-1
torch.autograd.set_detect_anomaly(True)

sTime=time()

for i in range(1):
    print(time()-sTime)
    print(model.calDisp(1/f,isFlat=True,mode=2).cpu().numpy()-v0)
    continue
    
    print(v,model.getsol(1/f[0],v-0.01,v-1,0.01,v-0.5,v+2,True,True,isFlat=True))
    continue
    deltaR=model.dltarR(waveNumber,omega[0])
    #deltaR.requires_grad_(True)
    deltaR.backward(retain_graph=True)
    print(deltaR)
    print(model.rho.grad/v.grad)
    print(model.vs.grad/v.grad)
    print(model.vp.grad/v.grad)
    print(v.grad)
print(time()-sTime)
#print([model.dltarR(waveNumber[i]*1.01,omega[i]) for i in range(len(f))])