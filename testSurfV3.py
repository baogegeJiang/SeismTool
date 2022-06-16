from SurfDisp.src import disp as disp
#from SeismTool.SurfDisp import dispersion as d
from pysurf96 import surf96
import numpy as np
from time import time
from matplotlib import pyplot as plt
requires_grad=False
from scipy import interpolate
from scipy import signal
#litho = d.Litho()
'''
thickness = np.array([2,4,8,10,12,13])
vp = np.array([3,4,5,5.5,6,6.1])
vs = vp/1.6
#vs[0]=0
rho = np.array([2,3,4,5,6,6.1])
'''
def smooth(x,N):
    G = np.exp(-np.arange(-N,N)**2/(N/3)**2)
    G /= G.sum()
    return np.convolve(x,G,'full')[N:-N+1]
z0 = np.array([0,1.25,2.5,5,7.5,8.5,10,12.5,15,17.5,20,22.5,25,27.5,30,35,40,45,50,55,60,65,70,75,80,90,100,110,120,130,140,150,160,180,200,220,240,260,280,300,320,360,380,410,450,480,520,600])
Z=np.array([0,2.5,7.5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,380,450,600])
#VS=np.array([3.2,3.152,3.274,3.519,3.589,3.569,3.732,3.751,3.925,4.063,4.1683,4.216,4.283,4.312,4.574,4.689,4.800,4.927,4.930,4.719,5.258])
VS=np.array([3.322,3.422,3.434,3.459,3.489,3.569,3.632,3.651,4.125,4.263,4.303,4.316,4.343,4.362,4.374,4.389,4.400,4.427,4.530,4.819,5.258])
vs0 = interpolate.interp1d(Z,VS)(z0)
vp0=vs0*1.7
rho = vs0/2.0

#z0,vp0,vs0,rho = litho(35,120)
#vp0=vs0*1.7
thickness= z0[1:]-z0[:-1]
dep=(z0[1:]+z0[:-1])/2
vp = (vp0[1:]+vp0[:-1])/2
vs = (vs0[1:]+vs0[:-1])/2
rho = (rho[1:]+rho[:-1])/2
f = 1/np.arange(5, 160,1)
vso=vs.copy()


#VS=np.array([3.322,3.352,3.374,3.419,3.489,3.569,3.632,3.651,4.125,4.563,4.483,4.416,4.373,4.362,4.374,4.389,4.400,4.427,4.530,4.819,5.258])
VS=np.array([3.1,3.152,3.274,3.519,3.589,3.569,3.732,3.751,3.925,4.063,4.1683,4.216,4.283,4.312,4.574,4.689,4.800,4.927,4.930,5.019,5.258])
vs0 = interpolate.interp1d(Z,VS)(z0)
vp0=vs0*1.7
rho = vs0/2.0

thickness0= z0[1:]-z0[:-1]
dep0=(z0[1:]+z0[:-1])/2
vp0 = (vp0[1:]+vp0[:-1])/2
vs0 = (vs0[1:]+vs0[:-1])/2
rho0 = (rho[1:]+rho[:-1])/2

c=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
c0=disp.calDisp(thickness0, vp0, vs0,rho0, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
KP=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vp')

plt.close()
plt.plot(c,1/f,label='c')
plt.plot(c0,1/f,label='c0')
plt.legend()
plt.savefig('predict/test.jpg',dpi=300)

KS=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vs')
dc = c-c0
g = -(dc.reshape([1,-1])*KS).sum(axis=1)
plt.close()
plt.plot(g,dep,label='g')
plt.plot(vs0-vs,dep,label='dc')
#plt.plot(c0,1/f,label='c0')
plt.legend()
plt.savefig('predict/test.jpg',dpi=300)
#exit()
for i in range(50):
    c=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
    KS=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vs',dc0=0.005)
    KP=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vp',dc0=0.005)
    #KS[np.abs(KS)<0.1*np.abs(KS).max(axis=0,keepdims=True)]=0
    #KP[np.abs(KP)<0.1*np.abs(KP).max(axis=0,keepdims=True)]=0
    dc = c-c0
    g = -(dc.reshape([1,-1])*KS).sum(axis=1)-(dc.reshape([1,-1])*KP).sum(axis=1)/1.7
    #g = signal.savgol_filter(g,13,3)
    g0 = g.copy()
    if i<10:
        g = smooth(g,12)
        a = 0.4
    elif i<20:
        g = smooth(g,8)
        a = 0.2
    else:
        g = smooth(g,3)
        a = 0.1
    gs = g.copy()
    g[g>0.05] = 0.05
    g[g<-0.05] = -0.05
    vs = vs+g*a
    vp = vp+1.7*g*a
plt.close()
plt.subplot(2,1,1)
plt.plot(dc,1/f,label='dc')
plt.plot(gs,dep,label='gs')
plt.plot(g0,dep,label='g0')
plt.plot(vs0-vs,dep,label='dvs')
plt.legend()
plt.subplot(2,1,2)
plt.plot(vs0,dep,label='$vs_{True}$',linewidth=0.5)
plt.plot(vs,dep,label='vs',linewidth=0.5)
plt.plot(vso,dep,label='$vs_o$',linewidth=0.5)
#plt.plot(c0,1/f,label='c0')
plt.legend()
plt.savefig('predict/test.jpg',dpi=300)
print(vs0,vs)
