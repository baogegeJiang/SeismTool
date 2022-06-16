from SurfDisp.src import disp as disp
from SeismTool.SurfDisp import dispersion as d
from pysurf96 import surf96
import numpy as np
from scipy import signal
from time import time
from matplotlib import pyplot as plt
requires_grad=False
glad = d.GLAD()
'''
thickness = np.array([2,4,8,10,12,13])
vp = np.array([3,4,5,5.5,6,6.1])
vs = vp/1.6
#vs[0]=0
rho = np.array([2,3,4,5,6,6.1])
'''

z0=np.array([0,2.5,7.5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,380,450,600])
vs0=np.array([3.322,3.352,3.374,3.419,3.489,3.569,3.632,3.651,4.125,4.563,4.483,4.416,4.373,4.362,4.374,4.389,4.400,4.427,4.530,4.819,5.258])
vp0=vs0*1.7
rho = vs0/2.0

z0,vp0,vs0,rho = glad(0,85)
#print(vs0[0])
#vp0=vs0*1.7
thickness= z0[1:]-z0[:-1]
dep=(z0[1:]+z0[:-1])/2
vp = (vp0[1:]+vp0[:-1])/2
vs = (vs0[1:]+vs0[:-1])/2
rho = (rho[1:]+rho[:-1])/2
vp = vp[thickness>0.01]
vs = vs[thickness>0.01]
dep = dep[thickness>0.01]
rho = rho[thickness>0.01]
thickness = thickness[thickness>0.01]

if True:
    f = 1/np.arange(5,140,1)
    #c= f*0
    omega = f*np.pi*2
    #sTime=time()
    #v0=surf96(thickness, vp, vs,rho, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
    #vg0=surf96(thickness, vp, vs,rho, 1/f,mode=1, velocity='group', flat_earth=True,wave='rayleigh')
    #print('time',time()-sTime)
    sTime=time()
    #K = np.zeros([5,len(f),len(vp)])
    c=disp.calDisp(thickness[:], vp[:], vs[:],rho[:], 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
    vg=disp.calDisp(thickness[:], vp[:], vs[:],rho[:], 1/f,mode=1, velocity='group', flat_earth=True,wave='rayleigh')
    print(time()-sTime)#KP=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vp')
    #KS=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vs')
    #vg=disp.calDisp(thickness, vp, vs,rho, 1/f,mo
    plt.plot(c,f)
    #plt.plot(vg,f)
    #plt.plot(vs,dep)
    plt.gca().set_yscale('log')
    print(c)
    plt.savefig('predict/test.jpg')
    exit()
cmin=10
la =999
lo=999
for i in range(0,180,5):
    for j in range(0,360,5):
        z0,vp0,vs0,rho = glad(i,j)
        #print(vs0[0])
        #vp0=vs0*1.7
        thickness= z0[1:]-z0[:-1]
        dep=(z0[1:]+z0[:-1])/2
        vp = (vp0[1:]+vp0[:-1])/2
        vs = (vs0[1:]+vs0[:-1])/2
        rho = (rho[1:]+rho[:-1])/2
        vp = vp[thickness>0.01]
        vs = vs[thickness>0.01]
        dep = dep[thickness>0.01]
        rho = rho[thickness>0.01]
        thickness = thickness[thickness>0.01]

        f = 1/np.arange(10,20,3)
        #c= f*0
        omega = f*np.pi*2
        #sTime=time()
        v0=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
        #vg0=surf96(thickness, vp, vs,rho, 1/f,mode=1, velocity='group', flat_earth=True,wave='rayleigh')
        #print('time',time()-sTime)
        #sTime=time()
        #K = np.zeros([5,len(f),len(vp)])
        c=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='phase', flat_earth=True,wave='rayleigh')
        #KP=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vp')
        #KS=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='kernel', flat_earth=True,wave='rayleigh',parameter='vs')
        #vg=disp.calDisp(thickness, vp, vs,rho, 1/f,mode=1, velocity='group', flat_earth=True,wave='rayleigh',domega=0.001,dc0=0.001)
        #print('time',time()-sTime)
        print(i,j,cmin,la,lo)
        if np.abs((v0-c)).max()>1e-4:
            #print('group',np.abs((vg-vg0)).max())
            print('############phase',np.abs((v0-c)).max())
            exit()
        if c[0]<cmin:
            la = i
            lo =j
            cmin=c[0]
            print(cmin,la,lo)

exit()
#print(c-v0,)

plt.figure(figsize=(5,5))
plt.plot(KP[:,5]/thickness,dep[:],'k',label='P:'+str(int(1/f[5])))
plt.plot(KS[:,5]/thickness,dep[:],'b',label='S:'+str(int(1/f[5])))
plt.plot(KP[:,-2]/thickness,dep[:],'r',label='P:'+str(int(1/f[-2])))
plt.plot(KS[:,-2]/thickness,dep[:],'g',label='S:'+str(int(1/f[-2])))
plt.ylim([dep.max(),0])
plt.legend()
plt.savefig('predict/test.jpg',dpi=300)

#print(cyDisp.calDisp(thickness, vp, vs,rho,1/f,c,len(thickness),len(f),isFlat=True,mode=1)-v0,v0[0])

