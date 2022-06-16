from SeismTool.SurfDisp.src.disp import calDisp
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
from SeismTool.SurfDisp import dispersion as d
#from SeismTool.deepLearning import fcn
from tensorflow.keras import  Model
#config=fcn.fcnConfigGather()
#I,O=config.inAndOut()
#model = Model(inputs=I,outputs=O)
#model.compile(loss=config.lossFunc, optimizer='Nadam')

a =  d.Gather(modeN=2)
data,y=a.generate()
plt.close()
plt.figure(figsize=(4,8))
plt.subplot(2,1,1)
plt.pcolor(a.xL,a.timeL,np.sign(data)*np.abs(data)**0.5,cmap='bwr')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.ylim([15,-1])
plt.subplot(2,1,2)
plt.pcolor(a.xL,a.timeL,y[:,:,:].max(axis=2),cmap='hot')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.ylim([15,-1])
plt.savefig('predict/test.jpg',dpi=400)
print(a.NX,a.NT)
exit()


t0 = -2
dt = 1/200
DT = 12
dx = 0.01
DX  = 2
TL = 10**np.arange(0,1.0001,1/10)
fL = 1/TL
modeN=3

NT = int(DT/dt)
timeL = np.arange(NT)*dt+t0
FL = np.fft.fftfreq(NT,dt)[::-1]
FL=FL[FL<=2/1]
FL=FL[FL>=1/20]
NX = int(DX/dx)
xL   = np.arange(NX)*dx

sigmaL = np.array([100,200,300,400,500,600,600,600])/1e3
midL    = np.array([1,1.1,1.2,1,0.8,0.7,0.6,0.5])
aL =  np.array([0.6,0.5,0.5,0.5,0.4,0.4,0.3,0.3])
thicknessL = np.array([0.1,0.2,0.4,0.4,0.4,0.6,0.6,0.4]).reshape([1,-1])*\
    (1+aL.reshape([1,-1])*np.exp(-(xL.reshape([-1,1])-midL.reshape([1,-1]))**2/sigmaL.reshape([1,-1])**2))
vs = np.array([0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.8])
vp = np.array([0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.8])*1.7
rho = vs*2
depL  = np.cumsum(thicknessL,axis=1)-0.5*thicknessL 

plt.close()
for i in range(8):
    plt.plot(xL,depL[:,i])
plt.savefig('predict/test.jpg',dpi=300)

DISPM = np.array([[calDisp(thicknessL[i], vp, vs,rho, 1/FL,mode=j+1, velocity='phase', flat_earth=True,wave='rayleigh') for i in range(NX)]for j in range(modeN)])

DTM = dx/DISPM
TM = np.cumsum(DTM,axis=1)-DTM[:,0:1,:]

dispM = np.array([[calDisp(thicknessL[i], vp, vs,rho, 1/fL,mode=j+1, velocity='phase', flat_earth=True,wave='rayleigh') for i in range(NX)]for j in range(modeN)])

dtM = dx/dispM
tM = np.cumsum(dtM,axis=1)-dtM[:,0:1,:]
modeIndex=0
plt.close()
for i in [0,4,9,20]:
    plt.plot(xL,TM[modeIndex,:,i])
plt.savefig('predict/test.jpg',dpi=300)
#exit()

AL = np.array([1,0.5,0.2])
data= np.zeros([NT,NX])
timeM = timeL.reshape([-1,1])
for i in range(len(FL)):
    F = FL[i]
    af = F**0.5
    for modeIndex in range(modeN):
        DT=TM[modeIndex:modeIndex+1,:,i]
        A = AL[modeIndex]
        pi=np.pi
        data+=A*ne.evaluate('cos(-pi*2*(timeM-DT)*F)*af')

plt.close()
plt.pcolor(xL,timeL,np.sign(data)*np.abs(data)**0.5,cmap='bwr')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.plot(xL,TM[0,:,20])
plt.savefig('predict/test.jpg',dpi=300)
y= np.zeros([NT,NX,len(fL),modeN])
sigma=0.1
for i in range(len(fL)):
    F = fL[i]
    for modeIndex in range(modeN):
        dt=tM[modeIndex:modeIndex+1,:,i]
        pi=np.pi
        y[:,:,i,modeIndex]=A*ne.evaluate('exp(-(timeM-DT)**2/sigma**2)')
plt.close()
plt.pcolor(xL,timeL,y[:,:,0,0],cmap='hot')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.plot(xL,TM[0,:,20])
plt.savefig('predict/test.jpg',dpi=300)


