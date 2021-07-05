import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf,erfc
'''
R = 300
L = 32
v = 80
u = 500
theta = 30/180*3.1415
T =L/v-L*np.cos(theta)/u
w = 2*np.pi/T

j = np.arange(-30,50)
Rj = (R**2-2*j*L*R*np.cos(theta)+j**2*L**2)**0.5
dT = (Rj-R)/u + j*L/v-T*j
phi = dT*w

E = np.exp(-1j*phi).sum()
plt.subplot(2,1,1)
plt.plot(j,Rj)
plt.subplot(2,1,2)
plt.plot(j,phi)
print(T,Rj[10]-Rj[11],np.angle(E)/np.pi)
plt.savefig('testB.jpg',dpi=300)
'''
k = 1 
b = 1
T1 = 300
T0 = 100
T2 = 0
N = 400
dt = 0.001
M = 100
n = 400
y = -(np.arange(N+1)/N*b).reshape([-1,1,1])
t = np.arange(1,M+1).reshape([1,-1,1])*dt

DT1 = (T1-T0)
T_01 = T0
DT2 = (T2-T0)
T_02 = 0
B1 =[0]
PM1=[1]
B2 =[b]
PM2=[1]
for i in range(n-1):
    if i%2==0:
        B1.append(2*b-B1[-1])
        PM1.append(-1)
        B2.append(0*b-B2[-1])
        PM2.append(-1)
    else:
        B1.append(0*b-B1[-1])
        PM1.append(1)
        B2.append(2*b-B2[-1])
        PM2.append(1)

B1= np.array(B1).reshape([1,1,-1])
B2= np.array(B2).reshape([1,1,-1])
PM1= np.array(PM1).reshape([1,1,-1])
PM2= np.array(PM2).reshape([1,1,-1])

T = (PM1*erfc(-PM1*(y+B1)/2/np.sqrt(k*t))*DT1+PM2*erfc(PM2*(y+B2)/2/np.sqrt(k*t))*DT2).sum(axis=2)+T_01+T_02
print(T.shape,t.shape,y.shape)
plt.pcolor(t.reshape([-1]),y.reshape([-1]),T,cmap='bwr')
plt.colorbar()
plt.xlabel('t/s')
plt.ylabel('y/m')
plt.savefig('testTemp.jpg',dpi=600)





