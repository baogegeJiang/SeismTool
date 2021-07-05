import numpy as np
from matplotlib import pyplot as plt
import torch
G = 6.754e-11
device='cuda:0'
device='cpu'
class point:
    def __init__(self,x=[0.,0.,0.],v=[0.,0.,0.],m=1.,t=0.,name='0'):
        self.x = np.array(x)
        self.v = np.array(v)
        self.m = np.array(m)
        self.t = np.array(t)
        self.a = self.v*0
        self.name = name
        self.xL = [self.x.copy()]
        self.vL = [self.v.copy()]
        self.tL = [self.t.copy()]
        self.aL = np.zeros([2,3],dtype=np.float64)
        self.pL=[]
    def inserPL(self,pL):
        for p in pL:
            if self.name!=p.name:
                self.pL.append(p)
    def getf(self,self1):
        dx = self1.x - self.x
        r = (dx**2).sum()**0.5
        n = dx/r
        f  = n*G*self.m*self1.m/r**2
        return f
    def getF(self):
        F = self.x*0
        for p in self.pL:
            F += self.getf(p)
        return F
    def updateA(self,index=0):
        self.a = self.getF()/self.m
        self.aL[index]=self.a
    def updateXT(self,dt,isAppend=True):
        self.x += self.v*dt+1/2*self.a*dt**2
        self.t += dt
        if isAppend:
            self.xL.append(self.x.copy())
            self.tL.append(self.t.copy())      
    def updateV(self,dt,iL=range(1),isAppend=True):
        #print(self.v,self.aL[-2],self.aL[-1])
        self.v += self.aL[iL].mean(axis=0)*dt
        if isAppend:
            self.vL.append(self.v.copy())

class solver:
    def __init__(self,pL,d=3):
        self.pL = pL
        self.N  = len(pL)
        self.d  = 3 
        self.x  = torch.zeros([self.N,1,self.d],dtype=torch.float64,device=device)
        self.v  = torch.zeros([self.N,1,self.d],dtype=torch.float64,device=device)
        self.m  = torch.zeros([1,self.N,1],dtype=torch.float64,device=device)
        for i in range(self.N):
            self.x[i,0,:]=torch.tensor(pL[i].x)
            self.v[i,0,:]=torch.tensor(pL[i].v)
            self.m[0,i,0]=torch.tensor(pL[i].m)
        self.xT=self.x.transpose(1,0)
    def getg(self):
        dx = self.x.transpose(1,0)-self.x
        r = (dx**2).sum(axis=2,keepdims=True)**0.5
        nrr = dx/r**3
        nrr[dx==0]=0
        g = (G*self.m*nrr).sum(axis=1,keepdims=True)
        return g
    def update(self,dt):
        g0 = self.getg()
        self.x += self.v*dt+1/2*g0*dt**2
        self.xT[0]=self.x[:,0] 
        g1 = self.getg()
        self.v += (g0+g1)/2*dt
    def append(self):
        for i in range(self.N):
            self.pL[i].xL.append(self.x[i,0,:].cpu().numpy().copy())
            self.pL[i].vL.append(self.v[i,0,:].cpu().numpy().copy())

def getTotalV(pL):
    m = 0
    P = np.array([0.,0.,0.])
    for p in pL:
        m+=p.m
        P+=p.m*p.v
    v = P/m
    return v

if __name__=='__main__':
    m0  = 5.965e24
    wd = np.pi*2/86400
    wy = wd/365
    Au = 149597870700.
    #####152097701000
    pL = [point([0.,0.,0.],v=[0,0.,0.],name='sun',m=1.9891e30),\
        point([0,152097701000.,0.],v=[29291.,0,0],name='earth',m=m0),\
        point([0.,405493000.+152097701000.,0.],v=[ 1023+29291,0.,0],name='moon',m=7.342e22),\
        point([0,0.3871*Au,0.],v=[0.3871*Au*1/87.97*wd,0,0],name='shuixing',m=0.055*m0),\
        point([0,0.7233*Au,0.],v=[0.7233*Au*1/225*wd,0,0],name='jinxing',m=0.857*m0),\
        point([0,1.5237*Au,0.],v=[1.5237*Au*1/687*wd,0,0],name='huoxing',m=0.107*m0),\
        point([0,5.2026*Au,0.],v=[5.2026*Au*1/11.86*wy,0,0],name='muxing',m=317.832*m0),\
        point([0,9.5549*Au,0.],v=[9.5549*Au*1/29.46*wy,0,0],name='tuxing',m=95.16*m0),\
        point([0,19.2184*Au,0.],v=[19.2184*Au*1/84.01*wy,0,0],name='tianwangxing',m=14.54*m0),\
        point([0,30.1104*Au,0.],v=[30.1104*Au*1/164.82*wy,0,0],name='haiwangxing',m=17.15*m0)]
    #for p in pL:
    #    p.inserPL(pL)
    v = getTotalV(pL)
    print(v)
    for p in pL:
        p.v-=v
    s = solver(pL)
    dt = 86400/5
    for i in range(500000):
        if i%10000==0:
            print(i*dt/86400/365,(pL[1].xL[-1]**2).sum()**0.5,(pL[1].vL[-1]**2).sum()**0.5,getTotalV(pL))
        if i%20 == 0:
            s.append()
        s.update(dt)
        '''
        isAppend=False
        if i%10000==0:
            print(i*dt/86400/365,pL[1].x,pL[1].v,getTotalV(pL))
        if i%100 == 0:
            isAppend=True
        for p in pL:
            p.updateA(0)
        for p in pL:
            p.updateXT(dt,isAppend=isAppend)
        for p in pL:
            p.updateA(1)
        for p in pL:
            p.updateV(dt,iL=range(2),isAppend=isAppend)
        '''
    
    xL = [np.array(p.xL) for p in pL]
    strL = 'kkkkkkkkkkk'
    for i in range(len(pL)):
        #plt.close()
        x = xL[i]
        plt.plot(x[:,0],x[:,1],''+strL[i],linewidth=0.1,markersize=0.01)
    plt.gca().set_aspect(1)
    plt.savefig('resFig/p%d.eps'%0,dpi=500)
