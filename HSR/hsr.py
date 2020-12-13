from ..io import seism
import numpy as np
class hsr:
    def __init__(self,f0=3.2,fmax=5,fL0=np.arange(0,100,0.025)):
        self.f0=f0
        self.fmax= fmax
        self.fL0=fL0
    def findFF(self,T3):
        spec,fL=T3.getSpec()
        N=self.fmax/self.f0
        df0=(fL[1]-fL[0])
        df=df0/N
        FFL = np.arange(2.7,3.6,df)
        FF0=-1
        S=-1
        for FF in  FFL:
            FL=np.arange(FF,self.fmax,FF)
            indexL=((FL-fL[0])/df0).astype(np.int)
            s = np.abs(spec)[indexL].sum()
            if s>=S:
                S=s
                FF0=FF
        return FF0
    def adjustSpec(self,T3,f,f0=3.2):
        spec,fL=T3.getSpec()
        fL=fL/f*f0
        df0=(fL[1]-fL[0])
        indexL=((self.fL0-fL[0])/df0).astype(np.int)
        return spec[indexL]
    def getAdd(self,T3):
        specBe = self.fL0*0
        specAf = self.fL0*0
        timeL,vL=T3.getDetec(minValue=30000)
        for time in timeL:
            t3=T3.slice(time-10,time+10)
            if t3.bTime>0:
                FF = self.findFF(t3)
                print(FF)
                if FF>3.2 or FF<3.1:
                    continue
                t3=T3.slice(time-35,time-10)
                if t3.bTime>0:
                    specBe += np.abs(self.adjustSpec(t3,FF))
                t3=T3.slice(time+10,time+35)
                if t3.bTime>0:
                    specAf += np.abs(self.adjustSpec(t3,FF))
        return specBe,specAf

        



        


