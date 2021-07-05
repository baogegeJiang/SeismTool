import numpy as np
from matplotlib import pyplot as plt
class pair:
    def __init__(self,L):
        self.L =L
    def getThetaPhi(self,x,lamb):
        L =self.L
        cosTheta = (1/4*(lamb/2)**2+2*L*x-lamb/2*L-x*lamb/2)/(x*L)/2
        if np.abs(cosTheta) >=1:
            return np.pi,2*np.pi*2*x/lamb
        else:
            return np.arccos(cosTheta),np.pi/2
    def calA(self,x,lamb):
        theta,phi = self.getThetaPhi(x,lamb)
        L=self.L
        Q=500
        return 1/((L+x)*(x))**0.5*theta*x*np.cos(phi/2)*np.exp(-x/lamb/Q)*np.exp(-(x+L)/lamb/Q),theta,phi

if __name__ == '__main__':
    P = pair(100)
    colorL = 'kbr'
    lambL = [1,10,20]
    x = 10**(np.arange(0,4,0.01))
    plt.close()
    for i in range(3):
        color = colorL[i]
        lamb  = lambL[i]
        AL = []
        thetaL = []
        phiL   = []
        for X in x:
            A,theta,phi=P.calA(X,lamb)
            AL.append(A)
            thetaL.append(theta)
            phiL.append(phi)
        AL = np.array(AL)
        thetaL = np.array(thetaL)
        phiL = np.array(phiL)
        AC=np.cumsum(AL[:-1]*(x[1:]-x[:-1]))
        plt.subplot(3,1,i+1)
        plt.plot(x,AL/AL.max(),'k')
        plt.plot(x,x*0+0.5,'--k')
        plt.plot(x[:-1],AC/AC.max(),'g')
        #plt.plot(x,thetaL,'b')
        #plt.plot(x,phiL,'r')
    plt.savefig('../hsrRes/AThetaPhi.eps')