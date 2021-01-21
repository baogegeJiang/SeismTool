import numpy as np
from numba import jit,float32, int64
import scipy.signal  as signal
from scipy.optimize import curve_fit
from .distaz import DistAz
from scipy import interpolate,stats
nptype=np.float32
rad2deg=1/np.pi*180

@jit
def xcorr(a,b):
    la=a.shape[0]
    lb=b.shape[0]
    if len(b.shape)>1:
        c=np.zeros([la-lb+1,b.shape[-1]])
    else:
        c=np.zeros([la-lb+1])
    tb=b[0]*b[0]*0
    for i in range(lb):
        tb+=b[i]*b[i]
    for i in range(la-lb+1):
        ta= (a[i:(i+lb)]*a[i:(i+lb)]).sum(axis=0)
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum(axis=0)
        c[i]=tc/np.sqrt(ta*tb+1e-20)
    return c

@jit
def xcorrSimple(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1)
    for i in range(la-lb+1):
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum()
        c[i]=tc
    return c

@jit
def xcorrEqual(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la)
    tb0=(b*b).sum()
    for i in range(la):
        i1=min(i+lb,la)
        ii1=i1-i
        #print(ii1)
        tc= (a[i:i1]*b[0:ii1]).sum()
        tb=tb0
        if ii1!=lb:
            tb=(b[0:ii1]*b[0:ii1]).sum()
        c[i]=tc/np.sqrt(tb)
    return c
@jit
def corrNP(a,b):
    a=a.astype(nptype)
    b=b.astype(nptype)
    if len(b)==0:
        return a*0+1
    c=np.array([signal.correlate(a[:,i],b[:,i],'valid')for i in range(3)]).transpose()
    tb=(b**2).sum(axis=0)**0.5
    taL=(a**2).cumsum(axis=0)
    ta0=taL[len(b)-1]**0.5
    taL=(taL[len(b):]-taL[:-len(b)])**0.5
    c[1:]=c[1:]/tb/taL
    c[0]=c[0]/tb/ta0
    return c,c.mean(),c.std()

@jit
def getDetec(x, minValue=0.2, minDelta=200):
    indexL = [-10000]
    vL = [-1]
    for i in range(len(x)):
        if x[i] <= minValue:
            continue
        if i > indexL[-1]+minDelta:
            vL.append(x[i])
            indexL.append(i)
            continue
        if x[i] > vL[-1]:
            vL[-1] = x[i]
            indexL[-1] = i
    if vL[0] == -1:
        indexL = indexL[1:]
        vL = vL[1:]
    return np.array(indexL), np.array(vL)

def matTime2UTC(matTime,time0=719529):
    return (matTime-time0)*86400

def UTC2MatTime(time,time0=719529):
    return time/86400+time0

@jit(int64(float32[:],float32,int64,int64,float32[:]))
def cmax(a,tmin,winL,laout,aM):
    i=0 
    while i<laout:
        if a[i]>tmin:
            j=0
            while j<min(winL,i):
                if a[i]>a[i-j]:
                    a[i-j]=a[i]
                j+=1
        if i>=winL:
            aM[i-winL]+=a[i-winL]
        i+=1
    while i<laout+winL:
        aM[i-winL]+=a[i-winL]
        i+=1
    return i

def cmax_bak(a,tmin,winL,laout,aM):
    i=0 
    indexL=np.where(a>tmin)[0]
    for i in indexL:
        a[max(i-winL,0):i]=np.fmax(a[max(i-winL,0):i],a[i])
    aM[:laout]+=a[:laout]

def CEPS(x):
    #sx=fft(x);%abs(fft(x)).^2;
    #logs=log(sx);
    #y=abs(fft(logs(1:end)));
    spec=np.fft.fft(x)
    logspec=np.log(spec*np.conj(spec))
    y=abs(np.fft.ifft(logspec))
    return y
def cart2pol(x,y):
    r=x+y*1j
    return np.abs(r),np.angle(r)
def pol2cart(R,theta):
    r=R*np.exp(theta*1j)
    return np.real(r),np.imag(r)
def angleBetween(x0,y0,x1,y1):
    X=x0*x1+y0*y1
    X=X/((x0**2+y0**2)*(x1**2+y1**2))**0.5
    Y=x0*y1-y0*x1
    Y=Y/((x0**2+y0**2)*(x1**2+y1**2))**0.5
    r,theta=cart2pol(X,Y)
    #print(X[np.where(X<0)[0]])
    return theta
    

class  R:
    """docstring for  R"""
    def __init__(self, pL):
        super( R, self).__init__()
        if pL.shape[0]==2:
            x0=pL[0,0]
            y0=pL[0,1]
            x1=pL[1,0]
            y1=pL[1,1]
            xL=np.array([x0,x1,x1,x0])
            yL=np.array([y0,y0,y1,y1])
        elif pL.shape[0]==3:
            x0=pL[0,0]
            y0=pL[0,1]
            x1=pL[1,0]
            y1=pL[1,1]
            x2=pL[2,0]
            y2=pL[2,1]
            x3=x2+x0-x1
            y3=y2+y0-y1
            xL=np.array([x0,x1,x2,x3,x0]).reshape(-1,1)
            yL=np.array([y0,y1,y2,y3,y0]).reshape(-1,1)
        self.xyL=np.concatenate([xL,yL],axis=1)
    def isIn(self,p):
        x0=p[0]
        y0=p[1]
        dxL=self.xyL[:,0]-x0
        dyL=self.xyL[:,1]-y0
        dxL=np.append(dxL,dxL[0])
        dyL=np.append(dyL,dyL[0])
        thetaL=angleBetween(dxL[:-1],dyL[:-1],\
            dxL[1:],dyL[1:])
        #print(dxL,dyL,thetaL/np.pi*180,np.abs(thetaL.sum()/np.pi-2))
        if np.abs(thetaL.sum()/np.pi-2)<0.2:
            return True
        else:
            return False
    def perKm(self):
        p0 = self.xyL[0]
        p1 = self.xyL[1]
        dLa,dLo= np.abs(p0-p1)
        dist =  DistAz(p0[0],p0[1],p1[0],p1[1]).getDelta()* 111.19
        return dLa/dist,dLo/dist

def prob2color(prob,color0=np.array([1,1,1])*0.8):
    # blue for no prob; gray for differetn p(>0.5);red for p(<0.5)
    # green for not detect phase
    if prob > 1:
        return np.array([0,1,0])
    elif prob >0.5:
        pN = (1-prob)*2
        return color0*pN
    elif prob > -1:
        return np.array([1,0,0])
    else:
        return np.array([0,0,1])

def gaussian(x,A, t0, sigma):
    return A*np.exp(-(x - t0)**2 / sigma**2)
def fitexp(y):
    N = len(y)
    x = np.arange(N)
    ATS,pcov = curve_fit(gaussian,x,y,p0=[1,N/2,1.5],\
        bounds=(0.1, [3, N, 8]),maxfev=40)
    A = ATS[0]
    t0 = ATS[1]
    sigma = ATS[2]
    #print(pcov)
    return t0
def findPos(y, moreN = 10):
    yPos  = y.argmax( axis=1).astype(np.float32)
    yMax = y.max(axis=1)
    for i in range(y.shape[0]):
        for j in range(y.shape[-1]):
            pos0 = int(yPos[i,0,j])
            max0 = yMax[i,0,j]
            if max0 > 0.5 and pos0>=moreN and pos0+moreN<y.shape[1] :
                try:
                    pos =  fitexp(y[i,pos0-moreN:pos0+moreN,0,j])+pos0-moreN
                except:
                    pass
                else:
                    if np.abs(pos-pos0)<0.5:
                        yPos[i,0,j]=pos
    return yPos, yMax

class  Line(R):
    """docstring for  R"""
    def __init__(self,pL,H,name=''):
        if not isinstance(pL,np.ndarray):
            pL=np.array(pL)
        self.H=H
        self.xyL = pL
        self.mid=pL.mean(axis=0)
        self.dLa =  DistAz(self.mid[0]-0.5,self.mid[1],self.mid[0]+0.5,self.mid[1]).getDelta()* 111.19
        self.dLo =  DistAz(self.mid[0],self.mid[1]-0.5,self.mid[0],self.mid[1]+0.5).getDelta()* 111.19
        self.dLaLo=np.array([self.dLa,self.dLo])
        self.XYL=self(self.xyL)
        self.n= self.XYL[1]/(self.XYL[1]**2).sum()**0.5
        self.v= np.array([-self.n[1],self.n[0]])
        self.L= self.l(self.xyL[1])
        self.name =name
    def __call__(self,p):
        if not isinstance(p,np.ndarray):
            p=np.array(p)
        if len(p.shape)==1:
            return self.dLaLo*(p-self.xyL[0])
        else :
            return self.dLaLo.reshape([1,-1])*(p-self.xyL[0].reshape([1,-1]))
    def l(self,p):
        xyL = self(p)
        if len(xyL.shape)==1:
            return (xyL*self.n).sum()
        else :
            return (self.n.reshape([1,-1])*xyL).sum(axis=1)
        return
    def l(self,p):
        xyL = self(p)
        if len(xyL.shape)==1:
            return (xyL*self.n).sum()
        else :
            return (self.n.reshape([1,-1])*xyL).sum(axis=1)
        return
    def h(self,p):
        xyL = self(p)
        if len(xyL.shape)==1:
            return (xyL*self.v).sum()
        else :
            return (self.v.reshape([1,-1])*xyL).sum(axis=1)
        return 
    def isIn(self,p):
        if np.abs(self.h(p))>self.H:
            return False
        if self.l(p)<0 or self.l(p)>self.L:
            return False
        return True
    def perKm(self):
        return 1
class Round:
    def __init__(self,p0,R):
        self.p0=p0
        self.R=R
    def isIn(self,p):
        ##print(p)
        return DistAz(self.p0[0],self.p0[1],p[0],p[1]).getDelta()*110.19<=self.R
    def isIN(self,pL):
        return [self.isIn(p) for p in pL ]
class Model:
    def __init__(self,config,mode,la,lo,z,v):
        self.mode = mode
        self.config=config
        self.nxyz = [len(la),len(lo),len(z)]
        self.z  =  z#.reshape([-1,1,1])
        self.la = la#.reshape([1,-1,1])
        self.lo = lo#.reshape([1,1,-1])
        self.v  = v
    def __call__(self,la,lo,z):
        i0 = np.abs(self.la - la).argmin()
        i1 = np.abs(self.lo - lo).argmin()
        i2 = np.abs(self.z  - z).argmin()
        v = self.v[i0,i1,i2]
        return v 
    def output(self,la,lo,z,interp=True):
        nxyz    = [len(la),len(lo),len(z)]
        nxyzTmp = [len(la),len(lo),len(self.z)]
        if interp == False:
            return self.v
        v       = np.zeros(nxyz)
        vTmp    = np.zeros(nxyzTmp)
        #print(z)
        V= self.v
        V[np.isnan(V)]=-1e9
        for i in range(nxyzTmp[-1]):
            vTmp[:,:,i] = interpolate.interp2d(self.lo, self.la, V[:,:,i],bounds_error=False,fill_value=1e-8,kind='linear')(lo,la)
        if la[-1]<la[0]:
            vTmp = vTmp[::-1]
        if lo[-1]<lo[0]:
            vTmp = vTmp[:,::-1]
        for i in range(nxyz[0]):
            for j in range(nxyz[1]): 
                v[i,j,:] = interpolate.interp1d(self.z,vTmp[i,j])(z)
        v[v<0]=np.nan
        return v
    def OutputGriddata(self,la,lo,z,isPer=False,vR='',P2='',maxH=300):
        V = self.v.copy()
        if vR !='':
            out  = outR(vR,self.la,self.lo)
        if isPer:
            for i in range(V.shape[-1]):
                v= V[:,:,i]
                if vR!='':
                    v[out]=np.nan
                V[:,:,i]/=v[np.isnan(v)==False].mean()
            V-=1
        V = V.reshape([-1])
        Lo,La,Z = np.meshgrid(self.lo,self.la,self.z)
        vaild = (np.isnan(V)==False)
        V = V.reshape([-1])[vaild]
        Lo,La,Z=[Lo.reshape([-1])[vaild],La.reshape([-1])[vaild],Z.reshape([-1])[vaild]]
        if P2!='':
            laLo=np.array([La.tolist(),Lo.tolist()]).transpose()
            h = P2.h(laLo)
            #print(h)
            vaild = (np.abs(h)<maxH)
            #V = V[vaild]
            V = V[vaild]
            Lo,La,Z=[Lo[vaild],La[vaild],Z[vaild]]
            laLo=np.array([La.tolist(),Lo.tolist()]).transpose()
            l = P2.l(laLo)
            vaild = (l>-300)*(l<P2.L+300)
            V = V[vaild]
            Lo,La,Z=[Lo[vaild],La[vaild],Z[vaild]]
        points = np.concatenate((Lo.reshape([-1,1]),La.reshape([-1,1]),Z.reshape([-1,1])),axis=1)
        v=interpolate.griddata(points,V,(lo,la,z),method='linear')
        #v[v<0]=np.nan
        isnan = isNan3D(self.la,self.lo,self.z,self.v,la,lo,z)
        v[isnan==1]=np.nan
        return v
    def Output(self,la,lo,z,isPer=False,vR=''):
        #Lo,La,Z = np.meshgrid(self.lo,self.la,self.z)
        V = self.v.copy()
        if vR !='':
            out  = outR(vR,self.la,self.lo)
        if isPer:
            for i in range(V.shape[-1]):
                v= V[:,:,i]
                if vR!='':
                    v[out]=np.nan
                V[:,:,i]/=v[np.isnan(v)==False].mean()
            V-=1
        V[np.isnan(V)]=-1e20
        shape = list(la.shape)
        shape.append(1)
        points = np.concatenate((la.reshape(shape),lo.reshape(shape),z.reshape(shape)),axis=-1)
        laIndex = self.la.argsort()
        v=interpolate.interpn((self.la[laIndex],self.lo,self.z),V[laIndex],points,method='linear')
        v[v<-10]=np.nan
        return v
    def Output2D(self,la,lo,V,isPer=False,vR=''):
        Lo,La = np.meshgrid(self.lo,self.la)
        V0=V
        V = V.reshape([-1])
        #V[np.isnan(V)]=-1e9
        vaild = (np.isnan(V)==False)
        points = np.concatenate((Lo.reshape([-1,1])[vaild],La.reshape([-1,1])[vaild]\
            ),axis=1)
        if vaild.sum()==0:
            return None
        v=interpolate.griddata(points,V[vaild],(lo,la),method='cubic')
        isnan = isNan(self.la,self.lo,V0,la,lo)
        v[isnan==1]=np.nan
        #v[v<0]=np.nan
        return v
    def denseLaLo(self,Per,N=300,dIndex=0,doDense=True):
        #mean=Per[np.isnan(Per)==False].mean()
        if not doDense:
            print('no Dense')
            return self.la[dIndex:-dIndex],self.lo[dIndex:-dIndex],Per[dIndex:-dIndex,dIndex:-dIndex]
        Per[np.isnan(Per)]=-5e20
        dLa = (self.la[-dIndex]-self.la[dIndex])/N
        dLo = (self.lo[-dIndex]-self.lo[dIndex])/N
        la  = np.arange(self.la[dIndex],self.la[-dIndex],dLa)
        la.sort()
        lo  = np.arange(self.lo[dIndex],self.lo[-dIndex],dLo)
        per = interpolate.interp2d(self.lo, self.la, Per,kind='linear')(lo,la)
        per[per<-2]=np.nan
        la,lo=np.meshgrid(la,lo)
        return la, lo, per
    def denseLaLoGrid(self,Per,N=300,dIndex=0,doDense=True):
        if not doDense:
            print('no Dense')
            return self.la[dIndex:-dIndex],self.lo[dIndex:-dIndex],Per[dIndex:-dIndex,dIndex:-dIndex]
        #mean=Per[np.isnan(Per)==False].mean()
        #Per[np.isnan(Per)]=mean
        dLa = (self.la[-dIndex]-self.la[dIndex])/N
        dLo = (self.lo[-dIndex]-self.lo[dIndex])/N
        la  = np.arange(self.la[dIndex],self.la[-dIndex],dLa)
        la.sort()
        lo  = np.arange(self.lo[dIndex],self.lo[-dIndex],dLo)
        la,lo=np.meshgrid(la,lo)
        return la,lo,self.Output2D(la,lo,Per)
    def outputP2(self,P2,N=100,isPer=False,line=''):
        La = P2[0][0]+(P2[1][0]-P2[0][0])/N*np.arange(N+1)
        Lo = P2[0][1]+(P2[1][1]-P2[0][1])/N*np.arange(N+1)
        dist= DistAz(P2[0][0],P2[0][1],P2[1][0],P2[1][1]).getDelta()* 111.19
        Dist = np.arange(N)/N*dist
        if len(P2[0])==3:
            Z = P2[0][2]+(P2[1][2]-P2[0][2])/N*np.arange(N+1)
        else:
            Z = self.z.min()+(self.z.max()-self.z.min())/N*np.arange(N)
        la = La.reshape([1,-1])+Z.reshape([-1,1])*0
        lo = Lo.reshape([1,-1])+Z.reshape([-1,1])*0
        z  =  La.reshape([1,-1])*0+Z.reshape([-1,1])
        V= self.OutputGriddata(la,lo,z,isPer=isPer,P2=line)
        #print(V.shape)
        if isPer and False:
            for i in range(z.shape[0]):
                V[i]/=V[i,np.isnan(V[i])==False].mean()
            V-=1
        return la,lo,z,Dist,V

def isNan(La,Lo,V,la,lo):
    shape=la.shape
    La=La.reshape([-1,1])
    Lo=Lo.reshape([-1,1])
    la=la.reshape([1,-1])
    lo=lo.reshape([1,-1])
    i = np.abs(La-la).argmin(axis=0)
    j = np.abs(Lo-lo).argmin(axis=0)
    #print(i,j)
    isnan = i*0
    for I in range(len(i)):
        if np.isnan(V[i[I],j[I]]):
            isnan[I]=1
            #print(I,i[I],j[I])
    return isnan.reshape(shape)
def isNan3D(La,Lo,Z,V,la,lo,z):
    shape=la.shape
    La=La.reshape([-1,1])
    Lo=Lo.reshape([-1,1])
    Z=Z.reshape([-1,1])
    la=la.reshape([1,-1])
    lo=lo.reshape([1,-1])
    z=z.reshape([1,-1])
    i = np.abs(La-la).argmin(axis=0)
    j = np.abs(Lo-lo).argmin(axis=0)
    k = np.abs(Z-z).argmin(axis=0)
    #print(i,j)
    isnan = i*0
    for I in range(len(i)):
        if np.isnan(V[i[I],j[I],k[I]]):
            isnan[I]=1
            #print(I,i[I],j[I])
    return isnan.reshape(shape)
def outR(vR,la,lo):
    lo,la=np.meshgrid(lo,la)
    print(lo,la)
    #print(np.conca(la,lo).shape)
    lalo=np.concatenate((la.reshape([1,la.shape[0],la.shape[1]]),\
            lo.reshape([1,la.shape[0],la.shape[1]])),axis=0).transpose([1,2,0])
    dlalo = lalo.reshape([lalo.shape[0],lalo.shape[1],1,2])- vR.reshape([1,1,-1,2])
    dlalo2 = np.concatenate((dlalo[:,:,:-1].reshape([1,dlalo.shape[0],dlalo.shape[1],\
            dlalo.shape[2]-1,dlalo.shape[3]])\
            ,dlalo[:,:,1:].reshape([1,dlalo.shape[0],dlalo.shape[1],\
            dlalo.shape[2]-1,dlalo.shape[3]]))\
        ,axis=0).transpose([0,4,1,2,3])
    print(dlalo2.shape,dlalo.shape)
    theta = np.arcsin((dlalo2[0,0]*dlalo2[1,1]-dlalo2[0,1]*dlalo2[1,0])\
    /((dlalo2[0]**2).sum(axis=0)*(dlalo2[1]**2).sum(axis=0))**0.5)
    sumTheta=theta.sum(axis=2)
    print(sumTheta)
    return np.abs(sumTheta)<np.pi/3

#calc_b is from Zhou Yijian
def gr_fit(mag, min_num=100, method='MAXC'):
    if len(mag) < min_num: return np.nan, [np.nan, np.nan], np.nan
    mag = np.array(mag)
    if method=='MAXC': mc = calc_mc_maxc(mag)
    mag = mag[mag>=mc]
    b_val, b_dev = calc_b(mag)
    return mc, [b_val, b_dev], np.log10(len(mag))


# calc b value
def calc_b(mag, min_num=None):
    num_events = len(mag)
    if min_num: 
        if num_events < min_num: return -1, -1
    b_val = np.log10(np.exp(1)) / (np.mean(mag) - np.min(mag) + 0.05)
    b_dev = 2.3 * b_val**2 * (np.var(mag) / num_events)**0.5
    return round(b_val,2), round(b_dev,2)


# calc fmd
def calc_fmd(mag):
    mag = mag[mag!=-np.inf]
    mag_max = np.ceil(10 * max(mag)) / 10
    mag_min = np.floor(10 * min(mag)) / 10
    mag_bin = np.around(np.arange(mag_min-0.1, mag_max+0.2, 0.1),1)
    num = np.histogram(mag, mag_bin)[0]
    cum_num = np.cumsum(num[::-1])[::-1]
    return mag_bin[1:], num, cum_num


# calc Mc by MAXC method
def calc_mc_maxc(mag):
    mag_bin, num, _ = calc_fmd(mag)
    return mag_bin[np.argmax(num)]


# calc b_val to Mc relation
def calc_b2mc(mag):
    mag_min = np.floor(10 * min(mag)) / 10
    mag_max = np.ceil(10 * max(mag)) / 10
    mc_min = calc_mc_maxc(mag)
    mag_rng = np.arange(mc_min-1., mc_min+1.5, 0.1)
    b_val_dev = np.array([calc_b(mag[mag>mi]) for mi in mag_rng])
    return b_val_dev, mag_rng


def devide(rM,pL,paraL):
    return [[ paraL[np.array(r.isIN(pL))] for r in rL]for rL in rM]

def calc_B(mag, min_num=1000,min_mag=-2,max_mag=5):
    mag=np.array(mag)
    mag=mag[mag>min_mag]
    mag=mag[mag<max_mag]
    if len(mag)<min_num:
        return [np.nan,np.nan]
    
    mc, [b_val, b_dev], a_val = gr_fit(mag, min_num=min_num)
    print(b_val,b_dev)
    if mc<0:
        return [np.nan,np.nan]
    else:
        return [b_val,mc]