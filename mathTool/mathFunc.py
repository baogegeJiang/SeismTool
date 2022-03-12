#from ..io.seism import NoneType
from re import S
import numpy as np
from numba import jit,float32, int64
import scipy.signal  as signal
from scipy import fftpack
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib import pyplot as plt
import torch

from scipy.stats.stats import _threshold_mgc_map
nptype=np.float32
rad2deg=1/np.pi*180

@jit
def xcorr(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1)
    tb=0
    for i in range(lb):
        tb+=b[i]*b[i]
    for i in range(la-lb+1):
        ta=0
        tc=0
        ta= (a[i:(i+lb)]*a[i:(i+lb)]).sum()
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum()
        if ta!=0 and tb!=0:
            c[i]=tc/np.sqrt(ta*tb)
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

def xcorrAndDe(a,b):
    la = a.size
    lb = b.size
    lab = min(la,lb)
    if la!=lab or lb != lab:
        a = a[:lab]
        b = b[:lab]
    A = np.fft.fft(a) 
    B = np.fft.fft(b)
    c0 =  signal.correlate(a,b,'full')[lab-1:]
    absB = np.abs(B)
    threshold = absB.max()*(1e-13)
    #c0 = np.real(xcorrFrom0(a,b))
    #np.real(np.fft.ifft(np.conj(B)*A))
    C1 = A.copy()
    C1[absB>threshold]/=C1[absB>threshold]/B[absB>threshold]
    c1 = np.real(np.fft.ifft(C1))
    c0 /= c0.max()
    c1 /= c1.max()
    return c0+c1*1j

def xcorrAndDeV2(a,b):
    la = a.size
    lb = b.size
    lab = min(la,lb)
    if la!=lab or lb != lab:
        a = a[:lab]
        b = b[:lab]
    A = np.fft.fft(a) 
    B = np.fft.fft(b)
    labMid = int(lab/2)
    #B[labMid:]= np.conj(B[labMid:])
    c0 =  signal.correlate(a,b,'full')[lab-1:]
    absB = np.abs(B)
    threshold = absB.max()*(1e-13)
    #c0 = np.real(xcorrFrom0(a,b))
    #np.real(np.fft.ifft(np.conj(B)*A))
    C1 = A.copy()
    C1[absB>threshold]/=B[absB>threshold]
    C1[absB<=threshold] = 0
    c1 = np.real(np.fft.ifft(C1))
    c0 /= c0.max()
    c1 /= c1.max()
    return c0+c1*1j
def xcorrAndDeV3(a,b):
    la = a.size
    lb = b.size
    lab = min(la,lb)
    if la!=lab or lb != lab:
        a = a[:lab]
        b = b[:lab]
    #a = fftpack.hilbert(a)*1j+a
    #b = fftpack.hilbert(b)*1j+b
    A = np.fft.fft(a) 
    B = np.fft.fft(b)
    absA = np.abs(A)
    absB = np.abs(B)
    thresholdA = absA.std()*(1e-1)
    thresholdB = absB.std()*(1e-1)
    C1=np.conj(B)*A
    #c0 = np.real(xcorrFrom0(a,b))
    #np.real(np.fft.ifft(np.conj(B)*A))
    #C1 = A.copy()
    #C1/=B
    C1[absA<=thresholdA] = 0
    C1[absB<=thresholdB] = 0
    return np.fft.ifft(C1)

def xcorrFrom0(a,b):
    la = a.size
    lb = b.size
    x =  signal.correlate(a,b,'full')
    return x[lb-1:]

def xcorrAndConv(a,b):
    la = a.size
    lb = b.size
    x0 =  signal.correlate(a,b,'full')
    x1 =  signal.convolve(b,a,'full')
    return x0[lb-1:]+1j*x1[lb-1:]


@jit
def xcorrComplex(a,b):
    a = fftpack.hilbert(a)*1j+a
    b = fftpack.hilbert(b)*1j+b
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1).astype(np.complex)
    for i in range(la-lb+1):
        tc= (a[i:(i+lb)]*b[0:(0+lb)].conj()).sum()
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
    return signal.correlate()

def corrNP(a,b):
    a=a.astype(nptype)
    b=b.astype(nptype)
    if len(b)==0:
        return a*0+1
    c=signal.correlate(a,b,'valid')
    tb=(b**2).sum()**0.5
    taL=(a**2).cumsum()
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
    return 1

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

def flat(z,vp,vs,rho,m=-2,R=6371):
    z = np.array(z)
    zmid = z.mean()
    miu  = vs**2*rho
    lamb = vp**2*rho-2*miu
    r = R-zmid
    zNew = R*np.log(R/(R-z))
    lambNew =  ((r/R)**(m-1))*lamb
    miuNew  =  ((r/R)**(m-1))*miu
    rhoNew  =  ((r/R)**(m+1))*rho
    vpNew   =  ((lambNew+2*miuNew)/rhoNew)**0.5
    vsNew   =  (miuNew/rhoNew)**0.5
    return zNew,vpNew,vsNew,rhoNew

@jit
def validL(v,prob, minProb = 0.7,minV=2,maxV=6):
    l    = []
    tmp  = []
    for i in range(len(v)):
        if v[i] > minV and v[i]<maxV and\
         prob[i]>minProb and (i==0 or np.abs(prob[i]-prob[i-1])/prob[i]<0.2):
            tmp.append(i)
            if i == len(v)-1:
                l.append(tmp)
            continue
        elif len(tmp)>0:
            l.append(tmp)
            tmp=[]
    return l


def randomSource(i,duraCount,data):
    if i==0:
        data[:duraCount] += 1
        data[:duraCount] += np.random.rand()*0.3*np.random.rand(duraCount)
    if i ==1:
        mid = int(duraCount/2)
        data[:mid] = np.arange(mid)
        data[mid:2*mid] = np.arange(mid-1,-1,-1)
        data[:duraCount] += np.random.rand()*0.3*np.random.rand(duraCount)*mid
    if i==2:                
        rise = 0.1+0.3*np.random.rand()
        mid = int(duraCount/2)
        i0 = int(duraCount*rise)
        data[:duraCount] += i0
        data[:i0] = np.arange(i0)
        data[duraCount-i0:duraCount] = np.arange(i0-1,-1,-1)
        data[:duraCount] += np.random.rand()*0.3*np.random.rand(duraCount)*i0
    if i ==3:
        T  = np.random.rand()*60+5
        T0 = np.random.rand()*2*np.pi
        data[:duraCount] = np.sin(np.arange(duraCount)/T*2*np.pi+T0)+1
        data[:duraCount] += (np.random.rand(duraCount)-0.5)*0.1
        data[:duraCount] *= np.random.rand(duraCount)+4
    if i == 4:
        T  = (np.random.rand()**3)*100+5
        T0 = np.random.rand()*2*np.pi
        data[:duraCount] = np.sin(np.arange(duraCount)/T*2*np.pi+T0)
        data[:duraCount] += (np.random.rand(duraCount)-0.5)*0.1
        data[:duraCount] *= np.random.rand(duraCount)+2
@jit
def gaussian(x,A, t0, sigma):
    return A*np.exp(-(x - t0)**2 / sigma**2)
@jit
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

def disDegree(dis,maxD = 100, maxTheta=20):
    delta = dis/111.19
    theta0 = maxD/111.19
    theta = theta0/np.sin(delta/180*np.pi)
    return min(theta,maxTheta)

def disDegreeBak(dis,maxD = 100, maxTheta=20):
    delta = dis/111.19
    if delta >90:
        delta = 90
    theta0 = maxD/111.19
    theta = theta0/np.sin(delta/180*np.pi)
    return min(theta,maxTheta)

def QC_bak(data,threshold=2.5):
    if len(data)<6:
        return data.mean(),999,len(data)
    #if len(data)<10:
    #    return data.mean(),data.std(),len(data)
    mData = np.median(data)
    d = np.abs(data - mData)
    lqr = stats.iqr(data)
    Threshold = lqr*threshold
    if (d>Threshold).sum()==0:
        return data.mean(),data.std(),len(data)
    else:
        return QC(data[d<Threshold],threshold)

def wMean(data,wL):
    return (data*wL).sum()/wL.sum()
def wStd(data,wL):
    m = wMean(data,wL)
    D2 = (data-m)**2
    std2 = wMean(D2,wL) 
    return std2**0.5
def iqrWeight(data,wL,mid):
    WS = np.cumsum(wL)
    w0 = mid[0]/100*WS[-1]
    w1 = mid[1]/100*WS[-1]
    #print(w0,w1)
    i0 =np.abs(WS-w0).argmin()
    i1 =np.abs(WS-w1).argmin()
    return data[i1]-data[i0],wMean(data[i0:i1+1],wL[i0:i1+1])
def QC(data,threshold=0.82285,it=3,minThreshold=0.02,minSta=5,resultL=None,mid=[25.,75.],wL=[],isSorted=False):
    if len(wL)==0:
        wL = data*0+1
    if not isSorted:
        iL = data.argsort()
        data = data[iL]
        wL   = wL[iL]
    mul = calNSigma(1,threshold)/calNSigma(1,(mid[1]-mid[0])/100)
    if len(data)<minSta:
        return wMean(data,wL),999,len(data)
    if  it==0:
        return wMean(data,wL),data.std(),len(data)
    #if len(data)<10:
    #    return data.mean(),data.std(),len(data)
    lqr,mData = iqrWeight(data,wL,mid)
    #print(lqr,stats.iqr(data))
    lqrHalf = lqr/2
    #mData  = data0/2+data1/2
    Threshold = max(lqrHalf*mul,mData*minThreshold)
    d = np.abs(data - mData)
    if isinstance(resultL,list):
        resultL.append([mData,Threshold,data])
    if (d>Threshold).sum() ==0:
        return wMean(data[d<Threshold],wL[d<Threshold]),wStd(data[d<Threshold],wL[d<Threshold]),len(data)
    else:
        return QC(data[d<Threshold],threshold,it-1,minThreshold=minThreshold,resultL=resultL,minSta=minSta,mid=mid,wL=wL[d<Threshold],isSorted=True)

def ms(data):
    m=np.mean(data)
    d = ((data-m)**2).sum()/(len(data)-1)
    return m,d**0.5
def QC__(data,threshold=0.95,it=-1,minThreshold=0.02,minSta=5,resultL=None,isSort=True):
    if it<0:
        it = int(len(data)*1/2)
    if it==0:
        print('***********************************reach depest*******************')
    if len(data)<=minSta:
        return data.mean(),999,len(data)
    if len(data)<=2*minSta or it==0:
        return data.mean(),data.std(),len(data)
    N = len(data)
    if isSort:
        data = np.array(data).copy()
        data.sort()
    m0,s0 = ms(data)
    D20 = s0**2*(N-1)
    if s0<=minThreshold*m0:
        return data.mean(),data.std(),len(data)
    m1,s1 = ms(data[1:])
    D21 = s1**2*(N-2)

    m2,s2 = ms(data[:-1])
    D22 = s2**2*(N-2)
    N  = calNSigma(N,threshold)
    if D21 < D22 and D20-D21>(N*s1)**2:
        return QC(data[1:],threshold,it-1,minThreshold=minThreshold,resultL=resultL,minSta=minSta,isSort=False)
    elif D21 >= D22 and D20-D22>(N*s2)**2:
        return QC(data[:-1],threshold,it-1,minThreshold=minThreshold,resultL=resultL,minSta=minSta,isSort=False)
    else:
        print('done',it)
        return data.mean(),data.std(),len(data)

def showQC(n1=20,n2=4,v1=3.0,v2=3.5,sigma=0.02,threshold=0.95):
    L1 = np.random.randn(n1)*sigma*v1+v1
    L2 = np.random.randn(n2)*sigma*v2+v2
    L = np.array(L1.tolist()+L2.tolist())
    print(L1.mean(),np.mean(L))
    return QC(L,threshold=threshold), QC__(L,threshold=1.5)
def showQC__(fileName,threshold=1.5,minThreshold=0.01):
    plt.close()
    plt.figure(figsize=[2.5,2.5])
    #data = np.array([3.1,3.11,3.105,3.09,3.05,3.0,3.12,3.13,3.2,3.15,3.08,3.07,3.5,3.095,3.112,3.05,3,3.04,3.07,3.11,3.095,3.099,2.9,2.95,3.099,3.098,3.101,3.12,3.01,3.23,3.096,3.091,3.092,3.093,3.0975])
    data0 = np.arange(-1,1,0.1)
    data = 3.1+data0**3*0.1
    #data = 3.1+np.random.randn(20)*0.1
    resultL = []
    m,s,c = QC(data,threshold=threshold,minThreshold=minThreshold,resultL=resultL)
    N = len(resultL)
    #print(resultL)
    for i in range(N):
        M,Thres,Data = resultL[i]
        plt.plot(Data,i+Data*0+1+0*(np.random.rand(len(Data))-0.5),'.b',markersize=0.5)
        plt.errorbar(M,i+1,fmt='k',ecolor='r',xerr=Thres,elinewidth=0.3,capsize=2,capthick=0.2)
    plt.xlabel('km/s')
    plt.ylabel('loop index')
    plt.yticks(np.arange(1,N+1))
    #plt.xlim([3,3.3])
    plt.ylim([0.5,N+1.5])
    plt.text(3.01,N+0.5,'mean: %.2f km/s std: %.2f %%\nQC: %.1f $Thres_{min}$:%.1f%%'%(m,s/m*100,threshold,minThreshold*100))
    plt.tight_layout()
    plt.savefig(fileName,dpi=300)

def rotate(rad,data):
    #RTZ
    rM = np.array([[np.sin(rad),np.cos(rad),0],\
                 [np.cos(rad),-np.sin(rad),0],\
                 [0,0,1]\
                ])
    dataNew = data*0
    dataNew[:,0]=np.sin(rad)*data[:,0]+np.cos(rad)*data[:,1]
    dataNew[:,1]=np.cos(rad)*data[:,0]-np.sin(rad)*data[:,1]
    dataNew[:,2]= data[:,2]
    return dataNew

def deConv_(wave,src,round=5000,threshold=0.05):
    L    = len(wave)
    l    = len(src)
    dl   = L-l
    resp = np.zeros(dl+1)
    #print(dl)
    wave0 = wave/(wave**2).sum()**0.5
    wave  = wave0.copy()
    src  = src/(src**2).sum()**0.5
    for i in range(round):
        cc = signal.correlate(wave,src,'valid')
        index =  np.abs(cc).argmax()
        resp[index]+=cc[index]
        wave[index:(index+l)] -= cc[index]*src
        if (wave**2).sum()**0.5<threshold:
            break
    waveNew = signal.convolve(src,resp)
    return resp,wave0,waveNew,(wave**2).sum()**0.5
def deConv(wave,src,Round=5000,threshold=0.05,device='cuda:0',f=[]):
    L    = len(wave)
    l    = len(src)
    dl   = L-l
    resp = np.zeros(dl+1)
    #print(dl)
    wave0 = (wave/(wave**2).sum()**0.5).astype(np.float32)
    wave  = wave0.copy().astype(np.float32)
    src  = (src/(src**2).sum()**0.5).astype(np.float32)
    wave = torch.tensor(wave,device=device).reshape(1,1,-1)
    src  =  torch.tensor(src,device=device).reshape(1,1,-1)
    resp  =  torch.tensor(resp,device=device).reshape(1,1,-1)
    for i in range(Round):
        cc = torch.nn.functional.conv1d(wave,src)
        for j in range(1):
            index = cc[0,0,:].abs().argmax()
            resp[0,0,index]+=cc[0,0,index]
            wave[0,0,index:(index+l)] -= cc[0,0,index]*src[0,0]
            cc[0,0,max(0,index-5):min(cc.shape[-1]-1,index+5)]=0
        if (wave**2).sum()**0.5<threshold:
            break
    src = src[0,0,:].cpu().numpy()
    resp = resp[0,0,:].cpu().numpy()
    wave = wave[0,0,:].cpu().numpy()
    waveNew = signal.convolve(src,resp)
    return resp,wave0,waveNew,(wave**2).sum()**0.5
def genSrc_(v=80,dL = np.arange(8)*25,delta=0.01,dt=1,diffOrder=0,f=[]):
    dL = dL-dL.min()
    maxT = dL.max()/v
    timeR = maxT+2*dt
    src = np.zeros(int(timeR/delta))
    if len(f)>0:
        fMax = f[-1]
        N = int(1/(4*fMax*delta))
        src = np.zeros(int(timeR/delta)+8*N-1)
        timeL=np.arange(8*N)-4*N
        g = np.exp(-(timeL/N)**2)
    for d in dL:
        time = d/v+dt
        index = round(time/delta)
        src[index]+=1
    if diffOrder>0:
        srcSpec = np.fft.fft(src)
        srcSpec *= (np.arange(len(srcSpec))/len(srcSpec)*1j)**diffOrder
        src     = np.fft.ifft(srcSpec).real
    if len(f)>0:
        src = signal.convolve(g,src)
    return src

modelD = {'380A':[8,25,26.5,17.5,1],
          '380AL':[16,25,26.5,17.5,1]
         }
def genSrc(v=80,model='380A',delta=0.01,dt=1,diffOrder=0,f=[]):
    N,L0,L1,l,M=modelD[model]
    n=round(N/M)
    #dM = np.zeros([M,n,2])
    D = (L1-l)/2
    DM = np.arange(2).reshape([1,1,-1])*l+D
    LL = L0*(n-2)*L0+L1*2
    LLM = np.arange(M).reshape([-1,1,1])*LL
    LM = np.arange(n).reshape([1,-1,1])*L0
    dM = DM+LLM+LM
    return genSrc_(v,dM.reshape([-1]),delta,dt,diffOrder,f)

def calNSigma(N,q0=0.95):
    pi =np.pi
    sqrt2 = 2.0**0.5
    qN = q0**(1/N)
    a = 8*(pi-3)/(3*pi*(4-pi))
    L=np.log(1-qN**2)
    b = 2/(pi*a)
    dr = ( ((b+L/2)**2-L/a)**0.5 -(b+L/2) )**0.5
    return dr*sqrt2


def Max(data,N=5):
    return data.argmax( axis=1),data.max(axis=1)
    youtPos = yout.argmax(axis=1)
    yinMax = yin.max(axis=1)
    youtMax = yout.max(axis=1)
    shape = list(data.shape)
    shapeNew= shape[:1]+shape[2:]
    data = data.reshape([shape[0],shape[1],-1])
    line = np.arange(-N,N+1)
    pos = data[:,N+1:-N-1].argmax(axis=1)+N+1
    A  = data[:,N+1:-N-1].max(axis=1)
    for i in range(data.shape[0]):
        for j in range(data.shape[-1]):
            POS = pos[i,j]
            if A[i,j]>10:
                DATA = data[i,POS+line,j]
                pos[i,j]+=(DATA*line).sum()/DATA.sum()
    return pos.reshape(shapeNew),A.reshape(shapeNew)
