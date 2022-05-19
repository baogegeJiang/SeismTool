import torch
import numpy as np
import warnings
from numba import jit
warnings.filterwarnings('ignore')
K=np
Data = np.array
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
#@jit
def calDisp(d,a,b,rho,T,mode=1,isR=True,sone0=1.5,dc0=0.01,h0=0.005,isFlat=False,ar=6370):
    if isFlat:
        d,a,b,rho = sphere(d,a,b,rho,isR=isR,ar=ar)
    if b.min()<0.01:
        betmn = a.min()
        A = betmn
        B = b[a.argmin()]
        jsol=0
    else:
        #b.min()
        betmn = b.min()
        B= betmn
        A = a[b.argmin()]
        jsol = 1
    betmx = b.max()
    if jsol==0:
        cc1 = betmn
    else:
        cc1 = gtsolh(A,B)
    #cc1=.95*cc1
    cc= cc1
    dc = dc0
    c1=cc
    cm=cc
    c  = K.zeros(len(T))
    ift=999
    onea = sone0
    del1st=1
    for m in range(mode):
        Is = 0
        Ie = len(T)
        itst= isR
        for k in range(Is,Ie):
            t1=T[k]
            '''
            c-----
            c     get initial phase velocity estimate to begin search
            c
            c     in the notation here, c() is an array of phase velocities
            c     c(k-1) is the velocity estimate of the present mode
            c     at the k-1 period, while c(k) is the phase velocity of the
            c     previous mode at the k period. Since there must be no mode
            c     crossing, we make use of these values. The only complexity
            c     is that the dispersion may be reversed. 
            c
            c     The subroutine getsol determines the zero crossing and refines
            c     the root.
            c-----
            '''
            if(k == Is and m==0):
                c1 = cc
                clow = cc
                ifirst = True
                #print(t1,c1,clow,dc,cm,betmx,isR,ifirst)
            elif(k == Is and m > 0):
                c1 = c[Is] + dc
                clow = c1
                ifirst = True
            elif(k>Is and m>0):
                ifirst = False
                clow = c[k] + dc
                c1 = c[k-1]
                if(c1 < clow):
                    c1 = clow
            elif(k > Is and m == 0):
                ifirst = False
                c1 = c[k-1] - onea*dc
                clow = cm
                #print(t1,c1,clow,dc,cm,betmx,isR,ifirst)
            #print(k,m,t1,cc,clow,dc,cm,betmx,isR,ifirst)
            c[k],del1st=getsol(d,a,b,rho,t1,c1,clow,dc,cm,betmx,isR,ifirst,del1st=del1st)
            #print(del1st)
            #print(c[0])
    return c
#@jit
def sphere(d,a,b,rho,isR,ar=6370):
    '''
    c-----
    c     Transform spherical earth to flat earth
    c
    c     Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
    c     mode computations, in  Methods in Computational Physics, 
    c         Volume 11,
    c     Seismology: Surface Waves and Earth Oscillations,  
    c         B. A. Bolt (ed),
    c     Academic Press, New York
    c
    c     Love Wave Equations  44, 45 , 41 pp 112-113
    c     Rayleigh Wave Equations 102, 108, 109 pp 142, 144
    c
    c     Revised 28 DEC 2007 to use mid-point, assume linear variation in
    c     slowness instead of using average velocity for the layer
    c     Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
    c
    c     ifunc   I*4 1 - Love Wave
    c                 2 - Rayleigh Wave
    c     iflag   I*4 0 - Initialize
    c                 1 - Make model  for Love or Rayleigh Wave
    c-----
    '''
    r0=ar
    #print(d)
    r1  =ar-d.cumsum(0)
    r0  = r1+d
    z0=ar*K.log(ar/r0)
    z1=ar*K.log(ar/r1)
    d = z1-z0
    tmp=(ar+ar)/(r1+r0)
    a=a*tmp
    b=b*tmp
    if isR:
        rho=rho*tmp**(-2.275)
    else:
        rho=rho*tmp**(-5)
    return d,a,b,rho 

#@jit
def gtsolh(a,b):
    '''
    c-----
    c     starting solution
    c-----
    '''
    c = 0.95*b
    for i in range(1,5):
        gamma = b/a
        kappa = c/b
        k2 = kappa**2
        gk2 = (gamma*kappa)**2
        fac1 = K.sqrt(1.0 - gk2)
        fac2 = K.sqrt(1.0 - k2)
        fr = (2.0 - k2)**2 - 4.0*fac1*fac2
        frp = -4.0*(2.0-k2) *kappa+4.0*fac2*gamma*gamma*kappa/fac1+4.0*fac1*kappa/fac2
        frp = frp/b
        c = c - fr/frp
    return c

#@jit
def dltarL(d,a,b,rho,wvno,omega):
    '''
    c   find SH dispersion values.
    c
        parameter (NL=100,NP=60)
        implicit double precision (a-h,o-z)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        integer llw,mmax
    c        common/modl/ d,a,b,rho,rtp,dtp,btp
    c        common/para/ mmax,llw,twopi
    c
    c   Haskell-Thompson love wave formulation from halfspace
    c   to surface.
    c
    '''
    mmax = len(d)
    xkb=omega/b[mmax-1]
    wvnop=wvno+xkb
    wvnom=K.abs(wvno-xkb)
    rb=K.sqrt(wvnop*wvnom)
    e1=rho[mmax-1]*rb
    e2=1/(b[mmax-1]*b[mmax-1])
    mmm1 = mmax - 2
    i0=0
    if b[0]<0.001:
        i0=1
    for m in range(mmm1-1,i0-1,-1):
        xmu=rho[m]*b[m]*b[m]
        xkb=omega/b[m]
        wvnop=wvno+xkb
        wvnom=K.abs(wvno-xkb)
        rb=K.sqrt(wvnop*wvnom)
        q = d[m]*rb
        if(wvno<xkb):
            sinq = K.sin(q)
            y = sinq/rb
            z = -rb*sinq
            cosq = K.cos(q)
        elif(wvno==xkb):
            cosq=1.000
            y=d[m]
            z=0.00
        else:
            fac = 0.000
            if(q<16):
                fac = K.exp(-2.0*q)
            cosq = ( 1.000 + fac ) * 0.500
            sinq = ( 1.000 - fac ) * 0.500
            y = sinq/rb
            z = rb*sinq
        e10=e1*cosq+e2*xmu*z
        e20=e1*y/xmu+e2*cosq
        xnor=K.abs(e10)
        ynor=K.abs(e20)
        if(ynor>xnor):
            xnor=ynor
        if(xnor<1e-40): 
            xnor=1.000
        e1=e10/xnor
        e2=e20/xnor
    return e1
#@jit
def dltarR(d,a,b,rho,wvno,omga):
    mmax = len(d)
    #print(mmax)
    omega=omga
    if omega < 1.0e-4:
        omega=1.0e-4
    wvno2=wvno**2
    xka=omega/a[-1]
    xkb=omega/b[-1]
    wvnop=wvno+xka
    wvnom=K.abs(wvno-xka)
    ra=K.sqrt(wvnop*wvnom)
    wvnop=wvno+xkb
    wvnom=K.abs(wvno-xkb)
    rb=K.sqrt(wvnop*wvnom)
    t = b[-1]/omega
    #E matrix for the bottom half-space.
    gammk = 2.0*t*t
    gam   = gammk*wvno2
    gamm1 = gam - 1
    rho1  = rho[-1]
    e = Data([(rho1*rho1*(gamm1*gamm1-gam*gammk*ra*rb)),(-rho1*ra),(rho1*(gamm1-gammk*ra*rb)),(rho1*rb),(wvno2-ra*rb)])
    #matrix multiplication from bottom layer upward
    i0=0
    if b[0]<0.001:
        i0=1
    for m in range(mmax-2,i0-1,-1):
        xka = omega/a[m]
        xkb = omega/b[m]
        t = b[m]/omega
        gammk = 2*t*t
        gam = gammk*wvno2
        wvnop=wvno+xka
        wvnom=K.abs(wvno-xka)
        ra= K.sqrt(wvnop*wvnom)
        wvnop=wvno+xkb
        wvnom=K.abs(wvno-xkb)
        rb=K.sqrt(wvnop*wvnom)
        dpth=d[m]
        rho1=rho[m]
        p=ra*dpth
        q=rb*dpth
        #evaluate cosP, cosQ,.... in var.#
        # evaluate Dunkin's matrix in dnka.
        w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz=var(p,q,ra,rb,wvno,xka,xkb,dpth)
        ca=dnka(wvno2,gam,gammk,rho1,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
        #print(ca)
        e = (e.reshape([-1,1])*ca).sum(axis=0)
        e=normc(e)
        #e[:]=ee[:]
        #print(e)
    if i0==1:
        #include water layer.
        xka = omega/a[0]
        wvnop=wvno+xka
        wvnom=K.abs(wvno-xka)
        ra=K.sqrt(wvnop*wvnom)
        dpth=d[0]
        rho1=rho[0]
        p = ra*dpth
        znul = 1e-5
        w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz=var(p,znul,ra,znul,wvno,xka,znul,dpth)
        w0=-rho1*w
        return cosp*e[0] + w0*e[1]
    else:
        return e[0]
#@jit
def getsol(d,a,b,rho,t1,c1,clow,dc,cm,betmx,isR,ifirst,del1st=1):
    '''
    c-----
    c     subroutine to bracket dispersion curve
    c     and then refine it
    c-----
    c     t1  - period
    c     c1  - initial guess on low side of mode
    c     clow    - lowest possible value for present mode in a
    c           reversed direction search
    c     dc  - phase velocity search increment
    c     cm  - minimum possible solution
    c     betmx   - maximum shear velocity
    c     iret    - 1 = successful
    c         - -1= unsuccessful
    c     ifunc   - 1 - Love
    c         - 2 - Rayleigh
    c     ifirst  - 1 this is first period for a particular mode
    c         - 0 this is not the first period
    c             (this is to define period equation sign
    c              for mode jumping test)
    c-----
    c-----
    c     to avoid problems in mode jumping with reversed dispersion
    c     we note what the polarity of period equation is for phase
    c     velocities just beneath the zero crossing at the 
    c         first period computed.
    c-----
    c     bracket solution
    c-----
    '''
    twopi=2*3.141592653589793
    omega=twopi/t1
    wvno=omega/c1
    if isR :
        del1 = dltarR(d,a,b,rho,wvno,omega)
    else:
        del1 = dltarL(d,a,b,rho,wvno,omega)
    if ifirst:
        del1st = del1
    plmn = K.sign(del1st)*K.sign(del1)
    if ifirst:
        idir = +1
    elif plmn >=0:
        idir = +1
    else:
        idir = -1
    '''
    c-----
    c     idir indicates the direction of the search for the
    c     true phase velocity from the initial estimate.
    c     Usually phase velocity increases with period and
    c     we always underestimate, so phase velocity should increase
    c     (idir = +1). For reversed dispersion, we should look
    c     downward from the present estimate. However, we never
    c     go below the floor of clow, when the direction is reversed
    c-----
    '''
    for i in range(1000):
        if(idir>0):
            c2 = c1 + dc
        else:
            c2 = c1 - dc
        if ifirst:
            pass
            #print(c1,c2)
        if(c2<=clow):
            idir = +1
            c1 = clow
        if(c2<=clow):
            continue
        #if ifirst:
        wvno=omega/c2
        if isR :
            del2 = dltarR(d,a,b,rho,wvno,omega)
        else:
            del2 = dltarL(d,a,b,rho,wvno,omega)
        #print(c1,i)
        if ifirst:
            pass
            #print(c1,c2,del1,del2,isR)
        if not (K.sign(del1)!=K.sign(del2)):
            c1=c2
            del1=del2
            if(c1<=cm):
                return -1,del1st
            if(c1>=(betmx+dc)):
                return -1,del1st
            continue
        else:
            cn=nevill(d,a,b,rho,t1,c1,c2,del1,del2,isR,twopi)
            c1 = cn
            if(c1>(betmx)):
                return -1,del1st
            return c1,del1st
    return c1,del1st
#@jit
def nevill(d,a,b,rho,t,c1,c2,del1,del2,isR,twopi):
    '''
    c-----
    c   hybrid method for refining root once it has been bracketted
    c   between c1 and c2.  interval halving is used where other schemes
    c   would be inefficient.  once suitable region is found neville s
    c   iteration method is used to find root.
    c   the procedure alternates between the interval halving and neville
    c   techniques using whichever is most efficient
    c-----
    c     the control integer nev means the following:
    c
    c     nev = 0 force interval halving
    c     nev = 1 permit neville iteration if conditions are proper
    c     nev = 2 neville iteration is being used
    c-----
    
    c        common/modl/ d,a,b,rho,rtp,dtp,btp
    c        common/para/ mmax,llw,twopi
    c-----
    c     initial guess
    c-----
    '''
    omega = twopi/t
    c3,del3,=half(d,a,b,rho,c1,c2,omega,isR)
    nev = 1
    x = K.zeros(20)
    y = K.zeros(20)
    #100 continue
    for nctrl in range(100):
        '''
        c-----
        c     make sure new estimate is inside the previous values. If not
        c     perform interval halving
        c-----
        '''
        if(c3 -c2)*(c3-c1)>=0:
            nev = 0
            c3,del3,=half(d,a,b,rho,c1,c2,omega,isR)
        s13 = del1 - del3
        s32 = del3 - del2
        '''
        c-----
        c     define new bounds according to the sign of the period equation
        c-----
        '''
        if(K.sign(del3)*K.sign(del1) < 0): 
            c2 = c3
            del2 = del3
        else:
            c1 = c3
            del1 = del3
        #check for convergence. A relative error criteria is used
        if K.abs(c1-c2)<= 1.e-6*c1:
            break
        #if the slopes are not the same between c1, c3 and c3
        if(K.sign(s13)!=K.sign (s32)): 
            nev = 0
        ss1=K.abs(del1)
        s1=0.01*ss1
        ss2=K.abs(del2)
        s2=0.01*ss2
        if(s1>=ss2 or s2>=ss1 or nev==0):
            c3,del3,=half(d,a,b,rho,c1,c2,omega,isR)
            nev = 1
            m = 0
        else:
            if(nev==2):
                x[m+1] = c3
                y[m+1] = del3
            else:
                x[0] = c1
                y[0] = del1
                x[1] = c2
                y[1] = del2
                m = 0
        '''
        c-----
        c     perform Neville iteration. Note instead of generating y(x)
        c     we interchange the x and y of formula to solve for x(y) when
        c     y = 0
        c-----
        '''
        isBreak =False
        for kk in range(0,m+1):
            j = m-kk+1
            denom = y[m+1] - y[j]
            if not (K.abs(denom)<1e-10*K.abs(y[m+1])):
                x[j]=(-y[j]*x[j+1]+y[m+1]*x[j])/denom
            else:
                c3,del3,=half(d,a,b,rho,c1,c2,omega,isR)
                nev = 1
                m = 1
                isBreak=True
                break
        if not isBreak:
            c3 = x[0]
            wvno = omega/c3
            if isR :
                del3 = dltarR(d,a,b,rho,wvno,omega)
            else:
                del3 = dltarL(d,a,b,rho,wvno,omega)
            nev = 2
            m = m + 1
            if(m>9):
                m = 9
    return c3
#@jit
def half(d,a,b,rho,c1,c2,omega,isR):    
    c3 = 0.5*(c1 + c2)
    wvno=omega/c3
    if isR :
        del3 = dltarR(d,a,b,rho,wvno,omega)
    else:
        del3 = dltarL(d,a,b,rho,wvno,omega)
    return c3,del3

#@jit
def normc_(ee):
    #c   This routine is an important step to control over- or
    #c   underflow.
    #c   The Haskell or Dunkin vectors are normalized before
    #c   the layer matrix stacking.
    #c   Note that some precision will be lost during normalization.
    #c   
    t1 = K.abs(ee).max()
    if t1<1e-40:
        t1=1.00
    ee /= t1
    return K.log(t1)  
#@jit
def normc(ee):
    #c   This routine is an important step to control over- or
    #c   underflow.
    #c   The Haskell or Dunkin vectors are normalized before
    #c   the layer matrix stacking.
    #c   Note that some precision will be lost during normalization.
    #c   
    t1 = K.abs(ee).max()
    if t1<1e-40:
        t1=1.00
    return ee/t1
#@jit
def var(p,q,ra,rb,wvno,xka,xkb,dpth):
    #var(p,q,ra,rb,wvno,xka,xkb,dpth)
    '''
    c-----
    c   find variables cosP, cosQ, sinP, sinQ, etc.
    c   as well as cross products required for compound matrix
    c-----
    c   To handle the hyperbolic functions correctly for large
    c   arguments, we use an extended precision procedure,
    c   keeping in mind that the maximum precision in double
    c   precision is on the order of 16 decimal places.
    c
    c   So  cosp = 0.5 ( exp(+p) + exp(-p))
    c            = exp(p) * 0.5 * ( 1.0 + exp(-2p) )
    c   becomes
    c       cosp = 0.5 * (1.0 + exp(-2p) ) with an exponent p
    c   In performing matrix multiplication, we multiply the modified
    c   cosp terms and add the exponents. At the last step
    c   when it is necessary to obtain a true amplitude,
    c   we then form exp(p). For normalized amplitudes at any depth,
    c   we carry an exponent for the numerator and the denominator, and
    c   scale the resulting ratio by exp(NUMexp - DENexp)
    c
    c   The propagator matrices have three basic terms
    c
    c   HSKA        cosp  cosq
    c   DUNKIN      cosp*cosq     1.0
    c
    c   When the extended floating point is used, we use the
    c   largest exponent for each, which is  the following:
    c
    c   Let pex = p exponent > 0 for evanescent waves = 0 otherwise
    c   Let sex = s exponent > 0 for evanescent waves = 0 otherwise
    c   Let exa = pex + sex
    c
    c   Then the modified matrix elements are as follow:
    c
    c   Haskell:  cosp -> 0.5 ( 1 + exp(-2p) ) exponent = pex
    c             cosq -> 0.5 ( 1 + exp(-2q) ) * exp(q-p)
    c                                          exponent = pex
    c          (this is because we are normalizing all elements in the
    c           Haskell matrix )
    c    Compound:
    c            cosp * cosq -> normalized cosp * cosq exponent = pex + qex
    c             1.0  ->    exp(-exa)
    c-----
    '''
    a0=1.0
    #examine P-wave eigenfunctions
    #checking whether c> vp c=vp or c < vp
    pex =0 #K.zeros(1)
    sex= 0#K.zeros(1)
    if wvno < xka:
        sinp = K.sin(p)
        w=sinp/ra
        x=-ra*sinp
        cosp=K.cos(p)
    elif wvno==xka:
        cosp =1# K.ones(1)
        w = dpth
        x = 0.0
    elif(wvno>xka):
        pex = p
        #fac = K.zeros(1)
        #if p < 16:
        fac = K.exp(-2*p)
        cosp = ( 1.00 + fac) * 0.500
        sinp = ( 1.000 - fac) * 0.500
        w=sinp/ra
        x=ra*sinp
    #examine S-wave eigenfunctions
    # #checking whether c > vs, c = vs, c < vs
    if(wvno < xkb):
        sinq=K.sin(q)
        y=sinq/rb
        z=-rb*sinq
        cosq=K.cos(q)
    elif(wvno==xkb):
        cosq=1
        y=dpth
        z=0
    elif(wvno > xkb):
        sex = q
        #fac = 0.000
        #if q < 16:
        fac = K.exp(-2.0*q)
        cosq = ( 1.000 + fac ) * 0.500
        sinq = ( 1.000 - fac ) * 0.500
        y = sinq/rb
        z = rb*sinq
    #form eigenfunction products for use with compound matrices
    exa = pex + sex
    #a0=0.000
    #if exa<60.000: 
    a0=K.exp(-exa)
    cpcq=cosp*cosq
    cpy=cosp*y
    cpz=cosp*z
    cqw=cosq*w
    cqx=cosq*x
    xy=x*y
    xz=x*z
    wy=w*y
    wz=w*z
    qmp = sex - pex
    #fac = K.zeros(1)
    #if qmp>-40.000:
    fac = K.exp(qmp)
    cosq = cosq*fac
    y=fac*y
    z=fac*z
    return w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    #      w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    #p,q,ra,rb,wvno,xka,xkb,dpth,w,cosp,exa,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz,
ca = K.zeros([5,5])
#@jit
def dnka(wvno2,gam,gammk,rho,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz,ca=ca):
    #(wvno2,gam,gammk,rho1,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
    #ca=dnka(wvno2,gam,gammk,rho1,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
    one=1
    two=2
    gamm1 = gam-one
    twgm1=gam+gamm1
    gmgmk=gam*gammk
    gmgm1=gam*gamm1
    gm1sq=gamm1*gamm1
    rho2=rho*rho
    a0pq=a0-cpcq
    #
    ca[0,0]=cpcq-two*gmgm1*a0pq-gmgmk*xz-wvno2*gm1sq*wy
    ca[0,1]=(wvno2*cpy-cqx)/rho
    ca[0,2]=-(twgm1*a0pq+gammk*xz+wvno2*gamm1*wy)/rho
    ca[0,3]=(cpz-wvno2*cqw)/rho
    ca[0,4]=-(two*wvno2*a0pq+xz+wvno2*wvno2*wy)/rho2
    ca[1,0]=(gmgmk*cpz-gm1sq*cqw)*rho
    ca[1,1]=cpcq
    ca[1,2]=gammk*cpz-gamm1*cqw
    ca[1,3]=-wz
    ca[1,4]=ca[0,3]
    ca[3,0]=(gm1sq*cpy-gmgmk*cqx)*rho
    ca[3,1]=-xy
    ca[3,2]=gamm1*cpy-gammk*cqx
    ca[3,3]=ca[1,1]
    ca[3,4]=ca[0,1]
    ca[4,0]=-(two*gmgmk*gm1sq*a0pq+gmgmk*gmgmk*xz+gm1sq*gm1sq*wy)*rho2
    ca[4,1]=ca[3,0]
    ca[4,2]=-(gammk*gamm1*twgm1*a0pq+gam*gammk*gammk*xz+gamm1*gm1sq*wy)*rho
    ca[4,3]=ca[1,0]
    ca[4,4]=ca[0,0]
    t=-two*wvno2
    ca[2,0]=t*ca[4,2]
    ca[2,1]=t*ca[3,2]
    ca[2,2]=a0+two*(cpcq-ca[0,0])
    ca[2,3]=t*ca[1,2]
    ca[2,4]=t*ca[0,2]
    #print(type(ca10),type(ca11),type(ca12),type(ca13),type(ca14))
    '''
    ca = K.cat(\
        [
    torch.cat([ca00.reshape([-1]), ca01.reshape([-1]), ca02.reshape([-1]), ca03.reshape([-1]), ca04.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca10.reshape([-1]), ca11.reshape([-1]), ca12.reshape([-1]), ca13.reshape([-1]), ca14.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca20.reshape([-1]), ca21.reshape([-1]), ca22.reshape([-1]), ca23.reshape([-1]), ca24.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca30.reshape([-1]), ca31.reshape([-1]), ca32.reshape([-1]), ca33.reshape([-1]), ca34.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca40.reshape([-1]), ca41.reshape([-1]), ca42.reshape([-1]), ca43.reshape([-1]), ca44.reshape([-1])]).reshape([1,-1]),
    ]
    )
    '''
    '''
    ca = Data(\
        [
    [ca00, ca01, ca02, ca03, ca04],
    [ca10, ca11, ca12, ca13, ca14],
    [ca20, ca21, ca22, ca23, ca24],
    [ca30, ca31, ca32, ca33, ca34],
    [ca40, ca41, ca42, ca43, ca44],
    ]
    )
    '''
    return ca

