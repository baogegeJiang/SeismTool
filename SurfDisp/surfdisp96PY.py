'''
#                                                                   c
#   COMPUTER PROGRAMS IN SEISMOLOGY                                 c
#   VOLUME IV                                                       c
#                                                                   c
#   PROGRAM: SRFDIS                                                 c
#                                                                   c
#   COPYRIGHT 1986, 1991                                            c
#   D. R. Russell, R. B. Herrmann                                   c
#   Department of Earth and Atmospheri#Sciences                    c
#   Saint LouIS University                                          c
#   221 North Grand Boulevard                                       c
#   St. LouIS, MISsouri 63103                                       c
#   U. S. A.                                                        c
#                                                                   c
#----------------------------------------------------------------------c
#  ThIS IS a combination of program 'surface80' which search the poles
#  on C-T domain, and the program 'surface81' which search in the F-K
#  domain.  The input data IS slightly different with its precessors.
#  -Wang   06/06/83.
c
#  The program calculates the dISpersion values for any
#  layered model, any frequency, and any mode.
c
#  ThIS program will accept one liquid layer at the surface.
#  In such case ellipticity of rayleigh wave IS that at the
#  top of solid array.  Love wave communications ignore
#  liquid layer.
c
#  Program developed by Robert B Herrmann Saint LouIS
#  univ. Nov 1971, and revISed by C. Y. Wang on Oct 1981.
#  Modified for use in surface wave inversion, and
#  addition of spherical earth flattening transformation, by
#  David R. Russell, St. LouIS University, Jan. 1984.
c
#    Changes
#    28 JAN 2003 - fixed minor but for sphericity correction by
#        saving one parameter in subroutine sphere
#    20 JUL 2004 - removed extraneous line at line 550 
#        since d#not defined
#        if(dabs(c1-c2) <= dmin1(1.d-6*c1,0.005d+0*dc) )go to 1000
#    28 DEC 2007 - changed the Earth flattening to now use layer
#        midpoint and the BISwas (1972: PAGEOPH 96, 61-74, 1972) 
#        density mapping for P-SV  - note a true comparISon
#        requires the ability to handle a fluid core for SH and SV
#        Also permit one layer with fluid IS base of the velocity IS 0.001 km/sec
#-----
#    13 JAN 2010 - modified by Huajian Yao at MIT for calculation of 
#         group or phase velocities
#-----
#    21 NOV 2018 - python wrapper by Marius Paul Isken
c
#-----
'''
import numpy as np
def surfdISp96(thkm,vpm,vsm,rhom,nlayer,iflsph,iwave,mode,igr,kmax,t,cg,err)
    '''       
       parameter(LER=0,LIN=5,LOT=6)
       integer NL, NL2, NLAY
       parameter(NL=100,NLAY=100,NL2=NL+NL)
       integer NP
       parameter (NP=60)

    #-----
    #    LIN - unit for FORTRAN read from terminal
    #    LOT - unit for FORTRAN write to terminal
    #    LER - unit for FORTRAN error output to terminal
    #    NL  - layers in model
    #    NP  - number of unique periods
    # ----
    # ---- parameters
    #    thkm, vpm, vsm, rhom: model for dISpersion calculation
    #    nlayer - I4: number of layers in the model
    #    iflsph - I4: 0 flat earth model, 1 spherical earth model
    #    iwave - I4: 1 Love wave, 2 Rayleigh wave
    #    mode - I4: ith mode of surface wave, 1 fundamental, 2 first higher, ....
    #    igr - I4: 0 phase velocity, > 0 group velocity
    #    kmax - I4: number of periods (t) for dISpersion calculation
    #    t - period vector (t(NP))
    #    cg - output phase or group velocities (vector,cg(NP))
    # ---- 
    
    double precISion twopi,one,onea
    double precISion cc,c1,clow,cm,dc,t1
    double precISion t(NP),c(NP),cb(NP),cg(NP)
    real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
    integer*4 iverb(2)
    integer*4 llw
    integer*4 nsph, ifunc, idISpl, idISpr, IS, ie
    real*4 sone0, ddc0, h0, sone, ddc, h
    '''
    
    iverb = np.zeros(2,dtype=np.int)
    #maximum number of layers in the model
    mmax = nlayer
    nsph = iflsph
    err = 0
    cb = t*0
    c =  t *0
    cg = t*0
    b = vsm.copy()
    a = vpm.copy()
    d = thkm.copy()
    rho = rhom.copy()
    rtp= vsm.copy()
    dtp= vsm.copy()
    btp= vsm.copy()

    if(iwave == 1):
        idISpl = kmax
        idISpr = 0
    elif(iwave == 2):
        idISpl = 0
        idISpr = kmax
         
#---- constant value
    sone0 = 1.500
#---- phase velocity increment for searching root      
    ddc0 = 0.005
#---- frequency increment (%) for calculating group vel. using g = dw/dk = dw/d(w/c)       
    h0 = 0.005
#---- period range IS:ie for calculation of dISpersion     

#-----
#     check for water layer
#-----
    llw=1
    if(b(1) <= 0.0): llw=2
    twopi=np.pi*2
    one=0.01
    if nsph==1: 
        sphere(0,0,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
    JMN = 1
    betmx=-1.e20
    betmn=1.e20
#-----
#     find the extremal velocities to assISt in starting search
#-----
    for i in range(mmax):
        if(b[i]>0.01  and  b[i]<betmn):
            betmn = b[i]
            jmn = i
            jsol = 1
        elif (b[i]<=0.01  and  a[i]<betmn):
            betmn = a[i]
            jmn = i
            jsol = 0
        if(b[i]>betmx): betmx=b[i]

        for ifunc in [1,2]:
            if(ifunc == 1 and idISpl<=0): continue
            if(ifunc == 2 and idISpr<=0): continue
            if(nsph == 1): sphere(ifunc,1,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
            ddc = ddc0
            sone = sone0
            h = h0
            if(sone< 0.01): sone=2.0
            onea=sone
            if(jsol == 0):
                cc1 = betmn
            else:
                gtsolh(a[jmn],b[jmn],cc1) 
            cc1=.95*cc1
            CC1=.90*CC1
            cc=cc1
            dc=ddc
            dc= dc
            c1=cc
            cm=cc
            cb=0
            c=0
            ift=999
            for iq in range(1,mode+1):
                IS = 0
                ie = kmax
        #          read(*,*) IS,ie
        #           write(*,*) 'IS =', IS, ',  ie = ', ie
                itst=ifunc
                for k in range(IS,ie):
                    if(k>=ift):
                        if(iq>1):
                            ift=k
                            itst=0
                            t1a=t[-1]
                            cg[k:] = 0.0
                        if(iverb[ifunc] == 0):
                            iverb[ifunc] = 1
                            err = 1
                    t1=t[k]
                    if(igr>0):
                        t1a=t1/(1.+h)
                        t1b=t1/(1.-h)
                        t1=t1a
                    else:
                        t1a=t1
                        tlb=0
                    if(k == IS  and  iq == 1):
                        c1 = cc
                        clow = cc
                        ifirst = 1
                    elif(k == IS  and  iq>1):
                        c1 = c[IS] + one*dc
                        clow = c1
                        ifirst = 1
                    elif(k>IS  and  iq>1):
                        ifirst = 0
                        clow = c[k] + one*dc
                        c1 = c[k-1]
                        if(c1 < clow):c1 = clow
                    elif(k>IS  and  iq == 1):
                        ifirst = 0
                        c1 = c[k-1] - onea*dc
                        clow = cm
                    getsol(t1,c1,clow,dc,cm,betmx,iret,ifunc,ifirst,d,a,b,rho,rtp,dtp,btp,mmax,llw)
                    if(iret == -1):
                        if(iq>1):
                            ift=k
                            itst=0
                            t1a=t[-1]
                            cg[k:] = 0.0
                        if(iverb[ifunc] == 0):
                            iverb[ifunc] = 1
                            err = 1
                    c[k] = c1
                    if(igr>0) :
                        t1=t1b
                        ifirst = 0
                        clow = cb[k] + one*dc
                        c1 = c1 -onea*dc
                        getsol(t1,c1,clow,dc,cm,betmx,iret,ifunc,ifirst,d,a,b,rho,rtp,dtp,btp,mmax,llw)
                        if(iret == -1):
                            c1 = c[k]  
                        cb[k]=c1
                    else:
                        c1 = 0
                    cc0 = c[k]
                    cc1 = c1
                    if(igr == 0) :
            #-----         output only phase velocity
            #               write(ifunc,*) itst,iq,t[k],cc0,0.0
                        cg[k] = cc0
                    else:
            #-----         calculate group velocity and output phase and group velocities
                        gvel = (1/t1a-1/t1b)/(1/(t1a*cc0)-1/(t1b*cc1))
                        cg[k] = gvel
                #               write(ifunc,*) itst,iq,t[k],(cc0+cc1)/2,gvel
                #-----         print *, itst,iq,t[k],t1a,t1b,cc0,cc1,gvel     
            
                #      write(LOT,*)'improper initial value in dISper - no zero found'
                #      write(LOT,*)'in fundamental mode '
                #      write(LOT,*)'ThIS may be due to low velocity zone '
                #      write(LOT,*)'causing reverse phase velocity dISpersion, '
                #      write(LOT,*)'and mode jumping.'
                #      write(LOT,*)'due to looking for Love waves in a halfspace'
                #      write(LOT,*)'which IS OK if there are Rayleigh data.'
                #      write(LOT,*)'If reverse dISpersion IS the problem,'
                #      write(LOT,*)'Get present model using OPTION 28, edit sobs.d,'
                #      write(LOT,*)'Rerun with onel large than 2'
                #      write(LOT,*)'which IS the default '
                #-----
                #  if we have higher mode data and the model does not find that
                #  mode, just indicate (itst=0) that it has not been found, but
                #  fill out file with dummy results to maintain format - note
                #  eigenfunctions will not be found for these values. The subroutine
                #  'amat' in 'surf' will worry about thIS in building up the
                #  input file for 'surfinv'
                #-----
                #      write(LOT,*)'ifun#= ',ifun#,' (1=L, 2=R)'
                #      write(LOT,*)'mode  = ',iq-1
                #      write(LOT,*)'period= ',t[k], ' for k,IS,ie=',k,IS,ie
                #      write(LOT,*)'cc,cm = ',cc,cm
                #      write(LOT,*)'c1    = ',c1
                #      write(LOT,*)'d,a,b,rho (d(mmax)=control ignore)'
                #      write(LOT,'(4f15.5)')(d[i],a[i],b[i],rho[i],i=1,mmax)
                #      write(LOT,*)' c[i],i=1,k (NOTE may be part)'
                #      write(LOT,*)(c[i],i=1,k)
                #    if(k>0)goto 1750
                #      go to 2000

def gtsolh(a,b,c):
    kappa= 0
    k2=0 
    gk2=0
    c[:] = 0.95*b
    for i in range(5):
        gamma = b/a
        kappa = c/b
        k2 = kappa**2
        gk2 = (gamma*kappa)**2
        fac1 = (1.0 - gk2)**0.5
        fac2 = (1.0 - k2)**0.5
        fr = (2.0 - k2)**2 - 4.0*fac1*fac2
        frp = -4.0*(2.0-k2) *kappa +4.0*fac2*gamma*gamma*kappa/fac1+4.0*fac1*kappa/fac2
        frp = frp/b
        c -= fr/frp

def getsol(t1,c1,clow,dc,cm,betmx,iret,ifunc,ifirst,d,a,b,rho,rtp,dtp,btp,mmax,llw):
# ----
#    subroutine to bracket dispersion curve
#    and  : refine it
# ----
#    t1  - period
#    c1  - initial guess on low side of mode
#    clow    - lowest possible value for present mode in a
#          reversed direction search
#    d# - phase velocity search increment
#    cm  - minimum possible solution
#    betmx   - maximum shear velocity
#    iret    - 1 = successful
#        - -1= unsuccessful
#    ifun#  - 1 - Love
#        - 2 - Rayleigh
#    ifirst  - 1 this is first period for a particular mode
#        - 0 this is not the first period
#            (this is to define period equation sign
#             for mode jumping test)
# ----
#        wvno, omega, twopi
#        real*8 c1, c2, cn, cm, dc, t1, clow
#        real*8 dltar, del1, del2, del1st, plmn
#        save del1st
#        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)        
#        integer llw,mmax
# ----
#    to avoid problems in mode jumping with reversed dispersion
#    we note what the polarity of period equation is for phase
#    velocities just beneath the zero crossing at the 
#        first period computed.
# ----
#    bracket solution
# ----
    twopi=np.pi*2
    omega=twopi/t1
    wvno=omega/c1
    del1 = dltar(wvno,omega,ifunc,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
    if(ifirst = 1): 
        del1st = del1
    plmn = np.sign(del1st)*np.sign(del1)
    if(ifirst = 1):
        idir = +1
    elif(ifirst != 1 .and. plmn >=0) :
        idir = +1
    elif(ifirst != 1 .and. plmn < 0) :
        idir = -1
    # ----
    #    idir indicates the direction of the search for the
    #    true phase velocity from the initial estimate.
    #    Usually phase velocity increases with period and
    #    we always underestimate, so phase velocity should increase
    #    (idir = +1). For reversed dispersion, we should look
    #    downward from the present estimate. However, we never
    #    go below the floor of clow, when the direction is reversed
    # ----
    for i in range(1000):
        if(idir > 0):
            c2 = c1 + dc
        else
            c2 = c1 - dc
        if(c2 <= clow) :
            idir = +1
            c1 = clow
        if(c2 <= clow) continue
        omega=twopi/t1
        wvno=omega/c2
        del2 = dltar(wvno,omega,ifunc,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
        if (np.sign(del1)!=np.sign(del2)):
            nevill(t1,c1,c2,del1,del2,ifunc,cn,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
            c1 = cn
            if(c1 > (betmx)):
                iret = -1
                return
            iret = 1
            return
        del1=del2
        if(c1 < cm) or (c1 >= (betmx+dc)):
            iret = 1
            return
def sphere(ifunc,iflag,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
# ----
#    Transform spherical earth to flat earth
#
#    Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
#    mode computations, in  Methods in Computational Physics, 
#        Volume 11,
#    Seismology: Surface Waves and Earth Oscillations,  
#        B. A. Bolt (ed),
#    Academic Press, New York
c
#    Love Wave Equations  44, 45 , 41 pp 112-113
#    Rayleigh Wave Equations 102, 108, 109 pp 142, 144
c
#    Revised 28 DEC 2007 to use mid-point, assume linear variation in
#    slowness instead of using average velocity for the layer
#    Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
c
#    ifun#  I*4 1 - Love Wave
#                2 - Rayleigh Wave
#    iflag   I*4 0 - Initialize
#                1 - Make model  for Love or Rayleigh Wave
# ----
       parameter(NL=100,NP=60)
       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
       integer mmax,llw
#       common/modl/ d,a,b,rho,rtp,dtp,btp
#       common/para/ mmax,llw,twopi
       double precision z0,z1,r0,r1,dr,ar,tmp,twopi
       save dhalf
       ar=6370.0d0
       dr=0.0d0
       r0=ar
       d(mmax)=1.0
       if(iflag = 0)  :
           do 5 i=1,mmax
               dtp(i)=d(i)
               rtp(i)=rho(i)
   5       continue
           do 10 i=1,mmax
               dr=dr+dble(d(i))
               r1=ar-dr
               z0=ar*dlog(ar/r0)
               z1=ar*dlog(ar/r1)
               d(i)=z1-z0
# ----
#              use layer midpoint
# ----
               TMP=(ar+ar)/(r0+r1)
               a(i)=a(i)*tmp
               b(i)=b(i)*tmp
               btp(i)=tmp
               r0=r1
  10       continue
           dhalf = d(mmax)
       else
           d(mmax) = dhalf
           do 30 i=1,mmax
               if(ifunc = 1) :
                    rho(i)=rtp(i)*btp(i)**(-5)
               else if(ifunc = 2) :
                    rho(i)=rtp(i)*btp(i)**(-2.275)
               endif
  30       continue
       endif
       d(mmax)=0.0
       return
       end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
       subroutine nevill(t,c1,c2,del1,del2,ifunc,cc,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
# ----
#  hybrid method for refining root once it has been bracketted
#  between c1 and c2.  interval halving is used where other schemes
#  would be inefficient.  once suitable region is found neville s
#  iteration method is used to find root.
#  the procedure alternates between the interval halving and neville
#  techniques using whichever is most efficient
# ----
#    the control integer nev means the following:
c
#    nev = 0 force interval halving
#    nev = 1 permit neville iteration if conditions are proper
#    nev = 2 neville iteration is being used
# ----
       parameter (NL=100,NP=60)
       implicit double precision (a-h,o-z)
       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
       dimension x(20),y(20)
       integer llw,mmax
#       common/modl/ d,a,b,rho,rtp,dtp,btp
#       common/para/ mmax,llw,twopi
# ----
#    initial guess
# ----
       omega = twopi/t
       call half(c1,c2,c3,del3,omega,ifunc,d,a,b,rho,rtp,dtp,btp,
    &  mmax,llw,twopi,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
       nev = 1
       nctrl=1
 100 continue
       nctrl=nctrl+1
       if(nctrl >= 100) go to 1000
# ----
#    make sure new estimate is inside the previous values. If not
#    perform interval halving
# ----
       if(c3  <  dmin1(c1,c2) .or. c3. gt.dmax1(c1,c2)) :
           nev = 0
           call half(c1,c2,c3,del3,omega,ifunc,d,a,b,rho,rtp,dtp,btp,
    &  mmax,llw,twopi,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
       endif
           s13 = del1 - del3
           s32 = del3 - del2
# ----
#    define new bounds according to the sign of the period equation
# ----
           if(dsign(1.d+00,del3)*dsign(1.d+00,del1)  < 0.0d+00) : 
               c2 = c3
               del2 = del3
           else
               c1 = c3
               del1 = del3
           endif
# ----
#    check for convergence. A relative error criteria is used
# ----
       if(dabs(c1-c2) <= 1.d-6*c1) go to 1000
# ----
#    if the slopes are not the same between c1, c3 and c3
#    do not use neville iteration
# ----
       if(dsign (1.0d+00,s13).ne.dsign (1.0d+00,s32)) nev = 0
# ----
#    if the period equation differs by more than a factor of 10
#    use interval halving to avoid poor behavior of polynomial fit
# ----
       ss1=dabs(del1)
       s1=0.01*ss1
       ss2=dabs(del2)
       s2=0.01*ss2
       if(s1 > ss2.or.s2 > ss1 .or. nev = 0)  :
           call half(c1,c2,c3,del3,omega,ifunc,d,a,b,rho,rtp,dtp,btp,
    &  mmax,llw,twopi,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
           nev = 1
           m = 1
       else
           if(nev = 2) :
               x(m+1) = c3
               y(m+1) = del3
           else
               x(1) = c1
               y(1) = del1
               x(2) = c2
               y(2) = del2
               m = 1
           endif
# ----
#    perform Neville iteration. Note instead of generating y(x)
#    we interchange the x and y of formula to solve for x(y) when
#    y = 0
# ----
           do 900 kk = 1,m
               j = m-kk+1
               denom = y(m+1) - y(j)
               if(dabs(denom) < 1.0d-10*abs(y(m+1)))goto 950
               x(j)=(-y(j)*x(j+1)+y(m+1)*x(j))/denom
 900       continue
           c3 = x(1)
           wvno = omega/c3
           del3 = dltar(wvno,omega,ifunc,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
#    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
           nev = 2
           m = m + 1
           if(m > 10)m = 10
           goto 951
 950       continue
           call half(c1,c2,c3,del3,omega,ifunc,d,a,b,rho,rtp,dtp,btp,
    &  mmax,llw,twopi,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
           nev = 1
           m = 1
 951       continue
       endif
       goto 100
1000 continue
       cc = c3
       return
       end

       subroutine half(c1,c2,c3,del3,omega,ifunc,d,a,b,rho,rtp,dtp,btp,
    &  mmax,llw,twopi,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
       implicit double precision (a-h,o-z)
       parameter(NL=100)
       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)    
       c3 = 0.5*(c1 + c2)
       wvno=omega/c3
       del3 = dltar(wvno,omega,ifunc,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
#    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
       return
       end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
       function dltar(wvno,omega,kk,d,a,b,rho,rtp,dtp,btp,mmax,llw,twop)
#    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
#  control the way to P-SV or SH.
c
       implicit double precision (a-h,o-z)
       parameter(NL=100)
       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)        
c
       if(kk = 1) :
#  love wave period equation
         dltar = dltar1(wvno,omega,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
       elseif(kk = 2) :
#  rayleigh wave period equation
         dltar = dltar4(wvno,omega,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
#    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
       endif
       end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
       function dltar1(wvno,omega,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
#  find SH dispersion values.
c
       parameter (NL=100,NP=60)
       implicit double precision (a-h,o-z)
       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
       integer llw,mmax
#       common/modl/ d,a,b,rho,rtp,dtp,btp
#       common/para/ mmax,llw,twopi
c
#  Haskell-Thompson love wave formulation from halfspace
#  to surface.
c
       beta1=dble(b(mmax))
       rho1=dble(rho(mmax))
       xkb=omega/beta1
       wvnop=wvno+xkb
       wvnom=dabs(wvno-xkb)
       rb=dsqrt(wvnop*wvnom)
       e1=rho1*rb
       e2=1.d+00/(beta1*beta1)
       mmm1 = mmax - 1
       do 600 m=mmm1,llw,-1
         beta1=dble(b(m))
         rho1=dble(rho(m))
         xmu=rho1*beta1*beta1
         xkb=omega/beta1
         wvnop=wvno+xkb
         wvnom=dabs(wvno-xkb)
         rb=dsqrt(wvnop*wvnom)
         q = dble(d(m))*rb
         if(wvno < xkb) :
               sinq = dsin(q)
               y = sinq/rb
               z = -rb*sinq
               cosq = dcos(q)
         elseif(wvno = xkb) :
               cosq=1.0d+00
               y=dble(d(m))
               z=0.0d+00
         else
               fac = 0.0d+00
               if(q < 16)fac = dexp(-2.0d+0*q)
               cosq = ( 1.0d+00 + fac ) * 0.5d+00
               sinq = ( 1.0d+00 - fac ) * 0.5d+00
               y = sinq/rb
               z = rb*sinq
         endif
         e10=e1*cosq+e2*xmu*z
         e20=e1*y/xmu+e2*cosq
         xnor=dabs(e10)
         ynor=dabs(e20)
         if(ynor > xnor) xnor=ynor
         if(xnor < 1.d-40) xnor=1.0d+00
         e1=e10/xnor
         e2=e20/xnor
 600 continue
       dltar1=e1
       return
       end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
       function dltar4(wvno,omga,d,a,b,rho,rtp,dtp,btp,mmax,llw,twopi)
#    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
#  find P-SV dispersion values.
c
       parameter (NL=100,NP=60)
       implicit double precision (a-h,o-z)
       dimension e(5),ee(5),ca(5,5)
       real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
#       common/modl/ d,a,b,rho,rtp,dtp,btp
#       common/para/ mmax,llw,twopi
#       common/ovrflw/ a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
c
       omega=omga
       if(omega < 1.0d-4) omega=1.0d-4
       wvno2=wvno*wvno
       xka=omega/dble(a(mmax))
       xkb=omega/dble(b(mmax))
       wvnop=wvno+xka
       wvnom=dabs(wvno-xka)
       ra=dsqrt(wvnop*wvnom)
       wvnop=wvno+xkb
       wvnom=dabs(wvno-xkb)
       rb=dsqrt(wvnop*wvnom)
       t = dble(b(mmax))/omega
# ----
#  E matrix for the bottom half-space.
# ----
       gammk = 2.d+00*t*t
       gam = gammk*wvno2
       gamm1 = gam - 1.d+00
       rho1=dble(rho(mmax))
       e(1)=rho1*rho1*(gamm1*gamm1-gam*gammk*ra*rb)
       e(2)=-rho1*ra
       e(3)=rho1*(gamm1-gammk*ra*rb)
       e(4)=rho1*rb
       e(5)=wvno2-ra*rb
# ----
#  matrix multiplication from bottom layer upward
# ----
       mmm1 = mmax-1
       do 500 m = mmm1,llw,-1
         xka = omega/dble(a(m))
         xkb = omega/dble(b(m))
         t = dble(b(m))/omega
         gammk = 2.d+00*t*t
         gam = gammk*wvno2
         wvnop=wvno+xka
         wvnom=dabs(wvno-xka)
         ra=dsqrt(wvnop*wvnom)
         wvnop=wvno+xkb
         wvnom=dabs(wvno-xkb)
         rb=dsqrt(wvnop*wvnom)
         dpth=dble(d(m))
         rho1=dble(rho(m))
         p=ra*dpth
         q=rb*dpth
         beta=dble(b(m))
# ----
#  evaluate cosP, cosQ,.... in var.
#  evaluate Dunkin's matrix in dnka.
# ----
         call var(p,q,ra,rb,wvno,xka,xkb,dpth,w,cosp,exa,
    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
         call dnka(ca,wvno2,gam,gammk,rho1,
    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
         do 200 i=1,5
           cr=0.0d+00
           do 100 j=1,5
             cr=cr+e(j)*ca(j,i)
 100     continue
           ee(i)=cr
 200   continue
         call normc(ee,exa)
         do 300 i = 1,5
           e(i)=ee(i)
 300   continue
 500 continue
       if(llw.ne.1)  :
# ----
#  include water layer.
# ----
         xka = omega/dble(a(1))
         wvnop=wvno+xka
         wvnom=dabs(wvno-xka)
         ra=dsqrt(wvnop*wvnom)
         dpth=dble(d(1))
         rho1=dble(rho(1))
         p = ra*dpth
         beta = dble(b(1))
         znul = 1.0d-05
         call var(p,znul,ra,znul,wvno,xka,znul,dpth,w,cosp,exa,
    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
         w0=-rho1*w
       dltar4 = cosp*e(1) + w0*e(2)
       else
       dltar4 = e(1)
       endif
       return
       end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       subroutine var(p,q,ra,rb,wvno,xka,xkb,dpth,w,cosp,exa,a0,cpcq,
    &  cpy,cpz,cqw,cqx,xy,xz,wy,wz)
# ----
#  find variables cosP, cosQ, sinP, sinQ, etc.
#  as well as cross products required for compound matrix
# ----
#  To handle the hyperbolic functions correctly for large
#  arguments, we use an extended precision procedure,
#  keeping in mind that the maximum precision in double
#  precision is on the order of 16 decimal places.
c
#  So  cosp = 0.5 ( exp(+p) + exp(-p))
#           = exp(p) * 0.5 * ( 1.0 + exp(-2p) )
#  becomes
#      cosp = 0.5 * (1.0 + exp(-2p) ) with an exponent p
#  In performing matrix multiplication, we multiply the modified
#  cosp terms and add the exponents. At the last step
#  when it is necessary to obtain a true amplitude,
#  we  : form exp(p). For normalized amplitudes at any depth,
#  we carry an exponent for the numerator and the denominator, and
#  scale the resulting ratio by exp(NUMexp - DENexp)
c
#  The propagator matrices have three basic terms
c
#  HSKA        cosp  cosq
#  DUNKIN      cosp*cosq     1.0
c
#  When the extended floating point is used, we use the
#  largest exponent for each, which is  the following:
c
#  Let pex = p exponent > 0 for evanescent waves = 0 otherwise
#  Let sex = s exponent > 0 for evanescent waves = 0 otherwise
#  Let exa = pex + sex
c
#  Then the modified matrix elements are as follow:
c
#  Haskell:  cosp -> 0.5 ( 1 + exp(-2p) ) exponent = pex
#            cosq -> 0.5 ( 1 + exp(-2q) ) * exp(q-p)
#                                         exponent = pex
#         (this is because we are normalizing all elements in the
#          Haskell matrix )
#   Compound:
#           cosp * cosq -> normalized cosp * cosq exponent = pex + qex
#            1.0  ->    exp(-exa)
# ----
       implicit double precision (a-h,o-z)
#       common/ovrflw/   a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
       exa=0.0d+00
       a0=1.0d+00
# ----
#  examine P-wave eigenfunctions
#     checking whether c> vp c=vp or c < vp
# ----
       pex = 0.0d+00
       sex = 0.0d+00
       if(wvno < xka) :
              sinp = dsin(p)
              w=sinp/ra
              x=-ra*sinp
              cosp=dcos(p)
       elseif(wvno = xka) :
              cosp = 1.0d+00
              w = dpth
              x = 0.0d+00
       elseif(wvno > xka) :
              pex = p
              fac = 0.0d+00
              if(p < 16)fac = dexp(-2.0d+00*p)
              cosp = ( 1.0d+00 + fac) * 0.5d+00
              sinp = ( 1.0d+00 - fac) * 0.5d+00
              w=sinp/ra
              x=ra*sinp
       endif
# ----
#  examine S-wave eigenfunctions
#     checking whether c > vs, c = vs, c < vs
# ----
       if(wvno < xkb) :
              sinq=dsin(q)
              y=sinq/rb
              z=-rb*sinq
              cosq=dcos(q)
       elseif(wvno = xkb) :
              cosq=1.0d+00
              y=dpth
              z=0.0d+00
       elseif(wvno > xkb) :
              sex = q
              fac = 0.0d+00
              if(q < 16)fac = dexp(-2.0d+0*q)
              cosq = ( 1.0d+00 + fac ) * 0.5d+00
              sinq = ( 1.0d+00 - fac ) * 0.5d+00
              y = sinq/rb
              z = rb*sinq
       endif
# ----
#  form eigenfunction products for use with compound matrices
# ----
       exa = pex + sex
       a0=0.0d+00
       if(exa < 60.0d+00) a0=dexp(-exa)
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
       fac = 0.0d+00
       if(qmp > -40.0d+00)fac = dexp(qmp)
       cosq = cosq*fac
       y=fac*y
       z=fac*z
       return
       end
c
c
c
       subroutine normc(ee,ex)
#  This routine is an important step to control over- or
#  underflow.
#  The Haskell or Dunkin vectors are normalized before
#  the layer matrix stacking.
#  Note that some precision will be lost during normalization.
c
       implicit double precision (a-h,o-z)
       dimension ee(5)
       ex = 0.0d+00
       t1 = 0.0d+00
       do 10 i = 1,5
         if(dabs(ee(i)) > t1) t1 = dabs(ee(i))
  10 continue
       if(t1 < 1.d-40) t1=1.d+00
       do 20 i =1,5
         t2=ee(i)
         t2=t2/t1
         ee(i)=t2
  20 continue
# ----
#  store the normalization factor in exponential form.
# ----
       ex=dlog(t1)
       return
       end
c
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
       subroutine dnka(ca,wvno2,gam,gammk,rho,
    &  a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
#   Dunkin's matrix.
c
       implicit double precision (a-h,o-z)
       dimension ca(5,5)
#       common/ ovrflw / a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
       data one,two/1.d+00,2.d+00/
       gamm1 = gam-one
       twgm1=gam+gamm1
       gmgmk=gam*gammk
       gmgm1=gam*gamm1
       gm1sq=gamm1*gamm1
       rho2=rho*rho
       a0pq=a0-cpcq
       ca(1,1)=cpcq-two*gmgm1*a0pq-gmgmk*xz-wvno2*gm1sq*wy
       ca(1,2)=(wvno2*cpy-cqx)/rho
       ca(1,3)=-(twgm1*a0pq+gammk*xz+wvno2*gamm1*wy)/rho
       ca(1,4)=(cpz-wvno2*cqw)/rho
       ca(1,5)=-(two*wvno2*a0pq+xz+wvno2*wvno2*wy)/rho2
       ca(2,1)=(gmgmk*cpz-gm1sq*cqw)*rho
       ca(2,2)=cpcq
       ca(2,3)=gammk*cpz-gamm1*cqw
       ca(2,4)=-wz
       ca(2,5)=ca(1,4)
       ca(4,1)=(gm1sq*cpy-gmgmk*cqx)*rho
       ca(4,2)=-xy
       ca(4,3)=gamm1*cpy-gammk*cqx
       ca(4,4)=ca(2,2)
       ca(4,5)=ca(1,2)
       ca(5,1)=-(two*gmgmk*gm1sq*a0pq+gmgmk*gmgmk*xz+
    *          gm1sq*gm1sq*wy)*rho2
       ca(5,2)=ca(4,1)
       ca(5,3)=-(gammk*gamm1*twgm1*a0pq+gam*gammk*gammk*xz+
    *          gamm1*gm1sq*wy)*rho
       ca(5,4)=ca(2,1)
       ca(5,5)=ca(1,1)
       t=-two*wvno2
       ca(3,1)=t*ca(5,3)
       ca(3,2)=t*ca(4,3)
       ca(3,3)=a0+two*(cpcq-ca(1,1))
       ca(3,4)=t*ca(2,3)
       ca(3,5)=t*ca(1,3)
       return
       end