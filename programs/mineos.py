import numpy as np
import sys
import os
from glob import glob
from matplotlib import pyplot as plt

class MINEOS:
    def __init__(self,runPath='/home/jiangyr/Surface-Wave-Dispersion//MINEOS/',model='PREMQL6ic_21808e.card'):
        self.runPath=runPath
        self.model=model
    def loadCheck(self):
        self.checkFiles = glob(self.runPath+\
            'CHECKMODE_FREQERR_FILES/*')
        self.checkFiles = glob(self.runPath+\
        'CHECKMODE_OUTPUT_FILES/*')
    def checking(self):
        with open(self.runPath+'check_log','w+') as f:
            for checkFile in self.checkFiles[:]:
                mode = Mode(checkfile=checkFile,runPath=self.runPath)
                print('recal')
                mode.reCal(f)
    def getNLF(self,lmax=700,sphtor='sph'):
        NL=[]
        LL=[]
        FL=[]
        for i in range(lmax):
            file='./tmp/%s_%s_%07d.fre'%(self.model,sphtor,i)
            print(i,sphtor)
            #if i==0 and sphtor=='sph':
            #    file='./tmp/%s_%s_%07d.fre'%(self.model,'rad',i)
            if os.path.exists(self.runPath+file):
                if i==0:
                    print(file)
                mode = Mode(file,runPath=self.runPath)
                nL,lL,fL,tL = mode.getNLF()
                NL+=nL
                LL+=lL
                FL+=fL
        return np.array(NL),np.array(LL),np.array(FL)
    def plotNLF(self,lmax=700,sphtor='sph'):
        NL,LL,FL=self.getNLF(lmax,sphtor)
        plt.close()
        plt.figure()
        for n in range(NL.max()):
            plt.plot(LL[NL==n],FL[NL==n],'k',linewidth=0.3)
        plt.xlabel('angular degree I')
        plt.ylabel('frequency(mHz)')
        plt.savefig('%s.pdf'%sphtor)
    def plotFun(self,n,l,sphtor):
        if l==0:
            sphtor='rad'
        file='./tmp/%s_%s_%07d.fre'%(self.model,sphtor,l)
        print(file)
        if os.path.exists(self.runPath+file):
            mode = Mode(file,runPath=self.runPath)
            mode.plotFun(n,l,sphtor)
    def calU(self):
        data=np.loadtxt(self.runPath+'21.out',skiprows=1)
        Rs = 6359000
        Rr = 6370000
        dnorm = 5515e-6
        g     = 6.6723e-11
        w = 0.2538264e-2
        r0 = 6371000
        r = data[:,0]
        iS=np.abs(r-Rs).argmin()
        iR=np.abs(r-Rr).argmin()
        mul0=1/(5510*6371000**3)**0.5
        mul1=1/(5510*6371000**3)**0.5/6371000
        U = data[:,1]*mul0
        dU = data[:,2]*mul1
        V = data[:,3]*mul0
        dV = data[:,4]*mul1
        UR = U[iR]
        US = U[iS]
        VR = V[iR]
        VS = V[iS]
        dUR = dU[iR]
        dUS = dU[iS]
        A=np.array([UR*(-0.5*dUS+1/(2*Rs)*US-2**0.5/4/Rs*VS),\
            0,\
            VR*(6**0.5/4*dUS-6**0.5/4/Rs*US+3**0.5/4/Rs*VS)]).reshape([1,3])
        M0=3.548e19
        time=np.arange(30000)
        s = (1-np.cos(w*time.reshape([-1,1])))/w**2*3/(4*np.pi)*M0*A
        print(1/w**2*3/(4*np.pi)*M0*A,w)
        print(s.max(axis=0))
        plt.close()
        plt.figure(figsize=(6,14))
        plt.subplot(3,1,1)
        plt.title('up(r)')
        plt.plot(time,s[:,0],'k')
        plt.xlabel('t/s')
        plt.ylabel('disp /m')
        plt.subplot(3,1,2)
        plt.title('south(theta)')
        plt.plot(time,s[:,1],'k')
        plt.xlabel('t/s')
        plt.ylabel('disp/m')
        plt.subplot(3,1,3)
        plt.title('east(phi)')
        plt.plot(time,s[:,2],'k')
        plt.xlabel('t/s')
        plt.ylabel('disp/m')
        plt.savefig(self.runPath+'21.pdf',dpi=300)
    def plotU(self):
        staL = ['ABNY','BRS','DESK','MAJO']
        for sta in staL:
            file=self.runPath+'20080512_062801.%s.ASC'%sta
            data = np.loadtxt(file,skiprows=6)
            t = data[:,0]
            Z = data[:,1]
            R = data[:,2]
            T = data[:,3]
            comp='ZRT'
            plt.figure(figsize=[6,10])
            plt.subplot(3,1,1)
            plt.plot(t,Z,'k')
            plt.xlim([t[0],t[-1]])
            #plt.xlabel('t/s')
            plt.ylabel('Z/m')
            plt.title(sta)
            plt.subplot(3,1,2)
            plt.plot(t,R,'k')
            plt.xlim([t[0],t[-1]])
            #plt.xlabel('t/s')
            plt.ylabel('R/m')
            plt.title(sta)
            plt.subplot(3,1,3)
            plt.plot(t,T,'k')
            plt.xlabel('t/s')
            plt.ylabel('T/m')
            plt.xlim([t[0],t[-1]])
            plt.title(sta)
            plt.savefig(self.runPath+sta+'.pdf')
            plt.close()
            

class Mode:
    def __init__(self,file='',checkfile='',runPath='.'):
        self.runPath = runPath
        if checkfile!='':
            print(checkfile)
            self.checkFile = checkfile
            with  open(checkfile) as f:
                lines = f.readlines()
            #/home/jiangyr/MINEOS/CHECKMODE_FREQERR_FILES/checkmode.spherr_0000138
            errmiss=checkfile.split('_')[-2][-3:]
            sphtor=checkfile.split('_')[-2].split('.')[-1][:3]
            lStr = checkfile.split('_')[-1]
            self.l=float(lStr)
            self.inputFile = 'MINEOS_INPUT_FILES/mineos.inp%s_%s'%(sphtor,lStr)
            #mineos.inptor_0000648
            self.inputFileNew = self.inputFile+'new'
            self.inputFileMerge = self.inputFile+'Merge'
            self.missModes = []
            self.errModes = []
            for line in lines:
                tmp = line.split()
                if tmp[0]=='Missing':
                    if tmp[2]=='mode:':
                        self.missModes.append([int(tmp[3]),int(tmp[4])])
                    else:
                        self.missModes.append([int(tmp[2]),int(tmp[3])])
                if tmp[0]=='Inaccurate:':
                    self.errModes.append([int(tmp[1]),int(tmp[2])])
        if file!='':
            self.fre = file
            self.fun = file[:-3]+'fun'
    def reCal(self,F=''):
        maxMode = 1
        for mode in self.missModes+self.errModes:
            maxMode = max(mode[0]+12,maxMode)
        if self.cal(maxMode):
            self.merge()
            F.write('%s find\n'%self.checkFile)
        else:
            F.write('%s not find\n'%self.checkFile)
            print('%s not find'%self.checkFile)
    def cal(self,Mode):
        cmd = 'cd %s;./mineos< %s'%(self.runPath,self.inputFileNew)
        for mul in (np.arange(1,100)*0.2).tolist():
            print('write')
            self.write(mul=mul,Mode=Mode)
            if os.path.exists(self.runPath+self.freNew):
                nL,lL,fL,tL,dL = self.getNLF(self.freNew,withD=True)
                #print('file')
                notFind = False
                #print(dL)
                if False:#np.abs(np.array(dL)).max()>2e-1:
                    print('too many err')
                    notFind = True
                else:
                    for mode in self.missModes+self.errModes:
                        if mode[0] not in nL:
                            notFind=True
                            break
                if not notFind:
                    print('find one')
                    return True
            print(cmd)
            os.system(cmd)
        return False
    def merge(self):
        with open(self.runPath+self.inputFileMerge,'w+') as f:
            f.write('2\n')
            f.write('%s\n'%self.fun)
            f.write('%s\n'%self.funNew)
            f.write('%s\n'%self.fun)
        cmd = 'cd %s;./mineos_merge< %s'%(self.runPath,\
            self.inputFileMerge)
        print(cmd)
        os.system(cmd)
    def write(self,mul=2,head='1',Mode=0):
        with open(self.runPath+self.inputFile) as f:
            lines=f.readlines()
        self.fre = lines[1][:-1]
        self.fun = lines[2][:-1]
        lines[1]=lines[1][:-10]+head+lines[1][-9:]
        lines[2]=lines[2][:-10]+head+lines[2][-9:]
        self.freNew = lines[1][:-1]
        self.funNew = lines[2][:-1]
        tmp = lines[-1].split()
        tmp[-1]=str(Mode)
        if self.l>5:
            tmp[2]=str(max(float(tmp[2]),self.l*0.08))
        tmp[2]=str(max(0.01,float(tmp[2])-mul))
        lines[-1]=''
        for t in tmp:
            lines[-1]+=t+' '
        with open(self.runPath+self.inputFileNew,'w+') as f:
            for line in lines:
                #print(line)
                f.write('%s'%line)
    def getNLF(self,freFile='',withD=False):
        if freFile=='':
            freFile=self.fre
        freFile =self.runPath+freFile
        with open(freFile) as f:
            nL=[]
            lL=[]
            fL=[]
            tL=[]
            dL=[]
            for line in f.readlines():
                tmp=line.split()
                n,t,l,f,d=[int(tmp[0]),tmp[1],\
                    int(tmp[2]),float(tmp[4]),float(tmp[-2])]
                nL.append(n)
                lL.append(l)
                tL.append(t)
                fL.append(f)
                dL.append(d)
            if withD:
                return nL,lL,fL,tL,dL
            return nL,lL,fL,tL
    def plotFun(self,n,l,sphtor):
        if l==0:
            sphtor='rad'
        with open(self.runPath+'readInput','w+') as f:
            f.write('%s\n'%self.fun)
            f.write('mode_fun.asc\n')
            f.write('%d,%d\n'%(n,l))
        cmd = 'cd %s;./read_mineos< readInput'%self.runPath
        os.system(cmd)
        data = np.loadtxt(self.runPath+'mode_fun.asc',skiprows=1)
        maxA = (np.abs(data[:,1:]).max(axis=0)*1.5).astype(np.int).tolist()
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA.append(0)
        maxA[0]=max(1,maxA[0])
        maxA[1]=max(70,maxA[1])
        maxA[2]=max(1,maxA[2])
        maxA[3]=max(70,maxA[3])
        if maxA[1]>50:
            maxA[4]=50
            maxA[5]=25
        if maxA[1]>200:
            maxA[4]=100
            maxA[5]=50
        if maxA[1]>400:
            maxA[4]=200
            maxA[5]=100
        if maxA[1]>1000:
            maxA[4]=1000
            maxA[5]=500
        if maxA[1]>2000:
            maxA[4]=2000
            maxA[5]=1000
        if maxA[1]>4000:
            maxA[4]=3000
            maxA[5]=1500
        if maxA[1]>6000:
            maxA[4]=4000
            maxA[5]=2000
        if maxA[3]>50:
            maxA[6]=50
            maxA[7]=25
        if maxA[3]>200:
            maxA[6]=100
            maxA[7]=50
        if maxA[3]>400:
            maxA[6]=200
            maxA[7]=100
        if maxA[3]>1000:
            maxA[6]=500
            maxA[7]=250
        if maxA[3]>1000:
            maxA[6]=1000
            maxA[7]=500
        if maxA[3]>2000:
            maxA[6]=2000
            maxA[7]=1000
        if maxA[3]>4000:
            maxA[6]=3000
            maxA[7]=1500
        if maxA[3]>6000:
            maxA[6]=4000
            maxA[7]=2000
        if maxA[0]>0:
            maxA[8]=1
            maxA[9]=0.5
        if maxA[0]>5:
            maxA[8]=2.5
            maxA[9]=1
        if maxA[0]>10:
            maxA[8]=5
            maxA[9]=2.5
        if maxA[0]>20:
            maxA[8]=10
            maxA[9]=5
        if maxA[2]>0:
            maxA[10]=1
            maxA[11]=0.5
        if maxA[2]>5:
            maxA[10]=2.5
            maxA[11]=1
        if maxA[2]>10:
            maxA[10]=5
            maxA[11]=2.5
        if maxA[2]>20:
            maxA[10]=10
            maxA[11]=5
        
        cmd = 'cd %s;bash eigenfun.gmt5 '%(self.runPath)
        for item in maxA:
            cmd +=' '+str(item)
        print(cmd)
        os.system(cmd)
        cmd = 'cd %s;cp mode_fun.pdf %d_%d_%s.pdf'%(self.runPath,n,l,sphtor)
        print(cmd)
        os.system(cmd)
            
