
def readFault(filename):
    faultL=[]
    strikeD={'N':0,'NNE':pi/8*1,'NE':pi/8*2,'NEE':pi/8*3,'E':pi/8*4\
    ,'SEE':pi/8*5,'SE':pi/8*6,'SSE':pi/8*7,'S':pi/8*8,'SSW':pi/8*9,\
    'SW':pi/8*10,'SWW':pi/8*11,'W':pi/8*12,'NWW':pi/8*13,'NW':pi/8*14\
    ,'NNW':pi/8*15}
    with open(filename) as f:
        for line in f.readlines():
            line=line.split()
            if line[0]=='>':
                faultL.append(Fault(laL=[],loL=[]))
            if line[0]=='#':
                line1=line[1].split('|')
                if len(line1)>=25:
                    for i in range(5,8):
                        strikeStr=line1[i]
                        if len(strikeStr)>0:
                            try:
                                if strikeStr in strikeD:
                                    faultL[-1].dip=strikeD[strikeStr]
                                else:
                                    strikeL=strikeStr.split('-')
                                    tmp=0
                                    ct=0
                                    for strike in strikeL:
                                        if len(strike)!=0:
                                            ct+=1
                                            tmp+=tmp+float(strike)
                                    if ct!=0:
                                        if i==5:
                                            faultL[-1].strike=tmp/ct/180*pi
                                        if i==6:
                                            faultL[-1].dip=tmp/ct/180*pi
                                        if i==7:
                                            faultL[-1].angle=tmp/ct/180*pi
                            except:
                                pass
                            else:
                                pass
            if line[0]!='>' and line[0]!='#':
                faultL[-1].laL.append(float(line[1]))
                faultL[-1].loL.append(float(line[0]))
    for fault in faultL:
        fault.update()
    return faultL
def plotOnMap(m, lat,lon,cmd='.b',markersize=0.5,alpha=1,linewidth=0.5,mfc=[]):
    x,y=m(lon,lat)
    if len(mfc)>0:
        return plt.plot(x,y,cmd,markersize=markersize,alpha=alpha,linewidth=linewidth,mfc=mfc)
    else:
        return plt.plot(x,y,cmd,markersize=markersize,alpha=alpha,linewidth=linewidth)

class Fault:
    def __init__(self,R=None,laL=[],loL=[],strike=None,dip=None,angle=None):
        self.R=R
        self.laL=laL
        self.loL=loL
        self.dip=dip
        self.angle=angle
        self.strike=strike
    def update(self):
        laL=np.array(self.laL)
        loL=np.array(self.loL)
        self.R=[laL.min(),laL.max(),loL.min(),loL.max()]
    def inR(self,R0):
        R=self.R
        if (R[1]<R0[0] or R[0]>R0[1]) or (R[3]<R0[2] or R[2]>R0[3]) :
            return False
        else:
            return True
    def plot(self,m=None,cmd='-k',markersize=0.5,alpha=1,isDip=False,l=0.3,linewidth=0.5):
        laL=np.array(self.laL)
        loL=np.array(self.loL)
        dipLaL=[]
        dipLoL=[]
        cmd0='r'
        if self.strike!=None:
            la0=laL[int(len(laL)/2)]
            lo0=loL[int(len(loL)/2)]
            if self.angle!=None:
                l=l*np.sin(self.angle)
            dipLaL=np.array([0,l*np.cos(self.strike+pi/2)])+la0
            dipLoL=np.array([0,l*np.sin(self.strike+pi/2)])+lo0
        if m!=None:
            return plotOnMap(m,laL,loL,cmd,markersize,alpha,linewidth=linewidth)
            if isDip and len(dipLaL)>0:
                plotOnMap(m,dipLaL,dipLoL,cmd0,markersize,alpha,linewidth=linewidth)
        else:
            return plt.plot(loL,laL,cmd,markersize=markersize,alpha=alpha)
            if isDip and len(dipLaL)>0:
                plt.plot(dipLoL,dipLaL,cmd0,markersize=markersize,alpha=alpha)