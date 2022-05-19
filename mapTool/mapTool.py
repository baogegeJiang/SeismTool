import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as basemap
from netCDF4 import Dataset
import os
from scipy import interpolate as interp
from matplotlib import cm
from lxml import etree
#from pykml.factory import KML_ElementMaker as KML
from pycpt.load import gmtColormap as cpt2cm
from ..mathTool.distaz import DistAz
from ..mathTool.mathFunc_bak import R as mathR
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from ..plotTool import figureSet as fs
import matplotlib.colors as mcolors
#from pycpt.load import gmtColormap
def gmtColormap_openfile(cptf, name=None):
    """Read a GMT color map from an OPEN cpt file

    Parameters
    ----------
    cptf : open file or url handle
        path to .cpt file
    name : str, optional
        name for color map
        if not provided, the file name will be used
    """
    # generate cmap name
    if name is None:
        name = '_'.join(os.path.basename(cptf.name).split('.')[:-1])

    # process file
    x = []
    r = []
    g = []
    b = []
    lastls = None
    A0=1
    A1=1
    for l in cptf.readlines():
        ls = l.split()

        # skip empty lines
        if not ls:
            continue

        # parse header info
        if ls[0] in ["#", b"#"]:
            if ls[-1] in ["HSV", b"HSV"]:
                colorModel = "HSV"
            else:
                colorModel = "RGB"
            if len(ls)>2:
                if ls[1]=='RANGE':
                    A0 = np.abs(float(ls[3].split('/')[0]))
                    A1 = float(ls[3].split('/')[1])
            continue

        # skip BFN info
        if ls[0] in ["B", b"B", "F", b"F", "N", b"N"]:
            continue

        # parse color vectors
        if '/' in l:
            x.append(float(ls[0]))
            R,G,B=ls[1].split('/')
            if x[-1]>0:
                x[-1]=x[-1]*A1
            else:
                x[-1]=x[-1]*A0
            r.append(float(R))
            g.append(float(G))
            b.append(float(B))
            x.append(float(ls[2]))
            if x[-1]>0:
                x[-1]=x[-1]*A1
            else:
                x[-1]=x[-1]*A0
            R,G,B=ls[3].split('/')
            r.append(float(R))
            g.append(float(G))
            b.append(float(B))
        else:
            x.append(float(ls[0]))
            r.append(float(ls[1]))
            g.append(float(ls[2]))
            b.append(float(ls[3]))
            x.append(float(ls[4]))
            r.append(float(ls[5]))
            g.append(float(ls[6]))
            b.append(float(ls[7]))

        # save last row
        #lastls = ls

    
    
    x = np.array(x)
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    if colorModel == "HSV":
        for i in range(r.shape[0]):
            # convert HSV to RGB
            rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360., g[i], b[i])
            r[i] = rr ; g[i] = gg ; b[i] = bb
    elif colorModel == "RGB":
        r /= 255.
        g /= 255.
        b /= 255.

    red = []
    blue = []
    green = []
    xNorm = (x - x[0])/(x[-1] - x[0])
    doneL =[]
    for i in range(len(x)):
        X = x[i]
        if X in doneL:
            continue
        else:
            doneL.append(X)
        red.append([xNorm[i],r[x==X][0],r[x==X][-1]])
        green.append([xNorm[i],g[x==X][0],g[x==X][-1]])
        blue.append([xNorm[i],b[x==X][0],b[x==X][-1]])

    # return colormap
    cdict = dict(red=red,green=green,blue=blue)
    return mcolors.LinearSegmentedColormap(name=name,segmentdata=cdict)


def gmtColormap(cptfile, name=None):
    """Read a GMT color map from a cpt file

    Parameters
    ----------
    cptfile : str or open file-like object
        path to .cpt file
    name : str, optional
        name for color map
        if not provided, the file name will be used
    """
    with open(cptfile, 'r') as cptf:
        return gmtColormap_openfile(cptf, name=name)
cmap = cpt2cm(os.path.dirname(__file__)+'/../data/temperatureInv')
cmapNoGreen = cpt2cm(os.path.dirname(__file__)+'/../data/no_green.cpt').reversed()
cmapTemp = cpt2cm(os.path.dirname(__file__)+'/../data/temperature')
#cmapETopo=cpt2cm(os.path.dirname(__file__)+'/ETOPO1.cpt')
cmapETopo=gmtColormap(os.path.dirname(__file__)+'/ETOPO1.cpt')
volcano=np.loadtxt(os.path.dirname(__file__)+'/../data/volcano')
pi=3.1415927
def genBaseMap(R=[0,90,0,180], topo=None,**kwags):
    m=basemap.Basemap(llcrnrlat=R[0],urcrnrlat=R[1],llcrnrlon=R[2],\
    urcrnrlon=R[3])
    #m=basemap.Basemap(width=111700*(R[1]*-R[0]), height=111700*(R[3]*-R[2]),lat_0=R[0]*0.5+R[1]*0.5,lat_1=R[0],lon_0=R[2]*0.5+R[3]*0.5,\
    #    projection='eqdc')
    if topo == None:
        #m.etopo()
        pass
    else:
        plotTopo(m,R,topo=topo,**kwags)
    return m
def plotOnMap(m, lat,lon,cmd='.b',markersize=0.5,alpha=1,linewidth=0.5,mfc=[],**kwags):
    x,y=m(lon,lat)
    if len(mfc)>0:
        return plt.plot(x,y,cmd,markersize=markersize,alpha=alpha,linewidth=linewidth,mfc=mfc,**kwags)
    else:
        return plt.plot(x,y,cmd,markersize=markersize,alpha=alpha,linewidth=linewidth,**kwags)

def scatterOnMap(m, lat,lon,s,alpha=1,c=None):
    x,y=m(lon,lat)
    plt.scatter(x,y,s=s,alpha=alpha,c=c)

def readnetcdf(R,file='/media/jiangyr/MSSD/ETOPO1_Ice_g_gmt4.grd'):
    nc=Dataset(file)
    laStr = 'lat'
    loStr = 'lon'
    zStr  = 'z'
    print(nc.variables)
    if 'etopo05' in file:
        laStr = 'ETOPO05_Y'
        loStr = 'ETOPO05_X'
        zStr  = 'ROSE'
    if laStr not in nc.variables:
        laStr = 'y'
        loStr = 'x'
        zStr  = 'z'
    la=np.array(nc.variables[laStr][:])
    lo=np.array(nc.variables[loStr][:])
    R = R.copy()
    R0 = R[:2]
    R1 = R[2:]
    R0.sort()
    R1.sort()
    laI0 = max(0,np.abs(la-R0[0]).argmin()-2)
    laI1 = min(np.abs(la-R0[1]).argmin()+2,len(la))
    loI0 = max(0,np.abs(lo-R1[0]).argmin()-2)
    loI1 = min(np.abs(lo-R1[1]).argmin()+2,len(lo))
    z=nc.variables[zStr][:]
    return np.array(la[laI0:laI1]),np.array(lo[loI0:loI1]),np.array(z[laI0:laI1,loI0:loI1])
def plotLaLoLine(m,dLa=10,dLo=10,La0=-90,Lo0=0,**kwags):
    parallels = np.arange(La0,90.01,dLa)
    m.drawparallels(parallels,labels=[1,0,0,1],**kwags)
    meridians = np.arange(Lo0,360.01,dLo)
    plt.gca().yaxis.set_ticks_position('right')
    m.drawmeridians(meridians,labels=[True,False,False,True],**kwags)

def getDlaDlo(R):
    DLA = R[1] -R[0]
    DLO = R[3] -R[2]
    if DLA<10:
        dLa = 2
    elif DLA<20:
        dLa = 4
    elif DLA<40:
        dLa = 5
    else:
        dLa = 10

    if DLO<10:
        dLo = 2
    elif DLO<20:
        dLo = 4
    elif DLO<40:
        dLo = 5
    else:
        dLo = 10
    return dLa,dLo
def getZInR(la0,lo0,z0,R,laN=500,loN=500):
    la=np.arange(R[0],R[1],(R[1]-R[0])/laN)
    lo=np.arange(R[2],R[3],(R[3]-R[2])/loN)
    Z=interp.interp2d(lo0,la0,z0)
    z=Z(lo,la)
    lo,la=np.meshgrid(lo,la)
    return la,lo,z
def getZInLine(la0,lo0,z0,line,laN=500,loN=500):
    la=np.arange(laN)/(laN-1)*(line.xyL[1][0]-line.xyL[0][0])+line.xyL[0][0]#(line.xyL[0][0],line.xyL[1][0],(line.xyL[1][0]-line.xyL[0][0])/laN)
    lo=np.arange(loN)/(loN-1)*(line.xyL[1][1]-line.xyL[0][1])+line.xyL[0][1]#np.arange(line.xyL[0][1],line.xyL[1][1],(line.xyL[1][1]-line.xyL[0][1])/loN)
    z=interp.interpn((la0,lo0),z0,np.concatenate([la.reshape([-1,1]),lo.reshape([-1,1])],axis=-1),method='linear')
    return la,lo,z
def plotTopo(m,R,topo='/media/jiangyr/MSSD/ETOPO1_Ice_g_gmt4.grd',laN=800,loN=800,cpt='wiki-2.0.cpt',vmax=5000,vmin=0,isColorbar=True):#'cpt17.txt'):
    la0,lo0,z0=readnetcdf(R,topo)
    la,lo,z=getZInR(la0,lo0,z0,R,laN=laN,loN=loN)
    #loM,laM=np.meshgrid(lo,la)
    x,y=m(lo,la)
    a=plt.gca()
    #print(la[0,0],lo[0,0])
    if isinstance(cpt,str):
        cpt = cpt2cm(cpt)
    pc=m.pcolormesh(x,y,z,cmap=cpt,vmin=vmin, vmax=vmax,rasterized=True)
    if isColorbar:
        bar=plt.colorbar()
        bar.set_label('elevation(m)')
    #z.set_clim(-9000,9000)
    a.set_xlim(a.get_xlim())
    return pc

def quakeLs2kml(quakeLs,filename):
    fold=KML.Folder()
    for quakeL in quakeLs:
        for quake in quakeL:
            fold.append(lalodep2Place(quake.loc[0],quake.loc[1],quake.loc[2]))
    content = etree.tostring(etree.ElementTree(fold),pretty_print=True)
    #print(type(content))
    with open(filename,'wb') as fp:
        fp.write(content)

def lalodep2Place(la,lo,dep):
    return KML.Placemark(KML.Point(\
        KML.coordinates(str(lo)+','+str(la)+','+str(dep))))

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
    def plot(self,m=None,cmd='-k',markersize=0.5,alpha=1,isDip=False,l=0.3,linewidth=0.5,**kwags):
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
            return plotOnMap(m,laL,loL,cmd,markersize,alpha,linewidth=linewidth,**kwags)
            if isDip and len(dipLaL)>0:
                plotOnMap(m,dipLaL,dipLoL,cmd0,markersize,alpha,linewidth=linewidth)
        else:
            return plt.plot(loL,laL,cmd,markersize=markersize,alpha=alpha,**kwags)
            if isDip and len(dipLaL)>0:
                plt.plot(dipLoL,dipLaL,cmd0,markersize=markersize,alpha=alpha,**kwags)

def readFault(filename,maxD=10000):
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
                la = float(line[1])
                lo = float(line[0])
                if len(faultL[-1].laL)>0:
                    dLa = faultL[-1].laL[-1]-la
                    dLo = faultL[-1].loL[-1]-lo
                    if (dLa**2+dLo**2)**0.5>maxD:
                        continue
                faultL[-1].laL.append(la)
                faultL[-1].loL.append(lo)
    for fault in faultL:
        fault.update()
    return faultL
faultL = readFault(os.path.dirname(__file__)+'/../data/Chinafault_fromcjw.dat')
class lineArea:
    def __init__(self,p0,p1,dkm):
        self.p0=np.array(p0)
        self.p1=np.array(p1)
        self.lakm=DistAz(p0[0],p0[1],p0[0]+1,p0[1]).getDelta()*111.19
        self.lokm=DistAz(p0[0],p0[1],p0[0],p0[1]+1).getDelta()*111.19
        self.dkm=dkm
        xy1=self.xy(p1)
        self.len=np.linalg.norm(xy1)
        self.v=(xy1)/self.len
    def xy(self,p):
        xy=np.zeros(2)
        xy[0]=(p[0]-self.p0[0])*self.lakm
        xy[1]=(p[1]-self.p0[1])*self.lokm
        return xy
    def convert(self, p, isIn=True):
        h=-999
        dp=self.xy(p)
        h=(dp*self.v).sum()
        if (h<0 or h>self.len) and isIn:
            return -999
        dkm=np.abs(self.calDkm(p))
        if dkm>self.dkm:
            return -999
        return h
    def calDkm(self,p):
        dp=self.xy(p)
        return -(dp[0]*self.v[1]-dp[1]*self.v[0])

def plotInline(quakeLs,p0,p1,dkm=50,mul=3,along=True,alpha=0.3,minSta=10,staInfos=None,minCover=0.7):
    line=lineArea(p0,p1,dkm)
    hL=[]
    depL=[]
    mlL=[]
    dkmL=[]
    timeL=[]
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            loc=quake.loc
            h=line.convert(loc[:2])
            if h>0:
                if staInfos != None:
                    if quake.calCover(staInfos)<minCover:
                        continue
                hL.append(h)
                depL.append(loc[2])
                mlL.append(max(0.1,quake.ml))
                dkmL.append(line.calDkm(loc[:2]))
                timeL.append(quake.time%86400)
                if quake.loc[2]>70 and quake.ml>2:
                    print(quake.time,quake.loc)
    hL=np.array(hL)
    depL=np.array(depL)
    mlL=np.array(mlL)
    dkmL=np.array(dkmL)
    timeL=np.array(timeL)
    if along:
        plt.scatter(hL,depL,mlL*mul,c=timeL,alpha=alpha)
    else:
        plt.scatter(dkmL,depL,mlL*mul,c=timeL,alpha=alpha)
    ax = plt.gca()
    ax.invert_yaxis()



def plotQuakeDis(quakeLs,output='quakeDis.png',cmd='.b',markersize=0.8,\
    alpha=0.3,R=None,topo=None,m=None,staInfos=None,minSta=8,minCover=0.8,\
    faultFile="Chinafault_fromcjw.dat",mul=1,loL0=[],laL0=[],isBox=False):
    la=[]
    lo=[]
    dep=[]
    mlL=[]
    plt.close()
    plt.figure(figsize=[12,8])
    for quakeL in quakeLs:
        for quake in quakeL:
            la.append(quake['la'])
            lo.append(quake['lo'])
            dep.append(quake['dep'])
            mlL.append(ml)
    la=np.array(la)
    lo=np.array(lo)
    dep=np.array(dep)
    mlL=np.array(mlL)
    if R==None:
        R=[la.min(),la.max(),lo.min(),lo.max()]
    #print(R)
    
    if m==None:
        m=genBaseMap(R=R,topo=topo)
    if not staInfos == None:
        sla=[]
        slo=[]
        sdep=[]
        for staInfo in staInfos:
            sla.append(staInfo['la'])
            slo.append(staInfo['lo'])
            sdep.append(staInfo['dep'])
        sla=np.array(sla)
        slo=np.array(slo)
        sdep=np.array(sdep)
        hS,=plotOnMap(m,sla,slo,'^r',markersize=5,alpha=1,linewidth=1)
    faultL=readFault(faultFile)
    hF=None
    for fault in faultL:
        if fault.inR(R):
            hFTmp,=fault.plot(m)
            if hFTmp!=None:
                hF=hFTmp
    if len(laL0)>1:
        hC,=plotOnMap(m,laL0,loL0,'ok',markersize=2,mfc=[1,1,1])
    if isBox:
        laLB0=[38.7,42.2]
        loLB0=[97.5,103.8]
        laLB=np.array([laLB0[0],laLB0[0],laLB0[1],laLB0[1],laLB0[0]])
        loLB=np.array([loLB0[1],loLB0[0],loLB0[0],loLB0[1],loLB0[1]])
        plotOnMap(m,laLB,loLB,'r',linewidth=3,markersize=3)
    hQ,=plotOnMap(m,la,lo,cmd,markersize,alpha)
    #mt.scatterOnMap(m,la,lo,s=np.exp(mlL/1.5)*mul,alpha=alpha,c=np.array([1,0,0]))
    
    plotTopo(m,R)
    if len(laL0)>1:
        #hC,=mt.plotOnMap(m,laL0,loL0,'ok',markersize=2)
        plt.legend((hQ,hC,hS,hF),('Quakes','Catalog','Station','Faults'),\
            bbox_to_anchor=(1, 1),loc='lower right')
    else:
        plt.legend((hQ,hS,hF),('Quakes','Station','Faults'),bbox_to_anchor=(1, 1),\
              loc='lower right')
    parallels = np.arange(0.,90,2.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,2.)
    plt.gca().yaxis.set_ticks_position('left')
    m.drawmeridians(meridians,labels=[True,False,False,True])
    plt.savefig(output)
    return m

def plotQuakeCCDis(quakeCCLs,quakeRefL,output='quakeDis.png',cmd='.r',markersize=0.8,\
    alpha=0.3,R=None,topo=None,m=None,staInfos=None,minSta=8,minCover=0.8,\
    faultFile="Chinafault_fromcjw.dat",mul=1,minCC=0.5):
    la=[]
    lo=[]
    dep=[]
    mlL=[]
    count=0
    for quakeL in quakeCCLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if staInfos!=None:
                if quake.calCover(staInfos,minCC=minCC)<minCover:
                    continue
            ml=0
            if quake.ml !=None:
                if quake.ml>-2:
                    ml=quake.ml
            la.append(quake.loc[0])
            lo.append(quake.loc[1])
            dep.append(quake.loc[2])
            mlL.append(ml)
            count+=1
    print(count)
    la=np.array(la)
    lo=np.array(lo)
    dep=np.array(dep)
    mlL=np.array(mlL)
    if R==None:
        R=[la.min(),la.max(),lo.min(),lo.max()]
    laR=[]
    loR=[]
    depR=[]
    mlLR=[]
    for quake in quakeRefL:
        ml=0
        if quake.ml !=None:
            if quake.ml>-2:
                ml=quake.ml
        laR.append(quake.loc[0])
        loR.append(quake.loc[1])
        depR.append(quake.loc[2])
        mlLR.append(ml)
    laR=np.array(laR)
    loR=np.array(loR)
    depR=np.array(depR)
    mlLR=np.array(mlLR)
    if m==None:
        m=genBaseMap(R=R,topo=topo)
    if not staInfos == None:
        sla=[]
        slo=[]
        sdep=[]
        for staInfo in staInfos:
            sla.append(staInfo['la'])
            slo.append(staInfo['lo'])
            sdep.append(staInfo['dep'])
        sla=np.array(sla)
        slo=np.array(slo)
        sdep=np.array(sdep)
        hS,=plotOnMap(m,sla,slo,'^k',markersize=5,alpha=1)
    faultL=readFault(faultFile)
    hF=None
    for fault in faultL:
        if fault.inR(R):
            hFTmp,=fault.plot(m)
            if hFTmp!=None:
                hF=hFTmp

    hT,=plotOnMap(m,laR,loR,'*b',markersize*2,1)
    hCC,=plotOnMap(m,la,lo,cmd,markersize,alpha)
    print(len(laR),len(la))

    plt.legend((hT,hCC,hS,hF),('Templates','Microearthquakes','Station','Faults'))
    #mt.plotOnMap(m,laR,loR,'*k',markersize*2,1)
    #mt.scatterOnMap(m,la,lo,s=np.exp(mlL/1.5)*mul,alpha=alpha,c=np.array([1,0,0]))
    plt.title('minSta:%d minCover:%.1f minCC:%.1f MFT:%d'%(minSta,minCover,minCC,count))
    dD=max(int((R[1]-R[0])*10)/40,int((R[3]-R[2])*10)/40)
    parallels = np.arange(int(R[0]),int(R[1]+1),dD)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(int(R[2]),int(R[3]+1),dD)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    return m

def plotDep(qL,R,filename):
    paraL = qL.paraL(['la','lo','dep','ml','time'],req={'R':R})
    plt.close()
    fig=plt.figure()
    timeL = (np.array(paraL['time'])-np.array(paraL['time']).min())/86400
    dLaPerKm,dLoPerKm = R.perKm()
    if dLaPerKm>dLoPerKm:
        x= paraL['la']
        perKm = dLaPerKm
        xLabel='la'
        xlim=[R.xyL[:,0].min(),R.xyL[:,0].max()]
    else:
        x= paraL['lo']
        perKm = dLoPerKm
        xLabel='lo'
        xlim=[R.xyL[:,1].min(),R.xyL[:,1].max()]
    pc=plt.scatter(x,paraL['dep'],s=((np.array(paraL['ml'])*0+1)**2)*0.3/3,c=timeL,cmap='gist_rainbow')
    cbar=fig.colorbar(pc, orientation="horizontal",fraction=0.046, pad=0.04)
    cbar.set_label('time from the first earthquake')
    plt.xlabel(xLabel)
    plt.ylabel('dep/km')
    plt.ylim([50,-10])
    plt.xlim(xLim)
    plt.gca().set_aspect(perKm)
    plt.savefig(filename,dpi=300)
    plt.close()

def plotDepV2(qL,R,filename,isPer=False,vModel=None,isTopo=False,vM=''):
    paraL = qL.paraL(['la','lo','dep','ml','time'],req={'R':R})
    plt.close()
    fig=plt.figure(figsize=[2.7,1.6])
    ax=plt.gca()
    ax_divider = make_axes_locatable(ax)
    if isTopo:
        la0,lo0,z0=readnetcdf(R.xyL.transpose().reshape([-1]).tolist())
        la,lo,z=getZInLine(la0,lo0,z0,R)
        laLo = np.array([la.tolist(),lo.tolist()]).transpose()
        l = R.l(laLo)
        Tax = ax_divider.append_axes("top", size="50%", pad="20%")
        Tax.plot(l,z/1000,'k',linewidth=0.3)
        #Tax.set_xlabel('Distance/km')
        Tax.set_ylabel('$T$/km')
        Tax.set_ylim([0,6])
        Tax.set_xlim([0,R.L])
        Tax.set_xticks([])
        Tax.text(10,6,'$'+R.name+'$',ha='left',va='bottom',c='r',size=8)
        Tax.text(R.L,6,'$'+R.name+'$\'',ha='right',va='bottom',c='r',size=8)
        #Tax.set_aspect(5)
    #timeL = (np.array(paraL['time'])-np.array(paraL['time']).min())/86400
    laLo = np.array([paraL['la'],paraL['lo']]).transpose()
    l = R.l(laLo)
    if not isinstance(vModel,type(None)):
        if vM!='':
            vM = vM(vModel.z)
        la,lo,z,Dist,V= vModel.outputP2(R.xyL,isPer=isPer,line=R,vM=vM)
        Dist = R.l(np.array([la[0].tolist(),lo[0].tolist()]).transpose())
        if isPer:
            pc=ax.pcolormesh(Dist,z,V,vmin=+np.abs(V).max(),vmax=-np.abs(V).max(),cmap=cmapNoGreen,rasterized=True)
        else:
            if vModel.mode == 'vp':
                vmax = 8#7.8
                vmin = 4#4.5#4
            if vModel.mode == 'vs':
                vmax = 8/1.7#7.8/1.7
                vmin = 4/1.7#4.5/1.7#4
            pc=ax.pcolormesh(Dist,z,V,vmin=vmin,vmax=vmax,cmap='jet_r',rasterized=True)
        #print(V.max(),V.min())
        #position=fig.add_axes([0.15, 0.05, 0.7, 0.03])
        #cbar=fig.colorbar(pc,fraction=0.046, pad=0.04)
        # add an axes above the main axes.
        cax = ax_divider.append_axes("bottom", size="10%", pad="155%")
        #cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
        cbar=fig.colorbar(pc, cax=cax, orientation="horizontal")
        # change tick position to top. Tick position defaults to bottom and overlaps
        # the image.
        #cax2.xaxis.set_ticks_position("top")
        if isPer:
            if 'p' in vModel.mode  :
                cbar.set_label('$dVp/V_0$')
            elif 's' in vModel.mode:
                cbar.set_label('$dVs/V_0$')
        else:
            cbar.set_label('$V$ (km/s)')
    #pc=plt.scatter(l,paraL['dep'],s=1,c=timeL*0,cmap='gist_rainbow')#((np.array(paraL['ml'])*0+1)**2)*0.3/3
    pc=ax.plot(l,paraL['dep'],'.k',markersize=0.1,linewidth=0.01)
    #cbar=fig.colorbar(pc, orientation="horizontal",fraction=0.046, pad=0.04)
    #cbar.set_label('time from the first earthquake')
    ax.set_xlabel('Distance/km')
    ax.set_ylabel('$Z$/km')
    ax.set_ylim([70,-10])
    ax.set_xlim([0,R.L])
    ax.set_aspect(1)
    if not isinstance(vModel,type(None)):
        ax.set(facecolor='#A9A9A9')
    fig.tight_layout()
    plt.savefig(filename,dpi=300)
    plt.close()

def showInMap(v,laL,loL,R,resFile,name='res',abc=''):
    plt.close()
    fig=plt.figure(figsize=[3,4])#[4,5]
    m = basemap.Basemap(llcrnrlat=R[0],urcrnrlat=R[1],llcrnrlon=R[2],\
        urcrnrlon=R[3])
    for fault in faultL:
        if fault.inR(R):
            fh=fault.plot(m,markersize=0.3)
    loM,laM=np.meshgrid(loL,laL)
    x,y=m(loM,laM)
    parallels = np.arange(0.,90,2.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,360.,2.)
    plt.gca().yaxis.set_ticks_position('left')
    m.drawmeridians(meridians,labels=[True,False,False,True])
    V = np.ma.masked_where(np.isnan(v), v)
    cmp=plt.get_cmap('coolwarm')
    #cmap = co.copy(cm.get_cmap(plt.rcParams['image.cmap']))
    cmap.set_bad('y', 0)
    #pc=m.pcolormesh(x,y,V,cmap=cmap,shading='gouraud')
    pc=m.pcolormesh(x,y,V,cmap=cmap,rasterized=True)
    if len(abc)>0:
        fs.setABC(abc)
    #cbar=fig.colorbar(pc,orientation="horizontal")
    ax=plt.gca()
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("bottom", size="7%", pad="5%")
    cbar=fig.colorbar(pc, cax=cax, orientation="horizontal")
    #cbar=plt.colorbar()
    cbar.set_label(name)
    fig.tight_layout()
    plt.savefig(resFile,dpi=300)