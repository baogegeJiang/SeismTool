import numpy as np
from matplotlib import pyplot as plt
from scipy import signal,interpolate
import tool2 as tool
from imp import reload
fcn.defProcess()
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
#tf.config.list_physical_devices('GPU')
import fcn
from obspy import read
from glob import glob
import os
import segyio

modelFile ='resStr_220621-085414_model.h5'
#modelFile='/home/jiangyr//Surface-Wave-Dispersion/SeismTool/predict/0130_0.95_0.05_3.2_randMove/resStr_220621-085414_model.h5'
model = fcn.modelUp(modelFile,channelList=[0,1],up=1,mul=1,randA=0.05,TL=(14**np.arange(0,1.000001,1/49))*10,disRandA=1/15,disMaxR=5,maxCount=3072,delta=0.5)

tL = (14**np.arange(0,1.000001,1/49)*10)/125

file = 'jiangyr/shot_ikey20.segy'
file = 'Deonise_B4_Sgy/Deonise_B4_SW0017.sgy'
file = '/home/jiangyr/shot_ikey20.segy'
file=glob('/autofs/2022pai/2022_test_code/2022Pai30/02_SuFaceWave_Supression/INPUT/2022PGD/Shot_sgy/Shot_*.sgy')[0]
resDir = 'plotFinalV2/%s/'%(os.path.basename(file))
if not os.path.exists(resDir):
    os.makedirs(resDir)


st = segyio.open(file,ignore_geometry=True)
sD = tool.decompose(st)
print(len(sD))
VL=np.arange(0.6,2.2,0.01)
resD = {}
count=0
for sourceId in sD:
    sourceDir='%s/%s/'%(resDir,sourceId)
    if not os.path.exists(sourceDir):
        os.makedirs(sourceDir)
    iL = sD[sourceId]
    g= tool.toGather(iL,st)
    plt.close()
    dy = g.yg-g.ys
    dy -= dy[int(len(dy)/2)]
    booL = np.abs(dy)<0.05
    if booL.sum()>10:
        plt.subplot(2,1,1)
        data = g.data[booL].transpose()
        dis = g.dis[booL]*np.sign(g.az[booL]-180.0001)
        xg = g.xg[booL]
        snrL=g.snrL[booL]
        plt.pcolor(dis,g.timeL,data/data.max(axis=0),cmap='bwr',vmin=-1,vmax=1,rasterized=True)
        plt.ylim([g.timeL[-1],g.timeL[0]])
        plt.ylabel('time(s)')
        plt.subplot(2,1,2)
        plt.xlabel('dis(km)')
        plt.plot(dis,snrL,'k')
        #plt.plot(g.xg[booL],g.snrL[booL],'k')
        plt.xlim([dis.min(),dis.max()])
        plt.savefig('%s/waveAndSnr.jpg'%(sourceDir),dpi=300)
    p = g.pair(maxDis=3.5,minDis=0.2,minDDis=0.5,maxDDis=2.5,maxTheta=5,deltaM=0.1,deltaD=0.1,minSnr=3)
    p.plot()
    v =tool.Pair.toV(p,model) 
    resD[sourceId]=tool.Velocity(v,p.xm,p.ym,p.ddis)
    V = v[:,:,0,:].sum(axis=0)
    V = V/V.max(axis=0)
    plt.close()
    plt.pcolor(1/tL,VL,V[:,:],vmin=0,vmax=1,cmap='jet',rasterized=True)
    plt.xlabel('f(Hz)')
    plt.ylabel('v(km/s)')
    plt.gca().set_xscale('log')
    plt.savefig('%s/f-v.jpg'%(sourceDir),dpi=300)
    indexL = V.argmax(axis=0)
    tmpV = VL[indexL]
vmax = 1.8
dt=-2
specL = np.array([tool.calSpec(p.data[i,p.timeL>p.ddis[i]/vmax+dt,0],1/tL,p.timeL[p.timeL>p.ddis[i]/vmax+dt]) for i in range(len(p.data))])
dt = p.ddis.reshape([1,-1])/VL.reshape([-1,1])
E =(specL.reshape([1,len(p.data),-1])*np.exp(1j*np.pi*2*dt.reshape([len(VL),-1,1])*(1/tL).reshape([1,1,-1]))).sum(axis=1)
plt.close()
plt.pcolor(1/tL,VL,np.abs(E)/np.abs(E).max(axis=0),vmin=0,vmax=1,cmap='jet',rasterized=True)
plt.plot(1/tL,tmpV,'k')
plt.xlabel('f(Hz)')
plt.ylabel('v(km/s)')
plt.gca().set_xscale('log')
plt.savefig('%s/f-v(fk).jpg'%(sourceDir),dpi=300)
    plt.close()
    plt.plot(p.xm,p.ym,'.k',markersize=0.5)
    plt.xlabel('X(km)')
    plt.ylabel('Y(km)')
    plt.savefig('%s/mid_loc.jpg'%(sourceDir),dpi=300)
    data = p.data[::10000]
    y = model.predict(data.reshape([len(data),-1,1,2]))
    for i in range(len(data)):
        plt.close()
        plt.subplot(2,1,1)
        plt.plot(p.timeL,data[i,:,0],'k',label='wave',linewidth=0.5)
        plt.plot(p.timeL,data[i,:,1],'r',label='dist',linewidth=0.5)
        #plt.xlabel('t(s)')
        plt.legend()
        plt.xlim([p.timeL[0],p.timeL[-1]])
        plt.subplot(2,1,2)
        plt.pcolor(p.timeL,1/tL,y[i,:,0,:].transpose(),vmin=0,vmax=1,cmap='bwr')
        plt.xlabel('t(s)')
        plt.ylabel('f(Hz)')
        plt.gca().set_yscale('log')
        plt.savefig('%s/example_%d.jpg'%(sourceDir,i),dpi=300)
    count+=1
    if count%10==0 or count==len(sD):
        #count=0
        resD = {}
        xMin = 999999
        yMin = 999999
        xMax=0
        yMax =0
        for sourceId in resD:
            vel = resD[sourceId]
            xMin= min(xMin,np.array(vel.xm).min())
            xMax= max(xMax,np.array(vel.xm).max())
            yMin= min(yMin,np.array(vel.ym).min())
            yMax= max(yMax,np.array(vel.ym).max())

        xMin = np.round(xMin-1)
        yMin = np.round(yMin-1)
        xMax = np.round(xMax+1)
        yMax = np.round(yMax+1)
        delta = 0.4
        xL=np.arange(xMin,xMax+0.0001,delta)
        yL=np.arange(yMin,yMax+0.0001,delta)
        denseXL=np.arange(xMin,xMax+0.0001,delta/10)
        denseYL=np.arange(yMin,yMax+0.0001,delta/10)
        vFinal = np.zeros([len(yL),len(xL),len(tL)])
        velL = [[[]for x in xL]for y in yL]
        for sourceId in resD:
            vel = resD[sourceId]
            ix = np.round((np.array(vel.xm)-xMin)/delta).astype(np.int)
            iy = np.round((np.array(vel.ym)-yMin)/delta).astype(np.int)
            for i in range(len(vel.v)):
                velL[iy[i]][ix[i]].append(vel.v[i])
        XMIN=99999
        XMAX =0
        YMIN=99999
        YMAX=0
        for i in range(len(yL)):
            print(i)
            for j in range(len(xL)):
                tmpL=np.array(velL[i][j])
                tmpL[tmpL<0.5]=0
                if len(tmpL)>5:
                    vsum = tmpL.sum(axis=0)
                    vFinal[i][j]=VL[vsum.argmax(axis=0)]
                    XMIN = min(XMIN,xL[j])
                    XMAX = max(XMAX,xL[j])
                    YMIN = min(YMIN,yL[i])
                    YMAX = max(YMAX,yL[i])
                else:
                    vFinal[i][j]=np.nan
        vFinalDense=np.concatenate([interpolate.interp2d(xL,yL,vFinal[:,:,i])(denseXL,denseYL).reshape([len(denseYL),len(denseXL),1])for i in range(len(tL))],axis=-1,)
        for i in range(len(tL)):
            plt.close()
            VFinal=vFinal[:,:,i]
            #VFinal[np.isnan(VFinal)]=VFinal[np.isnan(VFinal)==False].mean()
            #VFinal.set_mask()
            std=VFinal[np.isnan(VFinal)==False].std()
            mean=VFinal[np.isnan(VFinal)==False].mean()
            #plt.pcolormesh(xL,yL,VFinal,cmap='seismic',shading='gouraud',vmax=mean+std*5,vmin=mean-std*5)
            plt.pcolormesh(xL,yL,VFinal,cmap='seismic',vmax=mean+std*5,vmin=mean-std*5)
            plt.colorbar(label='vel(km/s)')
            plt.xlabel('X(km)')
            plt.ylabel('Y(km)')
            plt.title('phaseVel:%.3f(Hz)'%(1/tL[i]))
            plt.xlim([XMIN,XMAX])
            plt.ylim([YMIN,YMAX])
            plt.savefig('%s/%d_phaseVel_%.3f(Hz).jpg'%(resDir,count,1/tL[i]),dpi=300)
            plt.close()

plt.pcolor       

file = 'jiangyr/shot_ikey20.segy'
st = read(file,unpack_trace_headers=True)
reload(tool)
#sD = tool.decompose(st)
g= tool.toGather(st)
p = g.pair()
p.plot()

v =tool.Pair.toV(p,model) 
V = v[:,:,0,:].sum(axis=0)
V = V/V.max(axis=0)
plt.close()
plt.pcolor(1/tL,vL,V[:,:],vmin=0,vmax=1,cmap='jet')
plt.savefig('plot/v.jpg',dpi=300)
plt.close()
plt.plot(p.xm,p.ym,'.k')
plt.savefig('plot/mid.jpg',dpi=300)
exit()
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'wave' in layer.name:
        w = layer.weights[0].numpy()[:,0,0,0]
        plt.close()
        plt.subplot(2,1,1)
        plt.title(layer.name)
        plt.plot(w,'k')
        plt.xlim([0,900])
        plt.subplot(2,1,2)
        plt.plot(np.fft.fftfreq(len(w),d=0.5)[:int(len(w/2))-3],np.abs(np.fft.fft(w))[:int(len(w/2))-3],'k')
        plt.xlim([0,1])
        plt.savefig('plot/'+layer.name+'.jpg',dpi=300)

exit()
import fcn
from obspy import read
from glob import glob
import os
mul = 4
dirL = ['/autofs/2022pai/2022_test_code/2022Pai30/02_SuFaceWave_Supression/INPUT/2022PGD/Shot_sgy/']
#dirL = ['/home/jiangyr/']
for Dir in dirL:
    newDir = Dir.split('/')[-2]+'/'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    for file in glob(Dir+'*31.sgy'):
        st = read(file,unpack_trace_headers=True)
        newFile = newDir + os.path.basename(file)
        st.filter('bandpass',freqmin=125/160,freqmax=125/8,corners=4,zerophase=True)
        st.decimate(mul)
        st.integrate()
        st.integrate()
        st.filter('bandpass',freqmin=125/160,freqmax=125/8,corners=4,zerophase=True)
        for ST in st:
            ST.data=ST.data.astype('float32')
        st.write(newFile, format='SEGY')

