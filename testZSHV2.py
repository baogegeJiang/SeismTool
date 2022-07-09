from distutils.command.config import config
from dispCal  import disp
import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
from obspy import read,UTCDateTime
from SeismTool.io import seism
from SeismTool.SurfDisp import dispersion as d
from random import choice
isModel=True

if isModel:
    from SeismTool.deepLearning import node_cell,fcn
    import tensorflow as tf
    config = fcn.fcnConfig(mode='surfMul')
    inputs,output=config.inAndOut()
    print(inputs,output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss=config.lossFunc, optimizer='adam')
    print(model.summary())
#from SeismTool.deepLearning import fcn
#from tensorflow.keras import  Model
#config=fcn.fcnConfigGather()
#I,O=config.inAndOut()
#model = Model(inputs=I,outputs=O)
#model.compile(loss=config.lossFunc, optimizer='Nadam')
file = '/home/jiangyr/shot_ikey20.segy'
fs = 200
timeR = 5
freqmin= 0.8
freqmax= 15
perDeg = 111190
vs = np.array([0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.5])
thickness = np.array([50,50,100,100,200,200,200,200,400])/1e3
vp = vs*2
T = 0.1**np.arange(0,1.000001,1/49)
f = 1/T
N = fs*timeR
timeL = np.arange(N)/fs
freq = np.fft.fftfreq(N*4,d=1/fs)
freq=freq[freq>freqmin]
freq=freq[freq<freqmax]
st = read(file,unpack_trace_headers=True)
keyXG='group_coordinate_x'
keyYG='group_coordinate_y'
keyXS='source_coordinate_x'
keyYS='source_coordinate_y'
numberKey='trace_sequence_number_within_line'
sacsL =[]
sacNamesL=[]
i0=140*12
i1=140*20
for ST in st[i0:i1]:
    head = ST.stats['segy']['trace_header']
    ST.stats['_format']='segy'
    stloc =[head[keyYG]/perDeg-head[keyYS]/perDeg,head[keyXG]/perDeg-head[keyXS]/perDeg,0]
    eloc =[0,0,0]
    ST.filter('bandpass',freqmin=freqmin,freqmax=freqmax,corners=4,zerophase=True)
    ST=seism.adjust(ST,stloc=stloc,kzTime=0,kzTimeNew=None,tmpFile='test.sac',decMul=5,\
	eloc=eloc,chn=None,sta='%d'%head[numberKey],net='ZSH',o=None,pTime=None,sTime=None)
    sacsL.append([ST])
    sacNamesL.append(['ZSH'+'.%d'%head[numberKey]])
#print(head)
plt.figure(figsize=(4,4))
for i in range(10):
    plt.plot(freq,disp.calDisp(thickness*(np.random.rand(len(vp))*2+0.1),vp*(np.random.rand(len(vp))*0.08+0.98),vs*(np.random.rand(len(vp))*0.08+0.96)*(np.random.rand(len(vp))*0.08+0.96),vs*(np.random.rand(len(vp))*0.08+0.96),1/freq,flat_earth=False))
    plt.gca().set_xscale('log')
plt.savefig('predict/test.jpg',dpi=300)
disL = np.array([ sacs[0].stats['sac']['dist'] for sacs in sacsL])
indexL = disL.argsort()
data = np.zeros([len(indexL),N])
data0 = np.zeros([len(indexL),N])
dis  = np.zeros(len(indexL))
az  = np.zeros(len(indexL))
for i in range(len(indexL)):
    index = indexL[i]
    sac = sacsL[index][0]
    data[i,:]=sac.data[:N]/np.abs(sac.data[:N]).max()
    data0[i,:]=sac.data[:N]/np.abs(sac.data[:N]).max()
    dis[i] = sac.stats['sac']['dist']
    time=dis[i]/2.5+0.2
    tIndex= np.abs(timeL-time).argmin()
    data[i,tIndex:]=0
    az[i] = sac.stats['sac']['az']

class Gather:
    def __init__(self,data,dis,az,fL,FL,timeL):
        self.data=data
        self.dis=dis
        self.az = az
        self.fL=fL
        self.FL = FL
        self.timeL = timeL
    def __call__(self,i0=0,N=30,az0=0,az1=360,fv=None,n0=5,isFv=True,ratio=0.25):
        data = []
        az = []
        dis = []
        index = i0+0 
        data.append(self.data[index].reshape([-1,1]))
        az.append(self.az[index])
        dis.append(self.dis[index])
        dis.append(self.dis[index])
        N=N-1
        for i in range(1000):
            index=i+1
            if self.az[index]<az0 or self.az[index]>az1 or self.dis[index]<0.5:
                continue
            if np.random.rand()>ratio:
                continue
            N=N-1
            data.append(self.data[index].reshape([-1,1]))
            dis.append(self.dis[index])
            az.append(self.az[index])
            if N==0:
                break
        if not isinstance(fv,type(None)):
            v = fv(self.fL).reshape([1,-1])
            fL=self.fL.reshape([1,-1])
            V = fv(self.FL)
            timeL =self.timeL.reshape([-1,1])
            phi0=np.random.rand()*2*np.pi*0
            pi=np.pi
            mask=np.arange(len(data))
            mask = mask>n0
            if isFv:
                t0=(np.random.rand()*1-0.5)*0.5
                for i in range(len(data)):
                    dist = dis[i]
                    dt = dist/v 
                    f0 = choice(fL[0])
                    sigma=(np.random.rand()*6+2)
                    al = np.exp(-(fL-f0)**2/sigma**2)
                    surf = ne.evaluate('al*cos(phi0+fL*pi*2*(dt-timeL+t0))').sum(axis=1)
                    surf = surf/surf.max()*(np.random.rand()*0.8+0.6)
                    data[i][:,0]+= surf
            return np.array(data),np.array(dis).reshape([-1,1]),np.array(az),V.reshape([1,-1]),mask
        return np.array(data),np.array(dis).reshape([-1,1]),np.array(az)
    def getSample(self,n=4,fvL=[],az0=0,az1=360,N=30,isFv=True,ratio=0.25):
        dataL=[]
        disL =[]
        azL =[]
        vL =[]
        maskL=[]
        for i in range(n):
            fv = choice(fvL)
            i0=int(len(self.data)*np.random.rand()/2)
            data,dis,az,v,mask=self(i0=i0,az0=az0,az1=az1,fv=fv,N=N,isFv=isFv,ratio=ratio)
            dataL.append(data)
            disL.append(dis)
            azL.append(az)
            vL.append(v)
            maskL.append(mask)
        return np.array(dataL),np.array(disL),np.array(azL),np.array(vL),np.array(maskL)
            
        
fvL=[d.fv([freq,disp.calDisp(thickness*(np.random.rand(len(vp))*2+0.1),vp*(np.random.rand(len(vp))*0.08+0.98),vs*(np.random.rand(len(vp))*0.08+0.96)*(np.random.rand(len(vp))*0.08+0.96),vs*(np.random.rand(len(vp))*0.08+0.96),1/freq,flat_earth=False)])for i in range(2000)]
g =Gather(data,dis,az,freq,f,timeL)
g0 =Gather(data0,dis,az,freq,f,timeL)
DATAL0,DISL0,AZL0,VL,maskL= g0.getSample(az0=260,az1=280,fvL=fvL,n=10,N=20,isFv=False,ratio=0.5)
vL = np.arange(0.5,2,0.005)
specL = np.array([d.calSpec(DATAL0[0,i,g0.timeL>DISL0[0,i,0]/2-0.5],g0.timeL[g0.timeL>DISL0[0,i,0]/2-0.5],f) for i in range(len(DATAL0[0]))])
dt = DISL0[0,1:].reshape([1,-1])/vL.reshape([-1,1])
E =(specL.reshape([1,len(DATAL0[0]),-1])*np.exp(1j*np.pi*2*dt.reshape([len(vL),-1,1])*f.reshape([1,1,-1]))).sum(axis=1)
A =np.abs(E)
v0=vL[A.argmax(axis=0)]
plt.figure(figsize=(4,4))
plt.pcolor(f,vL,A/A.max(axis=0,keepdims=True),cmap='jet')
plt.ylabel('v(km/s)')
plt.xlabel('f (Hz)')
plt.colorbar()
plt.gca().set_xscale('log')
plt.savefig('predict/test.jpg',dpi=300)
from tensorflow.keras import backend as K
#DATA,DIS,AZ,V,mask= g(az0=80,az1=100,fv=fvL[0])
K.set_value(model.optimizer.lr, 1e-3)
for i in range(100):
    DATAL,DISL,AZL,VL,maskL= g.getSample(az0=1,az1=360,fvL=fvL,n=300,N=20,ratio=0.2)
    specL = np.array([d.calSpec(DATAL[0,i,g.timeL>DISL0[0,i,0]/2.5],g.timeL[g.timeL>DISL0[0,i,0]/2.5],f) for i in range(len(DATAL[0]))])
    dt = DISL[0,1:].reshape([1,-1])/vL.reshape([-1,1])
    E =(specL.reshape([1,len(DATAL[0]),-1])*np.exp(1j*np.pi*2*dt.reshape([len(vL),-1,1])*f.reshape([1,1,-1]))).sum(axis=1)
    A =np.abs(E)
    v0_=vL[A.argmax(axis=0)]
    out= model.predict([DATAL,np.diff(DISL,axis=1)],batch_size=16)
    out0= model.predict([DATAL0,np.diff(DISL0,axis=1)],batch_size=16)
    print(out.shape)
    print(out[-1,0,::15],VL[-1,0,::15],v0_[::15],out0[1,0,::15],v0[::15])
    #print(DATAL.shape, DISL.shape,maskL.shape,VL.shape)np.diff(DISL,axis=1)
    model.fit([DATAL,np.diff(DISL,axis=1)],VL,batch_size=16)
    K.set_value(model.optimizer.lr, 1e-2*0.95**i)
    print(1e-2*0.95**i)
    
exit()
plt.figure(figsize=(4,4))
#boolL = (az>260)*(az<280)
plt.pcolor(timeL,DIS,DATA,cmap='bwr',vmax=1.5,vmin=-1.5)

plt.savefig('predict/test.jpg',dpi=300)

exit()

boolL = (az>260)*(az<280)
plt.pcolor(timeL,dis[boolL],data[boolL],cmap='bwr')
plt.plot(dis[boolL]/1.8+0.2,dis[boolL],'k')
plt.plot(dis[boolL]/0.8+0.2,dis[boolL],'k')
plt.savefig('predict/test.jpg',dpi=300)
print(disp.calDisp(thickness,vp,vs,vs,T,flat_earth=False))
plt.figure(figsize=(4,4))
for i in range(1000):
    plt.plot(f,disp.calDisp(thickness*(np.random.rand(len(vp))*2+0.1),vp*(np.random.rand(len(vp))*0.08+0.98),vs*(np.random.rand(len(vp))*0.08+0.96)*(np.random.rand(len(vp))*0.08+0.96),vs*(np.random.rand(len(vp))*0.08+0.96),1/freq,flat_earth=False))
    plt.gca().set_xscale('log')
plt.savefig('predict/test.jpg',dpi=300)




exit()
a =  d.Gather(modeN=2)
data,y=a.generate()
plt.close()
plt.figure(figsize=(4,8))
plt.subplot(2,1,1)
plt.pcolor(a.xL,a.timeL,np.sign(data)*np.abs(data)**0.5,cmap='bwr')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.ylim([15,-1])
plt.subplot(2,1,2)
plt.pcolor(a.xL,a.timeL,y[:,:,:].max(axis=2),cmap='hot')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.ylim([15,-1])
plt.savefig('predict/test.jpg',dpi=400)
print(a.NX,a.NT)
exit()


t0 = -2
dt = 1/200
DT = 12
dx = 0.01
DX  = 2
TL = 10**np.arange(0,1.0001,1/10)
fL = 1/TL
modeN=3

NT = int(DT/dt)
timeL = np.arange(NT)*dt+t0
FL = np.fft.fftfreq(NT,dt)[::-1]
FL=FL[FL<=2/1]
FL=FL[FL>=1/20]
NX = int(DX/dx)
xL   = np.arange(NX)*dx

sigmaL = np.array([100,200,300,400,500,600,600,600])/1e3
midL    = np.array([1,1.1,1.2,1,0.8,0.7,0.6,0.5])
aL =  np.array([0.6,0.5,0.5,0.5,0.4,0.4,0.3,0.3])
thicknessL = np.array([0.1,0.2,0.4,0.4,0.4,0.6,0.6,0.4]).reshape([1,-1])*\
    (1+aL.reshape([1,-1])*np.exp(-(xL.reshape([-1,1])-midL.reshape([1,-1]))**2/sigmaL.reshape([1,-1])**2))
vs = np.array([0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.8])
vp = np.array([0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.8])*1.7
rho = vs*2
depL  = np.cumsum(thicknessL,axis=1)-0.5*thicknessL 

plt.close()
for i in range(8):
    plt.plot(xL,depL[:,i])
plt.savefig('predict/test.jpg',dpi=300)

DISPM = np.array([[calDisp(thicknessL[i], vp, vs,rho, 1/FL,mode=j+1, velocity='phase', flat_earth=True,wave='rayleigh') for i in range(NX)]for j in range(modeN)])

DTM = dx/DISPM
TM = np.cumsum(DTM,axis=1)-DTM[:,0:1,:]

dispM = np.array([[calDisp(thicknessL[i], vp, vs,rho, 1/fL,mode=j+1, velocity='phase', flat_earth=True,wave='rayleigh') for i in range(NX)]for j in range(modeN)])

dtM = dx/dispM
tM = np.cumsum(dtM,axis=1)-dtM[:,0:1,:]
modeIndex=0
plt.close()
for i in [0,4,9,20]:
    plt.plot(xL,TM[modeIndex,:,i])
plt.savefig('predict/test.jpg',dpi=300)
#exit()

AL = np.array([1,0.5,0.2])
data= np.zeros([NT,NX])
timeM = timeL.reshape([-1,1])
for i in range(len(FL)):
    F = FL[i]
    af = F**0.5
    for modeIndex in range(modeN):
        DT=TM[modeIndex:modeIndex+1,:,i]
        A = AL[modeIndex]
        pi=np.pi
        data+=A*ne.evaluate('cos(-pi*2*(timeM-DT)*F)*af')

plt.close()
plt.pcolor(xL,timeL,np.sign(data)*np.abs(data)**0.5,cmap='bwr')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.plot(xL,TM[0,:,20])
plt.savefig('predict/test.jpg',dpi=300)
y= np.zeros([NT,NX,len(fL),modeN])
sigma=0.1
for i in range(len(fL)):
    F = fL[i]
    for modeIndex in range(modeN):
        dt=tM[modeIndex:modeIndex+1,:,i]
        pi=np.pi
        y[:,:,i,modeIndex]=A*ne.evaluate('exp(-(timeM-DT)**2/sigma**2)')
plt.close()
plt.pcolor(xL,timeL,y[:,:,0,0],cmap='hot')
plt.xlabel('x(km)')
plt.ylabel('t(s)')
plt.plot(xL,TM[0,:,20])
plt.savefig('predict/test.jpg',dpi=300)
exit()

AttribDict({'endian': '>', 
'unpacked_header': None, 
'trace_sequence_number_within_line': 1505471, 
'trace_sequence_number_within_segy_file': 3999,
 'original_field_record_number': 40011338, \
 'trace_number_within_the_original_field_record': 3999, 
 'energy_source_point_number': 3651, 
 'ensemble_number': 13103651, 
 'trace_number_within_the_ensemble': 124, 'trace_identification_code': 1, 'number_of_vertically_summed_traces_yielding_this_trace': 1, 'number_of_horizontally_stacked_traces_yielding_this_trace': 1, 
 'data_use': 1,
  'distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group': 4569, 
  'receiver_group_elevation': 1380, 'surface_elevation_at_source': 1367, 'source_depth_below_surface': 20, 'datum_elevation_at_receiver_group': 0, 'datum_elevation_at_source': 0, 'water_depth_at_source': 0, 'water_depth_at_group': 0, 'scalar_to_be_applied_to_all_elevations_and_depths': 1, 'scalar_to_be_applied_to_all_coordinates': 1, 
  'source_coordinate_x': 278838, 'source_coordinate_y': 4375663, 
  'group_coordinate_x':  282262, 'group_coordinate_y' : 4378688, 
  'coordinate_units': 1, 'weathering_velocity': 0, 'subweathering_velocity': 0, 'uphole_time_at_source_in_ms': 20, 'uphole_time_at_group_in_ms': 0, 'source_static_correction_in_ms': 27, 'group_static_correction_in_ms': 4, 'total_static_applied_in_ms': -31, 'lag_time_A': 0, 'lag_time_B': 0, 'delay_recording_time': 0, 'mute_time_start_time_in_ms': 0, 'mute_time_end_time_in_ms': 0, 'number_of_samples_in_this_trace': 5001, 'sample_interval_in_ms_for_this_trace': 1000, 'gain_type_of_field_instruments': 0, 'instrument_gain_constant': 0, 'instrument_early_or_initial_gain': 0, 'correlated': 0, 'sweep_frequency_at_start': 0, 'sweep_frequency_at_end': 0, 'sweep_length_in_ms': 0, 'sweep_type': 0, 'sweep_trace_taper_length_at_start_in_ms': 0, 'sweep_trace_taper_length_at_end_in_ms': 0, 'taper_type': 0, 'alias_filter_frequency': 0, 'alias_filter_slope': 0, 'notch_filter_frequency': 0, 'notch_filter_slope': 0, 'low_cut_frequency': 0, 'high_cut_frequency': 0, 'low_cut_slope': 0, 'high_cut_slope': 0, 'year_data_recorded': 0, 'day_of_year': 0, 'hour_of_day': 0, 'minute_of_hour': 0, 'second_of_minute': 0, 'time_basis_code': 0, 'trace_weighting_factor': 0, 'geophone_group_number_of_roll_switch_position_one': 0, 'geophone_group_number_of_trace_number_one': 0, 'geophone_group_number_of_last_trace': 0, 'gap_size': 0, 'over_travel_associated_with_taper': 0, 'x_coordinate_of_ensemble_position_of_this_trace': 0, 'y_coordinate_of_ensemble_position_of_this_trace': 0, 'for_3d_poststack_data_this_field_is_for_in_line_number': 0, 'for_3d_poststack_data_this_field_is_for_cross_line_number': 0, 'shotpoint_number': 0, 'scalar_to_be_applied_to_the_shotpoint_number': 0, 'trace_value_measurement_unit': 0, 'transduction_constant_mantissa': 0, 'transduction_constant_exponent': 0, 'transduction_units': 0, 'device_trace_identifier': 0, 'scalar_to_be_applied_to_times': 0, 'source_type_orientation': 0, 'source_energy_direction_mantissa': 0, 'source_energy_direction_exponent': 0, 'source_measurement_mantissa': 0, 'source_measurement_exponent': 0, 'source_measurement_unit': 0, 'unassigned': b'\x00\x00\x00\x00\x00\x00\x00\x00'})