import os
import sys
import obspy
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.inventory import Inventory
from obspy import Stream
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")
from obspy.geodetics import base
from obspy import read, read_inventory
from obspy.io.sac import SACTrace
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#os.chdir("/Users/tongzhou/bp/coherence_measure/")
curdir = os.path.abspath(".")
print(curdir)

client_list = ["GEONET"]
client = Client("ISC")

min_lon = 177               
max_lon = 180
min_lat = -34
max_lat = -14
# search in center and radius
clat = -37.5628
clon = -179.4443
min_rad = 0
max_rad = 12
min_dep = 0
max_dep = 1000
min_mag = 8.0
max_mag = 10.0
time_start = UTCDateTime("2021-03-04T00:00:00")
timeL    = np.arange(-5,5)*86400+time_start.timestamp
time_end = UTCDateTime("2021-03-05T00:00:00")
catalog = "ISC"

client_list = ["GEONET"]
# network_list
net_list = ["*"]
# channel list
channel_list = ["*"]
# station search region (by rectangle)
sta_maxlat = 55
sta_minlat = 20
sta_maxlon = -95
sta_minlon = -125
# station search region (by circle)
center_lat = -37.5628
center_lon = -179.4443
#center_lon = -110  # west US 
sta_radius = 10
# station search time span
time_before = 0
time_after = 86400
# record time span for download (seconds before and after P arrival)
record_s = 300
record_e = 600


# loop for events
for time in timeL:
    inv = Inventory('','')
    origin_time = UTCDateTime(time)
    for icl in client_list:
        client = Client("GEONET")
        for inet in net_list:
            for ichan in channel_list:
                try:
                    inv += client.get_stations(network="*",station="*",starttime=origin_time+time_before,endtime=origin_time+time_after,
                                  latitude=center_lat,longitude=center_lon,minradius=0,maxradius=sta_radius,
                                  channel="*HZ",level="response")
                except:
                    print("no response for this network")
                    continue
    print(inv)

pre_filt = [0.005, 0.01, 10, 20]   # pre-filter for remove response # I use this to avoid ringing effect
output_resp = False
output_type = 'VEL'
resample_sr = 20
channel_list=["BHZ","BHN","HHN","EHN","LHN","BHE","HHE","EHE","LHE"]

for time in timeL:
    inv = Inventory()
    st  = Stream()
    time= UTCDateTime(time)
    YMD = time.strftime('../xinxilan2/%Y%m%d/')
    if(not os.path.exists(YMD)):
        os.makedirs(YMD)
    #os.chdir(YMD)
    origin_time = time
    print(origin_time)
    for icl in client_list:
        client = Client(icl)
        for ichan in channel_list:
                
                try:
        
                    inv += client.get_stations(starttime=origin_time+time_before,endtime=origin_time+time_after,
                                  latitude=center_lat,longitude=center_lon,minradius=0,maxradius=sta_radius,
                                  channel=ichan,level="response")
                except: 
                    print("no response for this network")
                    continue
    #print(inv)
        
    # get waveforms
    for net in inv:       # for all networks
        for sta in net:   # for stations in the current network
            try: # [m] to [degree] 
                #print(rdis)
                #print(origin_dep)
                #record_time = origin_time + P_arrival[0].time
                for ichan in sta.channels:
                    st += client.get_waveforms(net.code,sta.code, "*", ichan.code,origin_time+time_before, origin_time+time_after) # get waveform
                    print("st")
                    print(st)
                print(origin_time,record_time)
            except:
                pass

            # remove instrument response
            for tr in st:
                try:
                    tr.remove_response(inventory=inv, pre_filt=pre_filt, output=output_type,
                               water_level=60, plot=False)
                except:
                    print("remove response failed, remove this trace ...")
                    st.remove(tr)
                    pass
                # resample
            try:
                # tr.resample(resample_sr)
                #if 1:#try:
                tr=st.merge()[0]
                coordinate=inv.get_coordinates(tr.id)
#                 if(tr.id.split(".")[3][-1]=='E'):
#                     cmpaz = 90
#                     cmpinc = 90
#                 elif(tr.id.split(".")[3][-1]=='N'):
#                     cmpaz = 0
#                     cmpinc = 90
#                 else:
#                     cmpaz = 0
#                     cmpinc = 0
                cmpaz = 0
                cmpinc = 0
                sacheader = {'kstnm': tr.id.split(".")[1], 'knetwk':tr.id.split(".")[0], 'kcmpnm': tr.id.split(".")[3],
                        'stla':coordinate["latitude"],'stlo':coordinate["longitude"],'stel':coordinate["elevation"],
                        'stdp':coordinate["local_depth"],'cmpaz':cmpaz,'cmpinc':cmpinc,
                        'nzyear':origin_time.year,'nzjday':origin_time.julday,'nzhour':origin_time.hour,
                        'nzmin':origin_time.minute,'nzsec':origin_time.second,'nzmsec':origin_time.microsecond/1000,
                        'lcalda':True,'delta':tr.stats.delta,'iztype':'ia','a':record_s,'ka':"P"}
                sactr = SACTrace(data=tr.data,**sacheader)
                print("writing trace to sac "+YMD+" "+tr.id)
                sactr.write(YMD+tr.id + ".SAC")
            except:
                print('can not save')
               
        
            ###  REMEBER to reset stream (waveform traces)
            st = Stream()
    
    # go to parent dir
    #os.chdir("..")