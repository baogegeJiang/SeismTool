import obspy
from glob import glob
from SeismTool.io import seism
from matplotlib import pyplot as plt
import numpy as np
import os
tmpDir = '/media/jiangyr/CEATmp*/'
staInfoDir='../stations/CEAINFONewNew5_his/'

#FSH.2007210070000.LN.00.D.BHZ.sac
if not os.path.exists(staInfoDir):
    os.makedirs(staInfoDir)
#HI.YML. 19.8530 110.2795 0.129
for  File in glob('resp/history/station_adjust.txt'):
    with open(File) as F:
        for line in F.readlines():
            netSta,la,lo,dep=line.split()
            file = '%s/%s'%(staInfoDir,netSta[:-1])
            with open(file,'a') as f:
                for time in  np.arange(obspy.UTCDateTime(2007,1,1).timestamp,obspy.UTCDateTime(2009,7,1).timestamp,86400):
                    f.write('%s %s %s %.4f\n'%(obspy.UTCDateTime(time).strftime('%Y%m%d'),la,lo,1000*float(dep)))
if False:
    for  File in glob('resp/real-time/station_??.txt'):
        with open(File) as F:
            for line in F.readlines():
                netSta,la,lo,dep=line.split()
                file = '%s/%s'%(staInfoDir,netSta[:-1])
                with open(file,'a') as f:
                    for time in  np.arange(obspy.UTCDateTime(2009,7,1).timestamp,obspy.UTCDateTime(2013,1,1).timestamp,86400):
                        f.write('%s %s %s %.4f\n'%(obspy.UTCDateTime(time).strftime('%Y%m%d'),la,lo,1000*float(dep)))
if  False:
    dateDirs = glob('%s/????????/'%tmpDir)
    dateDirs.sort()
    for dateDir in dateDirs:
        dateStr = dateDir.split('/')[-2]
        for file in glob('%s/*Z.sac'%dateDir):
            sta,date,net,tmp,a,b,af=os.path.basename(file).split('.')
            infoFile = '%s/%s.%s'%(staInfoDir,net,sta)
            with open(infoFile,'a') as f:
                try:
                    trace = obspy.read(file,headonly=True)[0]
                except:
                    print('bad')
                else:
                    sac = trace.stats['sac']
                    f.write('%s %.4f %.4f %.4f\n'%(dateStr,sac['stla'],sac['stlo'],sac['stel']))
        print(dateStr,'done')
            
