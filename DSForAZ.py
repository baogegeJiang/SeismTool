from re import U
from SeismTool.io import seism
import obspy
import numpy as np
import os
staFile = '../stations/STALST_HIMA'
R=[41.5,43.5,108,112]
stations = seism.StationList(staFile)
stations.inR(R)
stations.write('../stations/himaIn41.5,43.5,108,112_2')
time0=obspy.UTCDateTime(2016,1,1)
time1=obspy.UTCDateTime(2017,2,20)
saveDir = 'az2/'
compL = ['3','2','1']
stations.set('nameMode','hima23')
for sta in stations:
    staDir = '%s/%s.%s/'%(saveDir,sta['net'],sta['sta'])
    if os.path.exists(staDir):
        continue
    for time in np.arange(time0.timestamp,time1.timestamp,86400):
        print(obspy.UTCDateTime(time))
        sacFilesL = sta.getFileNames(time)
        for i in range(3):
            comp = compL[i]
            if len(sacFilesL[i])>0:
                tmp =seism.mergeSacByName(sacFilesL[i], delta0=0.01,isDe=False)
                tmp.decimate(10)
                tmp.decimate(10)
                tmpName='%s/%s.%s.sac'%(staDir,obspy.UTCDateTime(time).strftime('%Y%m%d'),comp)
                if not os.path.exists(os.path.dirname(tmpName)):
                    os.makedirs(os.path.dirname(tmpName))
                tmp.write(tmpName,format='SAC')
                print(tmpName,'done')
