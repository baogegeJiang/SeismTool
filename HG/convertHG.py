from obspy import read,UTCDateTime
import obspy
from glob import glob
import os
import numpy as np
sacDir ='./'
saveDir = './sac/'
staL = ['PKU21']

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

for sta in staL:
    staDir  =  sacDir+sta+'/'
    saveStaDir  =  saveDir+sta+'/'
    if not os.path.exists(saveStaDir):
        os.makedirs(saveStaDir)
    streamD={'BH'+ comp :obspy.core.Stream() for comp in 'NEZ'}
    with open(staDir+'DATAFILE.LST') as f:
        for line in f.readlines():
            file = staDir+(line.split()[3][2:])
            stream = read(file)
            for trace in stream:
                streamD[trace.stats['channel']].append(trace)
    for comp in streamD:
        stream = streamD[comp]
        trace = stream[0]
        for tmp in stream[1:]:
            trace = trace.__add__(tmp, method=1, interpolation_samples=0, fill_value=0)
        t0 =int(trace.stats['starttime'].timestamp/86400)*86400
        print(t0, trace)
        for t in np.arange(t0,trace.stats['endtime'].timestamp+1,86400):
            tmp = trace.slice(UTCDateTime(max(t,trace.stats['starttime'].timestamp)),UTCDateTime(min(t+86400,trace.stats['endtime'].timestamp)))
            saveFileName = '%s/%s.%s.sac'%(saveStaDir,tmp.stats['starttime'].strftime('%Y%m%d%H%M'),tmp.stats['channel'])
            #print(saveFileName,tmp
            # )
            print(tmp)
            tmp.write(saveFileName)
