from obspy import read,UTCDatetime
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
    stream = obspy.core.Stream()
    with open(staDir+'DATAFILE.LST') as f:
        for line in f.readlines():
            file = staDir+(line.split()[3][2:])
            stream.append(read(file))
    steam=stream.merge()
    for i in range(3):
        trace = stream[i]
        t0 =int(trace.stats['starttime'].timestamp/86400)*86400
        for t in np.arange(t0,trace.stats['endtime'].timestamp+1,86400):
            tmp = trace.slice(max(t,trace.stats['starttime'].timestamp),min(t+86400,trace.stats['endtime'].timestamp))
            saveFileName = 
