from SeismTool.io import seism
from obspy import UTCDateTime
saveDir = '/HOME/jiangyr/GZ/'
time0 = UTCDateTime(2015,1,1)+1
stations = seism.StationList('../stations/STALST_HIMA23WithNameMode',isUnique=True)
stations.inR([34,44,108.5,115.5])
stations.plot()
stations.write('GZ.txt')
delta0=0.01
for station in stations:
    for i in range(8*365):
        time   = time0+i*86400
        filesL = station.getFileNames(time,time+86400-10)
        print(time)
        try:
            for j in range(3):
                files = filesL[j]
                comp=['BHE','BHN','BHZ'][j]
                if len(files)>0:
                    sac = seism.mergeSacByName(files,delta0=delta0)
                    sac.decimate(10) 
                    sac.decimate(10) 
                    seism.saveTrace(sac,saveDir,station['net'],station['sta'],comp=comp)
        except:
            pass
        else:
            pass
