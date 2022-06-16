from SeismTool.io import seism 
sL1 = seism.StationList('../stations/FWH.sta')
#../stations/STALST_HIMA23WithNameMode
#fielNew='/HOME/jiangyr/hima3_sac/STALST_HIMA3_correct'
sL0 = seism.StationList('/HOME/jiangyr/hima3_sac/STALST_HIMA3')

sL1 = seism.StationList('../stations/himaFinalWithSensorDasCheck.txt')
#../stations/STALST_HIMA23WithNameMode
#fielNew='/HOME/jiangyr/hima3_sac/STALST_HIMA3_correct'
#sL0 = seism.StationList('../stations/STALST_HIMA')

for s0 in sL0:
    sta =s0['sta']
    s1 = sL1.find(sta=sta)
    if s1==None:
        continue
    s0['la']=int(s0['la'])+(s0['la']%1)/0.6
    s0['lo']=int(s0['lo'])+(s0['lo']%1)/0.6
    if s0.dist(s1)>0.1:
        print(s0)
        print(s1)
        print(s0.dist(s1))

#sL1.write(fielNew)