# -*- coding: utf-8 -*-
#/net/hima_3/hima32/南京大学/201705-201711/15712/2017196/B690/1
from obspy import read
from glob import glob
import os
import sys
from SeismTool.io import seism
saveDir = '/HOME/jiangyr/FWH_sac_final/'
doIndex= int(sys.argv[1])
staFile = '/HOME/jiangyr/FWH_sac_final/staLst_FWH_%d_final'%doIndex
logFile = '/HOME/jiangyr/FWH_sac_final/convert_FWH_%d.log'%doIndex
staD  = {}
staInfos = seism.StationList()
count =0

hima3L=[['/HOME/jiangyr/FWH/北大台站半年数据(2022.4.15)/']]
with open(logFile,'w+') as f:
    f.write('staFile %s\n'%staFile)
    for oDir in hima3L[doIndex-1]:
        for staDir in glob(oDir+'/*/'):
            staName = staDir.split('/')[-2][:5]
            if staName in staD: continue
            dayDirL = glob(staDir+'/2??????/')
            if len(dayDirL)<5 and False:
                dayDirL = glob(staDir+'/*/2??????/')
            for dayDir in dayDirL:
                if staName in staD: continue
                dayName = dayDir.split('/')[-2]
                sacDir = '%s/%s/%s/'%(saveDir,staName,dayName)
                if not os.path.exists(sacDir):
                    os.makedirs(sacDir)
                if count%1000==0:
                    print(sacDir)
                for hourFile in glob(dayDir+'/*/1/*'):
                    if staName in staD: continue
                    try:
                        print('read',hourFile)
                        hour     = os.path.basename(hourFile).split('_')[0]
                        fileName = '%s/%s.?.SAC'%(sacDir,hour)
                        if len(glob(fileName))==3 and False:
                            pass
                            print(fileName+' already done')
                            f.write(fileName+' already done\n')
                            continue
                        refs = read(hourFile)
                        print('read done',hourFile)
                        for ref in  refs:
                            fileName = '%s/%s.%d.SAC'%(sacDir,hour,ref.stats['reftek130']['channel_number'])
                            f.write('doing %s %s %s\n'%(hourFile,ref.stats['reftek130']['channel_number'], fileName))
                            ref.write(fileName)
                            count += 1
                            if staName not in staD:
                                posStr = ref.stats['reftek130']['position']
                                #['N 3928.461E10726.466+01385']
                                la =float(posStr[1:4])+ float(posStr[4:10])/60
                                lo =float(posStr[11:14])+ float(posStr[14:20])/60
                                dep =  float(posStr[-6:])
                                if posStr[0]=='S':
                                    la*=-1
                                if posStr[10]=='S':
                                    lo*=-1
                                staD[staName] = 1
                                staInfos.append(seism.Station(net='FWH',sta=staName,la=la,lo=lo,dep=dep,nameMode='FWH'))
                                staInfos.write(staFile)
                                print('save',staFile)
                    except:
                        f.write('not_Done %s\n'%hourFile)
                    else:
                        f.write('done %s\n'%hourFile)

