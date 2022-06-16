# -*- coding: utf-8 -*-
#/net/hima_3/hima32/南京大学/201705-201711/15712/2017196/B690/1
from obspy import read
from glob import glob
import os
import sys
from SeismTool.io import seism
saveDir = '/HOME/jiangyr/FWH_sac_final/'
doIndex= int(sys.argv[1])
staFile = '/HOME/jiangyr/FWH_sac_final/staLst_FWH2_%d'%doIndex
logFile = '/HOME/jiangyr/FWH_sac_final/convert_FWH2_%d.log'%doIndex
staD  = {}
staInfos = seism.StationList()
count =0

hima3L=[['/HOME/jiangyr/FWH/北大台站半年数据(2022.4.15)/']]
with open(logFile,'w+') as f:
    f.write('staFile %s\n'%staFile)
    for oDir in hima3L[doIndex-1]:
        for staDir in glob(oDir+'/*/'):
            staName = staDir.split('/')[-2][:5]
            for hourFile in glob(staDir+'/*HH?.mseed'):
                basename = os.path.basename(hourFile)
                dayName = basename[:8]
                sacDir = '%s/%s/%s/'%(saveDir,staName,dayName)
                try:
                    print('read',hourFile)
                    hour     = os.path.basename(hourFile).split('_')[1][:8]
                    comp = os.path.basename(hourFile).split('.')[0][-3:]
                    fileName = '%s/%s.%s.SAC'%(sacDir,hour,comp)
                    if len(glob(fileName))==1:
                        print(fileName+' already done')
                        f.write(fileName+' already done\n')
                        continue
                    sac = read(hourFile)[0]
                    print('read done',hourFile)
                    f.write('doing %s %s %s\n'%(hourFile,comp, fileName))
                    if not os.path.exists(os.path.dirname(fileName)):
                        os.makedirs(os.path.dirname(fileName))
                    sac.write(fileName)
                    count += 1
                except:
                    f.write('not_Done %s\n'%hourFile)
                else:
                    f.write('done %s\n'%hourFile)

