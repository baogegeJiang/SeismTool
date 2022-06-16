#/net/hima_3/hima32/南京大学/201705-201711/15712/2017196/B690/1
from obspy import read
from glob import glob
import os
import sys
from SeismTool.io import seism
saveDir = '/HOME/jiangyr/hima3_sac/'
doIndex= int(sys.argv[1])
staFile = '/HOME/jiangyr/hima3_sac/staLst_hima3%dNew2'%doIndex
logFile = '/HOME/jiangyr/hima3_sac/convert_hima3%dNew2.log'%doIndex
staD  = {}
staInfos = seism.StationList()
count =0

hima3L=[['/net/hima_3/hima31/内蒙局/','/net/hima_3/hima31/山西局/','/net/hima_3/hima31/河北局/'],['/net/hima_3/hima32/南京大学/','/net/hima_3/hima32/陕西局/','/net/hima_3/hima32/博来银赛/'],['/net/hima_3/hima33/优赛/','/net/hima_3/hima33/地质所/','/net/hima_3/hima33/物探中心/','/net/hima_3/hima33/优赛/陕西局/']]
with open(logFile,'w+') as f:
    f.write('staFile %s\n'%staFile)
    for oDir in hima3L[doIndex-1]:
        for timeDir in glob(oDir+'/*2????????????/'):
            for staDir in glob(timeDir+'/?????/')+glob(timeDir+'/??????/'):
                staName = staDir.split('/')[-2]
                if staName[0]=='L':
                    staName = staName[1:]
                dayDirL = glob(staDir+'/2??????/')
                if len(dayDirL)<5:
                    dayDirL = glob(staDir+'/*/2??????/')
                for dayDir in dayDirL:
                    dayName = dayDir.split('/')[-2]
                    sacDir = '%s/%s/%s/'%(saveDir,staName,dayName)
                    if not os.path.exists(sacDir):
                        os.makedirs(sacDir)
                    if count%1000==0:
                        print(sacDir)
                    for hourFile in glob(dayDir+'/*/1/*'):
                        try:
                            hour     = os.path.basename(hourFile).split('_')[0]
                            fileName = '%s/%s.?.SAC'%(sacDir,hour)
                            if len(glob(fileName))==3:
                                print(fileName+' already done')
                                f.write(fileName+' already done\n')
                                continue
                            refs = read(hourFile)
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
                                    staInfos.append(seism.Station(net='hima3',sta=staName,la=la,lo=lo,dep=dep,nameMode='hima3'))
                                    staInfos.write(staFile)
                        except:
                            f.write('not_Done %s\n'%hourFile)
                        else:
                            f.write('done %s\n'%hourFile)