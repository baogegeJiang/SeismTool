'''
from SeismTool.io import seism

stations  = seism.StationList('../stations/staLstNMV2SelectNewSensorDasCheck')
stations.read('../stations/STALST_HIMA',isUnique=True)

stations.getSensorDas()
stations.write('../stations/HIMA_with_SensorDas.txt','net sta compBase la lo dep sensorName dasName sensorNum')

stas=seism.StationList('../stations/staLstNMV2SelectNewSensorDasCheck')
stas.plot('test.eps')
stations.plot('testAll.eps')


STAS = seism.StationList('../stations/HIMASD.txt')
sensorL = []
for sta in STAS:
    if sta['sensorName'] not in sensorL:
        sensorL.append(sta['sensorName'])
#T37552

keyD={'CMG-3ESPC':'CMG-3ESPC', 
'CMG-3ESP':'CMG-3ESP',
'CMG-3T':'CMG-3T', 
'UNKNOWN':'UNKNOWN', 
'3ESPCO':'CMG-3ESPC', 
'3ESP':'CMG-3ESP', 
'3ESPC':'CMG-3ESPC', 
'3espc':'CMG-3ESPC', 
'3ESPO':'CMG-3ESP', 
'3ESPcO':'CMG-3ESPC',
'CMG3ESPCO':'CMG-3ESPC',
'CMG-3TG':'CMG-3T',
'CMG-3ESPCO':'CMG-3ESPC',
'CMG-3ESPCG':'CMG-3ESPC',
'CMG-3ESPG':'CMG-3ESP',
'EspcG':'CMG-3ESPC',
'3espcG':'CMG-3ESPC',
'G':'UNKNOWN',
'CMG_3ESPG':'CMG-3ESP',
'3ESPCG':'CMG-3ESPC',
'CMG-3ESPO':'CMG-3ESP',
'cmg-3ESPC':'CMG-3ESPC',
'CMG-3espcG':'CMG-3ESPC',
'CMG_3ESPCO':'CMG-3ESPC', 
'cmg-3espc':'CMG-3ESPC',
'cmg-3TG':'CMG-3T',
'Cmg-3tG':'CMG-3T',
'Guralp3TG':'CMG-3T'}

for sta in STAS:
    sta['sensorName']=keyD[sta['sensorName']]

STAS.write('../stations/HIMASDADJUST.txt','net sta compBase la lo dep sensorName dasName sensorNum')

from eqcorrscan.utils import seismo_logs
from eqcorrscan import utils

rt2ms -d . -X
'''
'''
import os
from glob import glob
saveDir = '/HOME/jiangyr/hima3_sac/'
hima3L=[['/net/hima_3/hima31/内蒙局/','/net/hima_3/hima31/山西局/','/net/hima_3/hima31/河北局/'],['/net/hima_3/hima32/南京大学/','/net/hima_3/hima32/陕西局/','/net/hima_3/hima32/博来银赛/'],['/net/hima_3/hima33/优赛/','/net/hima_3/hima33/地质所/','/net/hima_3/hima33/物探中心/','/net/hima_3/hima33/优赛/陕西局/']]
runDir = '/home/jiangyr/testRT/'
saveDir = '/HOME/jiangyr/hima3_sac/'
staDone =[]
staDoneFile = 'done2.Lst'
with open(staDoneFile,'w+') as F:
    for hima3 in  hima3L:
        for oDir in  hima3:
            for timeDir in glob(oDir+'/*2????????????/'):
                for staDir in glob(timeDir+'/?????/')+glob(timeDir+'/??????/'):
                    staName = staDir.split('/')[-2]
                    sacDir = '%s/%s/'%(saveDir,staName)
                    if staName[0]=='L':
                        staName = staName[1:]
                    if staName in staDone:
                        continue
                    os.system('rm %s/*log'%sacDir)
                    print('doing on ',staName)
                    dayDirL = glob(staDir+'/2??????/')
                    if len(dayDirL)<5:
                        dayDirL = glob(staDir+'/*/2??????/')
                    if len(dayDirL)==5:
                        continue
                    count=0
                    for dayDir in dayDirL:
                        count+=1
                        os.system('rm -r %s'%runDir)
                        os.makedirs(runDir)
                        os.system('cp -r %s %s'%(dayDir,runDir))
                        os.system('cd %s;rt2ms -d . -X'%runDir)
                        logFiles = glob(runDir+'LOGS/*log')
                        if len(logFiles)>0:
                            F.write('%s:\n'%staName)
                            for logFile in logFiles:
                                sensorName = 'UNKNOWN'
                                dasName    = 'UNKNOWN'
                                sensorNum  = 'UNKNOWN'
                                F.write('%s\n'%logFile)
                                #print('cp %s %s'%(logFile,sacDir))
                                with open(logFile) as f:
                                    while sensorName=='UNKNOWN'or dasName=='UNKNOWN'or sensorNum=='UNKNOWN':
                                        line = f.readline()
                                        if line=='':
                                            break
                                        if sensorName == 'UNKNOWN':
                                            if 'Sensor Model' in line:
                                                if len(line)<=21:
                                                    tmp = 'UNKNOWN'
                                                else:
                                                    tmp = line[20:].split()
                                                    if len(tmp)>0:
                                                        tmp0 =tmp
                                                        tmp = ''
                                                        for TMP in tmp0:
                                                            tmp+=TMP
                                                    else:
                                                        tmp = 'UNKNOWN'
                                                sensorName = tmp
                                                #print(line)
                                                continue
                                        if sensorNum == 'UNKNOWN':
                                            if 'Sensor Serial Number' in line:
                                                if len(line)<=29:
                                                    tmp = 'UNKNOWN'
                                                else:
                                                    tmp = line[28:].split()
                                                    if len(tmp)>0:
                                                        tmp0 =tmp
                                                        tmp = ''
                                                        for TMP in tmp0:
                                                            tmp+=TMP
                                                    else:
                                                        tmp = 'UNKNOWN'
                                                sensorNum = tmp
                                                #print(line)
                                                continue
                                        if dasName =='UNKNOWN':
                                            if 'REF TEK' in line:
                                                dasName    = line.split()[-1]
                                                #print(line)
                                                continue
                                if sensorName!='UNKNOWN':
                                    print('find',sensorName)
                                    os.system('cp %s %s'%(logFile,sacDir))
                                    staDone.append(staName)
                                if staName in staDone:
                                    break
                            if staName in staDone:
                                    break
                        if count>10:
                            print('not find')
                            break
'''

from SeismTool.io import seism,dataLib
from scipy.stats.stats import trim_mean


#stations  = seism.StationList('../stations/HIMASDADJUST.txt')
#stations.read('../stations/STALST_HIMA',isUnique=True)

#stations  = seism.StationList('../stations/STALST_HIMA',isUnique=True)
#stations.getSensorDas(isUnkown=True)
#stations.write('../stations/HIMASDADJUSTAll.txt','net sta compBase la lo dep sensorName dasName sensorNum')

stations = seism.StationList('../stations/HIMASDADJUSTAll.txt')
keyD={'CMG-3ESPC':'CMG-3ESPC', 
'CMG-3ESP':'CMG-3ESP',
'CMG-3T':'CMG-3T', 
'UNKNOWN':'UNKNOWN', 
'3ESPCO':'CMG-3ESPC', 
'3ESP':'CMG-3ESP', 
'3ESPC':'CMG-3ESPC', 
'3espc':'CMG-3ESPC', 
'3ESPO':'CMG-3ESP', 
'3ESPcO':'CMG-3ESPC',
'CMG3ESPCO':'CMG-3ESPC',
'CMG-3TG':'CMG-3T',
'CMG-3ESPCO':'CMG-3ESPC',
'CMG-3ESPCG':'CMG-3ESPC',
'CMG-3ESPG':'CMG-3ESP',
'EspcG':'CMG-3ESPC',
'3espcG':'CMG-3ESPC',
'G':'UNKNOWN',
'CMG_3ESPG':'CMG-3ESP',
'3ESPCG':'CMG-3ESPC',
'CMG-3ESPO':'CMG-3ESP',
'cmg-3ESPC':'CMG-3ESPC',
'CMG-3espcG':'CMG-3ESPC',
'CMG_3ESPCO':'CMG-3ESPC', 
'cmg-3espc':'CMG-3ESPC',
'cmg-3TG':'CMG-3T',
'Cmg-3tG':'CMG-3T',
'Guralp3TG':'CMG-3T',
'3TG':'CMG-3T',
'CMG-3TO':'CMG-3T',
'CMG-3TB':'CMG-3TB'}


findD=dataLib.loadKeyD('/home/jiangyr/Surface-Wave-Dispersion/stations/SensorNum.txt')
for sta in stations:
    sta['sensorNum'] = dataLib.sensorNumCorrect(sta['sensorNum'])
    #print(sta['sensorNum'])
    if sta['sensorNum'] in findD:
        sta['sensorName']=findD[sta['sensorNum']]
    else:
        print('notFind SN',sta['sensorName'],sta['sensorNum'])
        sta['sensorName']='UNKNOWN'
    if sta['sensorName'] in keyD:
        sta['sensorName']=keyD[sta['sensorName']]
    else: 
        print('notFind',sta['sensorName'],sta['sensorNum'])

stations.write('../stations/himaFinalWithSensorDasCheck.txt','net sta compBase la lo dep sensorName dasName sensorNum')




'''
from SeismTool.io import dataLib
dataLib.doc2txt('/home/jiangyr/Surface-Wave-Dispersion/stations/SensorNum/')
dataLib.readTXTSensorNum('/home/jiangyr/Surface-Wave-Dispersion/stations/SensorNum/','/home/jiangyr/Surface-Wave-Dispersion/stations/SensorNum.txt')

print(dataLib.loadKeyD('/home/jiangyr/Surface-Wave-Dispersion/stations/SensorNum.txt'))
'''
'''
from SeismTool.io import dataLib
import yagmail
import sys
import time
import os

#mail=yagmail.SMTP(user='', password=sys.argv[1],host='smtp.163.com')
mail=yagmail.SMTP(user='', password=sys.argv[1],host='smtp.pku.edu.cn')
contents = ['']
findD=dataLib.loadKeyD('/home/jiangyr/Surface-Wave-Dispersion/stations/SensorNum.txt')
count =0 
findL = []
for key in findD:
    key = dataLib.sensorNumCorrect(key)
    findL.append(key)
if os.path.exists('/home/jiangyr/Surface-Wave-Dispersion/stations/allSensorNumber_doneV2'):
    with open('/home/jiangyr/Surface-Wave-Dispersion/stations/allSensorNumber_doneV2','r') as f:
        for line in f.readlines():
            key = line.split()[0]
            key = dataLib.sensorNumCorrect(key)
            findL.append(key)
with open('/home/jiangyr/Surface-Wave-Dispersion/stations/allSensorNumberV2') as f:
    with open('/home/jiangyr/Surface-Wave-Dispersion/stations/allSensorNumber_doneV2','a+') as F:
        for line in f.readlines():
            key = line.split()[0]
            key = dataLib.sensorNumCorrect(key)
            count+=1
            if key not in findL:
                print(key,count)
                mail.send('caldoc@guralp.com', key, contents) 
                findL.append(key)
                F.write('%s\n'%key)
                time.sleep(1) 
'''