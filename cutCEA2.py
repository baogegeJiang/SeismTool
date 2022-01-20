from SeismTool.io import seism
import sys
import os
N= int(sys.argv[1])
n= int(sys.argv[2])
kind = int(sys.argv[3])
print(N,n)
para={\
'delta0' :1,
'freq'   :[-1,-1],#[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e19,
}
para0= {\
'delta0'    :0.02,\
'freq'      :[-1, -1],\
'filterName':'bandpass',\
'corners'   :2,\
'zerophase' :True,\
'maxA'      :1e5,\
}
sacPara = {'pre_filt': (1/500, 1/350, 1/2, 1/1.5),\
'output':'DISP','freq':[-1,-1],\
'filterName':'bandpass',\
'corners':4,'toDisp':False,\
'zerophase':True,'maxA':1e15}
para0.update(sacPara)

quakes = seism.QuakeL('CEA_quakesAll')
stations = seism.StationList('../stations/CEA.sta_labeled_sort')#'../stations/CEA.sta_know_few'#../stations/CEA.sta_labeled_sort
print(len(stations))
stations.getInventory()
if kind ==0:
    quakes.cutSac(stations[n::N],bTime=-1500,eTime =12300,\
    para=para,byRecord=False,isSkip=True,resDir='/media/jiangyr/1TSSD/eventSac/')
if kind ==1:
    for quake in quakes:
        print(quake)
        quake.getSacFiles(stations[n::N],isRead = True,strL='Z',\
        byRecord=False,minDist=0,maxDist=1e10,\
        remove_resp=True,para=para0,isSave=True,isSkip=False,isDoneSkip=False,\
        resDir = '/media/jiangyr/1TSSD/eventSac/')