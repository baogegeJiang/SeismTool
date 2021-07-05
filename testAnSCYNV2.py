from SeismTool.io import seism,sacTool
from SeismTool.detector import detecQuake
from SeismTool.tomoDD import tomoDD
import os
from imp import reload
import numpy as np
from SeismTool.mapTool import mapTool
from SeismTool.mathTool import mathFunc_bak
workDir='/HOME/jiangyr/detecQuake/'
staInfos = seism.StationList('../stations/SCYN_withComp_ac')
v_i='SCYNV2_2021-0608'
p_i='SCYNV2_2021-0608V1'
matDir =  workDir+'output/output%s/'%v_i
phaseLFile =  workDir+'phaseDir/phaseLst%s'%p_i
phaseLFileNew =  workDir+'phaseDir/phaseLst%s_sel'%p_i
laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]

req = {
    "maxDep":100,\
    "minCover":0.8,
    'minN' :8,
    'maxRes':3,
    'minSNR':1.5,
    'staInfos':staInfos,
    'matDir':matDir,
    'f':[2,15]
    }
req = {
    "maxDep":100,\
    "minCover":0.7,
    'minN' :5,
    'maxRes':3,
    'staInfos':staInfos,
    'matDir':matDir,
    'f':[2,15]
    }
reload(seism)
quakeL = seism.QuakeL(phaseLFile)
A=sacTool.areaMat(laL,loL,80,80)
A.insert(quakeL)
seism.printDetail=False
quakeLNew=A.select(10,req,req)
quakeLNew.set('sort','time')
quakeLNew.sort()

#quakeL.select(req)

quakeLNew.write(phaseLFileNew)
T3PSLL = [q.loadPSSacs(staInfos,matDir=matDir,f=[2,8]) for q in quakeLNew]
tomoDir = workDir+'output/output%s/tomoDD/input/'%v_i
dTM = tomoDD.calDTM(quakeLNew,T3PSLL,staInfos,minSameSta=1,maxD=0.5,minC=0.35)
tomoDD.saveDTM(dTM,workDir+'output/output%s/dTM'%v_i)
if not os.path.exists(tomoDir):
	os.makedirs(tomoDir)
tomoDD.preEvent(quakeLNew,staInfos,tomoDir+'event.dat')
tomoDD.preABS(quakeLNew,staInfos,tomoDir+'ABS.dat',isNick=False)
tomoDD.preSta(staInfos,tomoDir+'station.dat',isNick=False)
tomoDD.preDTCC(quakeLNew,staInfos,dTM,maxD=0.35,minSameSta=1,minPCC=0.35,minSCC=0.35,filename=tomoDir+'dt.cc',isNick=False,perCount=350,minDP=2/0.7)
tomoDD.preMod(laL+loL,nx=16,ny=16,nz=11,filename=tomoDir+'../inversion/MOD')

qLNew = seism.QuakeL(phaseLFileNew)
qLNew = seism.QuakeL(tomoDD.getReloc(qLNew,tomoDir+'../inversion/tomoDD.reloc'))


tomoDD.plotDistPSFreq(quakeLNew,staInfos,tomoDir+'travel.jpg')
NL=np.array([[30,102],[33,102],[33,106],[30,106],[27,104.5],[24,103],[23,100],[24,99],[27,100],[30,102]])  
mL=tomoDD.model(tomoDir+'/../inversion/',qLNew,[],isDWS=True,minDWS=2,R=laL+loL,vR=NL)
tomoDD.figType='jpg'
mL.plot(tomoDir+'/real/',['vp','vs'])
qL = seism.QuakeL(phaseLFileNew)
nameL = qLNew.paraL(['filename'])['filename']
for q in qL:
    if q['filename'] in nameL:
        qTmp = qLNew[nameL.index(q['filename'])]
        q['la']=qTmp['la']
        q['lo']=qTmp['lo']
        q['dep']=qTmp['dep']
    else:
        q['dep']=-30
        print(q)
R=[-90,90,-180,180]
tomoDD.preEvent(qL,staInfos,tomoDir+'/eventReloc.dat')


tomoDD.analyDTM4Quake(dTM,matDir+'tomoDD/quakeSel_dTM.jpg',quakeLNew,minCC=0.35,maxD=0.35)
tomoDD.analyDTM(dTM,matDir+'tomoDD/dTM.jpg')
detecQuake.plotQuakeLDis(staInfos,quakeLNew,laL,loL,filename\
          =matDir+'tomoDD/quakeSel.jpg')

detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          =matDir+'quakeSelReloc.jpg')


figDir = tomoDir+'../'
RL=[]
pL0 = np.array([[33,103],[30,106]])
RL.append(mathFunc_bak.Line(pL0,20,name='A'))
pL0 = np.array([[32,102],[29,105]])
RL.append(mathFunc_bak.Line(pL0,20,name='B'))
pL0 = np.array([[30,102],[33,106]])
RL.append(mathFunc_bak.Line(pL0,20,name='C'))
#pL0 = np.array([[30,100],[25.5,104.5]])
pL0 = np.array([[30-2.3,100],[25.5-2.3,104.5]])
RL.append(mathFunc_bak.Line(pL0,20,name='D'))
pL0 = np.array([[26.5,102],[26.5001,104.5]])
RL.append(mathFunc_bak.Line(pL0,20,name='E'))
pL0 = np.array([[28,102.7],[25,102.71]])
RL.append(mathFunc_bak.Line(pL0,20,name='F'))
pL0 = np.array([[21,103],[29,103]])
RL.append(mathFunc_bak.Line(pL0,20,name='G'))
pL0 = np.array([[23,102.75],[28,102.75]])
RL.append(mathFunc_bak.Line(pL0,20,name='H'))
#detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename=figDir+'resFig/tomoQuakeWithLine.eps',isTopo=True,rL=RL)
#detecQuake.plotQuakeLDis(staInfos,qCCLNew2,laL,loL,filename        =figDir+'resFig/ccQuakeWithLine.eps',isTopo=True,rL=RL)
#RL=[]
for r in  RL[:]:
    mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaP.jpg'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
    mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaS.jpg'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
    mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sP.jpg'%r.name,vModel=mL.vp,isPer=False,isTopo=True)
    mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sS.jpg'%r.name,vModel=mL.vs,isPer=False,isTopo=True)