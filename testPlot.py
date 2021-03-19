import numpy as np
from SeismTool.io import seism
from SeismTool.mapTool import mapTool
from SeismTool.mathTool import mathFunc_bak
from SeismTool.tomoDD import tomoDD
from SeismTool.detector import detecQuake
from obspy import UTCDateTime,read
from glob import glob
from SeismTool.plotTool import figureSet as fs
import sys
from matplotlib import pyplot as plt
import matplotlib.font_manager
templateFile = 'phase/SCYNTomoReloc'
f_0 = [0.5,20]
f_1 = [2,10]
staLstFileL=['/home/jiangyr/Surface-Wave-Dispersion/stations/XU_sel.sta','/home/jiangyr/Surface-Wave-Dispersion/stations/SCYN_withComp_ac',]
templateDir = '/media/jiangyr/MSSD/output/'
ccDir       = '/HOME/jiangyr/detecQuake/output/outputV20210106CC8_2-10/'
figDir = '../accuratePickerV4/resDir/APP++/'
staInfos=seism.StationList(staLstFileL[0])
for staLstFile in staLstFileL[1:]:
    staInfos+=seism.StationList(staLstFile)
if len(sys.argv)>1:
    doL = [ arg for arg in sys.argv[1:]]
else:
    doL =['quake','h-ml'] 

if 'quake' in doL:
    templateL = seism.QuakeL(templateFile)
    for i in range(10):
        T3L = templateL[i].loadSacs(staInfos,matDir=templateDir)
        #templateL[i].plotSacs(T3L,figDir)
        seism.Quake.plotSacs(templateL[i],T3L,figDir+'quake/',key='HBDZKX')
if 'quakeCC' in doL:
    print('loading')
    templateL = seism.QuakeL(templateFile)
    qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)
    print('load done')
    filenameL=templateL.paraL(keyL=['filename'])['filename']
    for i in range(10):
        quakeCC = qCCLNew2[i]
        T3LCC = quakeCC.loadSacs(staInfos,matDir=ccDir)
        tmpName = quakeCC['tmpName']
        template= templateL[filenameL.index(tmpName)]
        T3L = template.loadSacs(staInfos,matDir=templateDir,f=f_1)
        #templateL[i].plotSacs(T3L,figDir)
        seism.Quake.plotSacs(quakeCC,T3LCC,figDir+'quakeCC/',key='HBDZKX',quakeRef=template,T3LRef=T3L)

laL=[21,35]#area: [min latitude, max latitude]
loL=[97,109]
R= laL+loL

styleKey = 'HBDZKX'
f_1 = [2,10]
fs.init(styleKey)

if 'h-ml' in doL:
    ###load results
    qLAll = seism.QuakeL('phase/phaseSCYN')
    #qL=seism.QuakeL('phase/SCYNTOMOSort')
    #qLNew=seism.QuakeL('phase/SCYNTomoRelocV5')
    #qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)	
    #qCCLNew = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust',Quake=seism.QuakeCC)

    ##adjust magnitude##
    #qLAll.adjustMl()
    #qL.adjustMl()
    #qLNew.adjustMl()
    #qCCLNew.adjustMl()
    #qLNew.adjustMl()

    ###plot###
    ##get quakes infromation
    pL = qLAll.paraL(req={})
    timeL  = np.array(pL['time'])
    mlL  = np.array(pL['ml'])
    laL = np.array(pL['la'])
    loL = np.array(pL['lo'])
    timeL  = np.array(pL['time'])+8*3600
    timeL -=UTCDateTime(2014,1,1).timestamp
    timeL %= 86400
    timeL /=3600

    ###plot num-t
    plt.close()
    fig=plt.figure(figsize=[5,2.5])
    plt.subplot(1,2,1)
    plt.hist(timeL,np.arange(0,24,0.3))
    plt.xlabel('hour')
    plt.xlim([0,23.5])
    plt.ylabel('number')
    fs.setABC('(a)',pos=[0.05,0.98],key=styleKey)
    plt.subplot(1,2,2)
    plt.hist(mlL,np.arange(0,7,0.1),cumulative=-1,log=True)
    plt.ylabel('number')
    plt.xlabel('magnitude')
    fig.tight_layout()
    fs.setABC('(b)',pos=[0.05,0.90],key=styleKey)
    plt.savefig(figDir+'hl+ml.eps',dpi=300)
    plt.close()

if 'b' in doL:

    if 'h-ml' not in doL:
        qLAll = seism.QuakeL('phase/phaseSCYN')
        pL = qLAll.paraL(req={})
        timeL  = np.array(pL['time'])
        mlL  = np.array(pL['ml'])
        laL = np.array(pL['la'])
        loL = np.array(pL['lo'])
    loLR = np.arange(98,108,0.2)
    laLR = np.arange(21,35,0.2)
    rMax=75
    pointL=np.array([laL.tolist(),loL.tolist()]).transpose()
    rM=[[mathFunc_bak.Round([la,lo],rMax) for lo in loLR]for la in laLR]
    mlLM=mathFunc_bak.devide(rM,pointL,mlL)
    #min_mag=2.25,max_mag=4.2
    bM=np.array([[mathFunc_bak.calc_B(mlL,min_num=150,min_mag=0.8,max_mag=5)for mlL in mlLL]for mlLL in mlLM])
    mapTool.showInMap(bM[:,:,0],laLR,loLR,R,resFile=figDir+'bMap.eps',name='b',abc='(a)')
    mapTool.showInMap(bM[:,:,1],laLR,loLR,R,resFile=figDir+'cMap.eps',name='mc',abc='(b)') 

if 'allQuake' in doL:
    ###load results
    fs.init(styleKey)
    qLAll = seism.QuakeL('phase/phaseSCYN')
    detecQuake.plotQuakeLDis(staInfos,qLAll,laL,loL,filename\
          =figDir+'./allQuake.eps',isTopo=True)
tomoDir = '/HOME/jiangyr/detecQuake/output/outputVSCYNdoV40/tomoDD_bak/input/'
if 'tomoQuake' in doL:
    ###load results
    
    qL=seism.QuakeL('phase/SCYNTOMOSort')
    qLNew=seism.QuakeL('phase/SCYNTomoRelocV5')
    tomoDD.analyReloc(tomoDir+'/../inversion/tomoDD.reloc',figDir+'analyReloc.pdf')
    dTM = tomoDD.loadDTM(tomoDir+'../../dTM')
    tomoDD.analyDTM(dTM,figDir+'dTM.pdf')
    tomoDD.diff(qL,qLNew,figDir+'diff.eps')

    detecQuake.plotQuakeLDis(staInfos,qL,laL,loL,filename\
          =figDir+'./tomoQuake.eps',isTopo=True)
    detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          =figDir+'./tomoQuakeReloc.eps',isTopo=True)

if 'line' in doL:
    mL=tomoDD.model(tomoDir+'../inversion/')
    qL=seism.QuakeL('phase/SCYNTOMOSort')
    qLNew=seism.QuakeL('phase/SCYNTomoRelocV5')
    qCCLNew2 = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust_select',Quake=seism.QuakeCC)	
    #qCCLNew = seism.QuakeL('phase/phaseSCYNCC_reloc_adjust',Quake=seism.QuakeCC)
    laL=[21,35]#area: [min latitude, max latitude]
    loL=[97,109]
    RL=[]
    pL0 = np.array([[33,103],[30,106]])
    RL.append(mathFunc_bak.Line(pL0,20,name='A'))
    pL0 = np.array([[32,102],[29,105]])
    RL.append(mathFunc_bak.Line(pL0,20,name='B'))
    pL0 = np.array([[30,102],[33,106]])
    RL.append(mathFunc_bak.Line(pL0,20,name='C'))
    pL0 = np.array([[30,100],[25.5,104.5]])
    RL.append(mathFunc_bak.Line(pL0,20,name='D'))
    pL0 = np.array([[26.5,102],[26.5001,104.5]])
    RL.append(mathFunc_bak.Line(pL0,20,name='E'))
    pL0 = np.array([[28,102.7],[25,102.71]])
    RL.append(mathFunc_bak.Line(pL0,20,name='F'))
    detecQuake.plotQuakeLDis(staInfos,qLNew,laL,loL,filename\
          =figDir+'resFig/tomoQuakeWithLine.eps',isTopo=True,rL=RL)
    detecQuake.plotQuakeLDis(staInfos,qCCLNew2,laL,loL,filename\
        =figDir+'resFig/ccQuakeWithLine.eps',isTopo=True,rL=RL)
    for r in RL:
        mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaP.eps'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
        mapTool.plotDepV2(qLNew,r,figDir+'resFig/dep%sRelaS.eps'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
        mapTool.plotDepV2(qCCLNew2,r,figDir+'resFig/dep%sRelaPCC.eps'%r.name,vModel=mL.vp,isPer=True,isTopo=True)
        mapTool.plotDepV2(qCCLNew2,r,figDir+'resFig/dep%sRelaSCC.eps'%r.name,vModel=mL.vs,isPer=True,isTopo=True)
    