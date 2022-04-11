from math import degrees
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit
from sklearn import cluster
import multiprocessing
from numba import jit,float32, int64
from scipy import fftpack,interpolate,signal
import os 
from scipy import io as sio
import obspy
from multiprocessing import Process, Manager,Pool
import random
from glob import glob
from obspy.taup import TauPyModel
from tensorflow.core.framework.attr_value_pb2 import _ATTRVALUE_LISTVALUE
from ..io.seism import Dist, taup
from ..io import seism
from .fk import FK,getSourceSacName,FKL
from ..mathTool.mathFunc import getDetec,xcorrSimple,xcorrComplex,flat,validL,randomSource,disDegree,QC,fitexp
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from ..mathTool.distaz import DistAz
import gc
from matplotlib import colors,cm
import h5py
from ..plotTool import figureSet
from mathTool import mathFunc
from pysurf96 import surf96
figureSet.init()
gpdcExe = '/home/jiangyr/program/geopsy/bin/gpdc'
Vav = -1
import numexpr as ne
ne.set_num_threads(16)
class config:
    def __init__(self,originName='models/prem',srcSacDir='/home/jiangyr/home/Surface-Wave-Dispersion/',\
        distance=np.arange(400,1500,100),srcSacNum=100,delta=0.5,layerN=1000,\
        layerMode='prem',getMode = 'norm',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=150,\
        xcorrFuncL = [xcorrSimple,xcorrComplex],isFlat=False,R=6371,flatM=-2,pog='p',calMode='fast',\
        T=np.array([0.5,1,5,10,20,30,50,80,100,150,200,250,300]),threshold=0.1,expnt=10,dk=0.1,\
        fok='/k',gpdcExe=gpdcExe,order=0,minSNR=5,isCut=False,minDist=0,maxDist=1e8,\
        minDDist=0,maxDDist=1e8,para={},isFromO=False,removeP=False,doFlat=True,\
        QMul=1,modelMode='norm', convolveSrc=False,fromT=0):
        para0= {\
            'delta0'    :0.02,\
            'freq'      :[-1, -1],\
            'filterName':'bandpass',\
            'corners'   :2,\
            'zerophase' :True,\
            'maxA'      :1e5,}
        para0.update(para)
        self.originName = originName
        self.srcSacDir  = srcSacDir
        self.distance   = distance
        self.srcSacNum  = srcSacNum
        self.delta      = delta
        self.layerN     = layerN
        self.layerMode  = layerMode
        self.getMode    = getMode
        self.surfaceMode= surfaceMode
        self.nperseg    = nperseg
        self.noverlap   = noverlap
        self.halfDt     = halfDt
        self.xcorrFuncL = xcorrFuncL
        self.isFlat     = isFlat
        self.R          = R
        self.flatM      = flatM
        self.pog        = pog
        self.calMode    = calMode
        self.T          = T
        self.threshold  = threshold
        self.expnt      = expnt
        self.dk         = dk
        self.fok        = fok
        self.gpdcExe    = gpdcExe
        self.order      = order
        self.minSNR     = minSNR
        self.isCut      = isCut
        self.minDist    = minDist
        self.maxDist    = maxDist
        self.minDDist   = minDDist
        self.maxDDist   = maxDDist
        self.para0      = para0
        self.isFromO    = isFromO
        self.removeP    = removeP
        self.doFlat     = doFlat
        self.QMul       = QMul
        self.modelMode = modelMode
        self.convolveSrc = convolveSrc
        self.fromT=fromT
    def getDispL(self):
        return [disp(nperseg=self.nperseg,noverlap=self.noverlap,fs=1/self.delta,\
            halfDt=self.halfDt,xcorrFunc = xcorrFunc) for xcorrFunc in self.xcorrFuncL]
    def getModel(self, modelFile=''):
        if len(modelFile) == 0:
            modelFile = self.originName
        return model(modelFile, mode=(self.surfaceMode), getMode=(self.getMode), layerMode=(self.layerMode),
          layerN=(self.layerN),
          isFlat=(self.isFlat),
          R=(self.R),
          flatM=(self.flatM),
          pog=(self.pog),
          gpdcExe=(self.gpdcExe),
          doFlat=(self.doFlat),
          QMul=(self.QMul))

    def genModel(self, modelFile='', N=100, perD=0.1, depthMul=2):
        if len(modelFile) == 0:
            modelFile = self.originName
        model0 = np.loadtxt(modelFile)
        for i in range(N):
            model = model0.copy()
            depthLast = 0
            for j in range(model.shape[0]):
                depth0     = model[j,0]
                depth      = max(depthLast, depthLast + (depth0 - depthLast) * (1 + perD * depthMul * (2 * np.random.rand() - 1)))
                if j == 0:
                    depth = 0
                depthLast = depth
                model[(j, 0)] = depth
                for k in range(2, model.shape[1]):
                    if k < 4:
                        if j == 0:
                            d = 1
                        else:
                            d = model0[(j, k)] - model0[(j - 1, k)]
                        if j != 0:
                            model[(j, k)] = model[(j - 1, k)] + (1 + perD * (2 * np.random.rand() - 1)) * d
                        else:
                            model[(j, k)] = model[(j, k)] + (0 + perD * (2 * np.random.rand() - 1)) * d
                    else:
                        model[(j, k)] = int(model[(j, k)] * (np.random.rand() + 8) / 8.5)

                model[(j, 1)] = model[(j, 2)] * (1.7 + 2 * (np.random.rand() - 0.5) * 0.18)

            np.savetxt('%s%d' % (modelFile, i), model)
    def genFvFile(self, modelFile='', fvFile='', afStr=''):
        if len(modelFile) == 0:
            modelFile = self.originName
        if len(fvFile) == 0:
            if not self.isFlat:
                fvFile = '%s_fv' % modelFile
            else:
                fvFile = '%s_fv_flat' % modelFile
            fvFile += '_' + self.getMode
            fvFile += '_' + self.pog
            fvFile = '%s_%d' % (fvFile, self.order)
        if afStr != '':
            fvFile = '%s%s' % (fvFile, afStr)
        m = self.getModel(modelFile + afStr)
        print(m.modelFile)
        f, v = m.calDispersion(order=0, calMode=(self.calMode), threshold=(self.threshold), T=(self.T), pog=(self.pog))
        f = fv([f, v], 'num')
        print(fvFile)
        f.save(fvFile)
    def calFv(self, iL, pog=''):
        pog0 = self.pog
        if len(pog) == 0:
            pog = pog0
        self.pog = pog
        for i in iL:
            modelFile = self.getModelFileByIndex(i)
            if isinstance(modelFile, list):
                for modelFileTmp in modelFile:
                    afStr = '_/' + modelFileTmp.split('/')[(-1)]
                    print(modelFileTmp[:-len(afStr)] + afStr)
                    self.genFvFile((modelFileTmp[:-len(afStr)]), afStr=afStr)

            else:
                self.genFvFile(modelFile)

        self.pog = pog0

    def getModelFileByIndex(self, i, modelMode=''):
        if modelMode == '':
            modelMode = self.modelMode
        if modelMode == 'norm':
            return '%s%d' % (self.originName, i)
        if modelMode == 'fileP':
                return glob('%s%d_/*[0-9]' % (self.originName, i))

    def plotModelL(self, modelL):
        plt.close()
        for model in modelL:
            z, vp, vs = model.outputZV()
            plt.plot(vp, z, 'b', linewidth=0.3, alpha=0.3, label='rand_vp')
            plt.plot(vs, z, 'r', linewidth=0.3, alpha=0.3, label='rand_vp')

        z, vp, vs = self.getModel().outputZV()
        plt.title(self.originName)
        plt.gca().invert_yaxis()
        plt.xlabel('v/(m/s)')
        plt.ylabel('depth')
        plt.savefig((self.originName + '.jpg'), dpi=300)

    def plotFVL(self, fvD, pog=''):
        if len(pog) == 0:
            pog = self.pog
        plt.close()
        for key in fvD:
            FV = fvD[key]
            f = FV.f
            v = FV.v
            plt.plot(v, f, '.b', linewidth=0.3, alpha=0.3, label='rand', markersize=0.3)

        originFv = self.getFV(pog=pog)
        f = originFv.f
        v = originFv.v
        plt.plot(v, f, 'r', linewidth=2, label=(self.originName))
        plt.xlabel('v/(m/s)')
        plt.ylabel('f/Hz')
        plt.gca().semilogy()
        plt.gca().invert_yaxis()
        plt.title(self.originName + pog)
        plt.savefig(('%s_fv_%s.jpg' % (self.originName, pog)), dpi=300)

    def getFV(self, index=-1, pog=''):
        if len(pog) == 0:
            pog = self.pog
        if index == -1:
            tmpName = self.originName + '_fv'
        else:
            tmpName = '%s%d_fv' % (self.originName, index)
        if self.isFlat:
            tmpName += '_flat'
        tmpName += '_%s_%s_%d' % (self.getMode, pog, self.order)
        print(tmpName)
        return fv(tmpName, 'file')
    def quakeCorr(self, quakes, stations, byRecord=True, remove_resp=False, para={}, minSNR=-1, isLoadFv=False, fvD={}, isByQuake=False, quakesRef=[], resDir='eventSac/', maxCount=-1, **kwags):
        corrL = []
        disp = self.getDispL()[0]
        if minSNR < 0:
            minSNR = self.minSNR
        self.para0.update(para)
        count = 0
        countAll = len(quakes)
        for quake in quakes:
            print('%d of %d have done' % (count, countAll))
            count += 1
            quakeName = quake.name('_')
            if len(quakesRef) > 0:
                index = quakesRef.find(quake)
                if index >= 0:
                    quake = quakesRef[index]
                else:
                    continue
            sacsL = quake.getSacFiles(stations, isRead=True, strL='Z', byRecord=byRecord,
                minDist=(self.minDist),
                maxDist=(self.maxDist),
                remove_resp=remove_resp,
                para=(self.para0),
                isSave=False,
                isSkip=False,
                resDir=resDir,
                isCal=False)
            sacNamesL = quake.getSacFiles(stations, isRead=False, strL='Z', byRecord=byRecord,
                minDist=(self.minDist),
                maxDist=(self.maxDist),
                remove_resp=remove_resp,
                para=(self.para0),
                isSave=False,
                isSkip=False,
                resDir=resDir)
            if self.isFromO:
                for sacs in sacsL:
                    sacs[0] = seism.sacFromO(sacs[0])
                    sacs[0].data -= sacs[0].data.mean()
                    sacs[0].detrend()

            corrL += corrSacsL(disp, sacsL, sacNamesL, modelFile=self.originName, minSNR=minSNR, 
                minDist=self.minDist, maxDist=self.maxDist, minDDist=self.minDDist, 
                maxDDist=self.maxDDist, srcSac=quake.name(s='_'), 
                isCut=self.isCut, isFromO=self.isFromO, removeP=self.removeP, 
                fvD=fvD, isLoadFv=isLoadFv, quakeName=quakeName, isByQuake=isByQuake, 
                maxCount=maxCount,fromT=self.fromT, **kwags)
            print('###########', len(corrL))

        return corrL

    def modelCorr(self, count=1000, randDrop=0.3, noises=None, para={}, minSNR=-1):
        corrL = []
        disp = self.getDispL()[0]
        if minSNR < 0:
            minSNR = self.minSNR
        if isinstance(count,int):
            iL = range(count)
        else:
            iL = count
        for i in iL:
            modelFile = self.getModelFileByIndex(i, modelMode='norm')
            sacsLFile = modelFile + 'sacFile'
            sacsL, sacNamesL, srcSac = self.getSacFile(sacsLFile, randDrop=randDrop, para=para)
            if self.isFromO:
                for sacs in sacsL:
                    sacs[0] = seism.sacFromO(sacs[0])
            if not isinstance(noises,type(None)):
                noises(sacsL,channelL=[0])
            corrL += corrSacsL(disp,sacsL,sacNamesL, modelFile=modelFile, srcSac=srcSac,
              minSNR=minSNR,
              isCut=(self.isCut),
              minDist=(self.minDist),
              maxDist=(self.maxDist),
              minDDist=(self.minDDist),
              maxDDist=(self.maxDDist),
              isFromO=(self.isFromO),
              removeP=(self.removeP))
            print('###########', len(corrL))

        return corrL

    def getSacFile(self, sacFile, randDrop=0.3, para={}):
        sacsL = []
        sacNamesL = []
        srcSac = ''
        self.para0.update(para)
        print(self.para0)
        with open(sacFile) as (f):
            lines = f.readlines()
        srcData = ''
        if np.random.rand() < randDrop:
            duraCount = int(5 + 200 * np.random.rand() ** 2)
            srcData = np.zeros(duraCount)
            if np.random.rand() < randDrop:
                if np.random.rand() < randDrop:
                    randomSource(3, duraCount, srcData)
                else:
                    randomSource(2, duraCount, srcData)
            else:
                randomSource(4, duraCount, srcData)
            if self.convolveSrc:
                print('convolve data')
        for line in lines:
            if line[0] == '#':
                srcSac = line[1:]
            else:
                if np.random.rand() < randDrop:
                    pass
                else:
                    sacNames = line.split()
                    sacNamesL.append(sacNames)
                    sacsL.append([obspy.read(sacName)[0] for sacName in sacNames])
                    if self.para0['freq'][0] > 0:
                        for sac in sacsL[(-1)]:
                            if len(srcData) > 0:
                                if self.convolveSrc:
                                    sac.data = np.convolve(sac.data, srcData, 'same')
                            sac.filter((self.para0['filterName']), freqmin=(self.para0['freq'][0]),
                              freqmax=(self.para0['freq'][1]),
                              corners=(self.para0['corners']),
                              zerophase=(self.para0['zerophase']))

        return sacsL, sacNamesL, srcSac

    def getNoise(self, quakes, stations, mul=0.2, byRecord=False, remove_resp=False, para={}):
        self.para0.update(para)
        sacsL = quakes.getSacFiles(stations, isRead=True, strL='ZNE', byRecord=byRecord,
          minDist=(self.minDist),
          maxDist=(self.maxDist),
          remove_resp=remove_resp,
          para=(self.para0))
        return seism.Noises(sacsL, mul=mul)

    def loadNEFV(self, stations, fvDir='models/NEFV', mode='NEFile'):
        fvD = {}
        fvFileD = {}
        fvFileL = glob('%s/*avgpvt' % fvDir)
        if len(fvFileL) == 0:
            fvFileL = glob('%s/*.dat' % fvDir)
        for fvFile in fvFileL:
            key = os.path.basename(fvFile).split('-')[0]
            fvFileD[key] = fvFile

        for i in range(len(stations)):
            for j in range(len(stations)):
                pairKey = '%s.%s_%s.%s' % (stations[i]['net'], stations[i]['sta'],
                 stations[j]['net'], stations[j]['sta'])
                if pairKey in fvFileD:
                    fvD[keyConvert(pairKey)] = fv((fvFileD[pairKey]), mode=mode)

        return fvD

    def loadQuakeNEFV(self, stations, quakeFvDir='models/QuakeNEFV', quakeD=seism.QuakeL(), isGetQuake=True,**kwags):
        fvD = {}
        for i in range(len(stations)):
            print('do for %d in %d' % (i + 1, len(stations)))
            for j in range(len(stations)):
                file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt_sel.dat' % (quakeFvDir,
                 stations[i]['net'], stations[i]['sta'], stations[j]['net'], stations[j]['sta']))
                if len(file) > 0:
                    for f in file:
                        getFVFromPairFile(f, fvD, quakeD, isGetQuake=isGetQuake,stations=stations,**kwags)

                else:
                    file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt.dat' % (quakeFvDir,
                     stations[i]['net'], stations[i]['sta'], stations[j]['net'], stations[j]['sta']))
                    if len(file) > 0:
                        for f in file:
                            getFVFromPairFile(f, fvD, quakeD, isGetQuake=isGetQuake,stations=stations,**kwags)

                    else:
                        file = glob('%s/%s.%s_%s.%s-pvt*' % (quakeFvDir,
                         stations[i]['net'], stations[i]['sta'], stations[j]['net'], stations[j]['sta']))
                        if len(file) > 0:
                            for f in file:
                                getFVFromPairFile(f, fvD, quakeD, isGetQuake=isGetQuake,stations=stations,**kwags)

        return (
         fvD, quakeD)
    def loadQuakeNEFVAv(self, stations, quakeFvDir='models/QuakeNEFV', threshold=2, minP=0.5, minThreshold=0.02, minSta=5):
        with multiprocessing.Manager() as (m):
            fvD = m.dict()
            quakeD = m.dict()
            arg = []
            for i in range(len(stations)):
                for j in range(i):
                    sta0 = stations[i]
                    sta1 = stations[j]
                    dist = stations[i].dist(stations[j])
                    if dist > 1800:
                        pass
                    else:
                        arg.append([sta0['net'], sta1['net'], sta0['sta'], sta1['sta'], fvD, quakeD, quakeFvDir, threshold, minP, minThreshold, minSta])
            with Pool(30) as (p):
                p.map(loadOne, arg)
                qcFvD(fvD)
            return (
             {key:fvD[key] for key in fvD}, seism.QuakeL([quakeD[key] for key in quakeD]))
def loadOne(l):
    net0, net1, sta0, sta1, fvD, quakeD, quakeFvDir, threshold, minP, minThreshold, minSta = l
    fvDPair = {}
    file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt_sel.dat' % (quakeFvDir,
     net0, sta0, net1, sta1))
    if len(file) > 0:
        for f in file:
            getFVFromPairFile(f, fvDPair, quakeD, isPrint=False)

    else:
        file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt.dat' % (quakeFvDir,
         net0, sta0, net1, sta1))
        if len(file) > 0:
            for f in file:
                getFVFromPairFile(f, fvDPair, quakeD, isPrint=False)

        else:
            file = glob('%s/%s.%s_%s.%s-pvt*' % (quakeFvDir,
             net0, sta0, net1, sta1))
    if len(file) > 0:
        for f in file:
            getFVFromPairFile(f, fvDPair, quakeD, isPrint=False)

    sta0, sta1 = sta1, sta0
    net0, net1 = net1, net0
    file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt_sel.dat' % (quakeFvDir,
     net0, sta0, net1, sta1))
    if len(file) > 0:
        for f in file:
            getFVFromPairFile(f, fvDPair, quakeD, isPrint=False)

    else:
        file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt.dat' % (quakeFvDir,
         net0, sta0, net1, sta1))
        if len(file) > 0:
            for f in file:
                getFVFromPairFile(f, fvDPair, quakeD, isPrint=False)

        else:
            file = glob('%s/%s.%s_%s.%s-pvt*' % (quakeFvDir,
             net0, sta0, net1, sta1))
    if len(file) > 0:
        for f in file:
            getFVFromPairFile(f, fvDPair, quakeD, isPrint=False)

    sta0, sta1 = sta1, sta0
    net0, net1 = net1, net0
    if len(fvDPair) < 5:
        return
    for key in fvDPair:
        fvDPair[key].qc(threshold=(-minP))

    qcFvD(fvDPair)
    fvAv = averageFVL([fvDPair[key] for key in fvDPair], minThreshold=minThreshold, minSta=minSta)
    fvAv.qc(threshold=threshold)
    if len(fvAv.f) > 2:
        fvD[net0 + '.' + sta0 + '_' + net1 + '.' + sta1] = fvAv
        if np.random.rand() < 0.01:
            print('%s std: %6.5f minT: %5.1f %4.2f maxT:  %5.1f %4.2f quake: %5d num: %4d' % (
             net0 + '.' + sta0 + '_' + net1 + '.' + sta1, fvAv.std.mean(),
             1 / fvAv.f[(-1)], fvAv.v[(-1)], 1 / fvAv.f[0], fvAv.v[0], len(quakeD), len(fvDPair)))


class layer:
    __doc__ = "\n    class for layered media;\n    the p velocity(vp), s velocity(vs), density(rho), [top depth, bottom depth](z) is needed; \n    p and s 's Q(Qp, Qs) is optional(default is 1200,600)\n    After specifying the above parameters, the lame parameter(lambda, miu), zeta and xi would\n     be calculate as class's attributes. \n    "
    def __init__(self, vp, vs, rho, z=[0, 0], Qp=1200, Qs=600):
        self.z = np.array(z)
        self.vp = np.array(vp)
        self.vs = np.array(vs)
        self.rho = np.array(rho)
        self.Qp = Qp
        self.Qs = Qs
        self.lamb, self.miu = self.getLame()
        self.zeta = self.getZeta()
        self.xi = self.getXi()

    @jit
    def getLame(self):
        miu = self.vs ** 2 * self.rho
        lamb = self.vp ** 2 * self.rho - 2 * miu
        return (lamb, miu)

    @jit
    def getZeta(self):
        return 1 / (self.lamb + 2 * self.miu)

    @jit
    def getXi(self):
        zeta = self.getZeta()
        return 4 * self.miu * (self.lamb + self.miu) * zeta

    @jit
    def getNu(self, k, omega):
        return (k ** 2 - (omega / self.vs.astype(np.complex)) ** 2) ** 0.5

    @jit
    def getGamma(self, k, omega):
        return (k ** 2 - (omega / self.vp.astype(np.complex)) ** 2) ** 0.5

    @jit
    def getChi(self, k, omega):
        nu = self.getNu(k, omega)
        return k ** 2 + nu ** 2

    @jit
    def getEA(self, k, omega, z, mode='PSV'):
        nu = self.getNu(k, omega)
        gamma = self.getGamma(k, omega)
        chi = self.getChi(k, omega)
        alpha = self.vp
        beta = self.vs
        miu = self.miu
        if mode == 'PSV':
            E = 1 / omega * np.array([
             [
              alpha * k, beta * nu, alpha * k, beta * nu],
             [
              alpha * gamma, beta * k, -alpha * gamma, -beta * k],
             [
              -2 * alpha * miu * k * gamma, -beta * miu * chi, 2 * alpha * miu * k * gamma, beta * miu * chi],
             [
              -alpha * miu * chi, -2 * beta * miu * k * nu, -alpha * miu * chi, -2 * beta * miu * k * nu]])
            A = np.array([
             [
              np.exp(-gamma * (z - self.z[0])), 0, 0, 0],
             [
              0, np.exp(-nu * (z - self.z[0])), 0, 0],
             [
              0, 0, np.exp(gamma * (z - self.z[1])), 0],
             [
              0, 0, 0, np.exp(nu * (z - self.z[1]))]])
        else:
            if mode == 'SH':
                E = np.array([
                 [
                  1, 1],
                 [
                  -miu * nu, miu * nu]])
                A = np.array([
                 [
                  np.exp(-nu * (z - self.z[0])), 0],
                 [
                  0, np.exp(nu * (z - self.z[1]))]])
        return (
         E, A)


class surface:
    __doc__ = "\n    class for surface of layer\n    the layers above and beneath(layer0, layer1) is needed\n    Specify the bool parameters isTop and isBottom, if the surface is the first or last one； the default is false\n    the default waveform mode(mode) is 'PSV', you can set it to 'SH'\n    "

    def __init__(self, layer0, layer1, mode='PSV', isTop=False, isBottom=False):
        self.layer0 = layer0
        self.layer1 = layer1
        self.z = layer0.z[(-1)]
        self.Td = 0
        self.Tu = 0
        self.Rud = 0
        self.Rdu = 0
        self.TTd = 0
        self.TTu = 0
        self.RRud = 0
        self.RRdu = 0
        self.mode = mode
        self.isTop = isTop
        self.isBottom = isBottom
        self.E = [None, None]
        self.A = [None, None]

    @jit
    def submat(self, M):
        shape = M.shape
        lenth = int(shape[0] / 2)
        newM = M.reshape([2, lenth, 2, lenth])
        newM = newM.transpose([0, 2, 1, 3])
        return newM

    @jit
    def setTR(self, k, omega):
        E0, A0 = self.layer0.getEA(k, omega, self.z, self.mode)
        E1, A1 = self.layer1.getEA(k, omega, self.z, self.mode)
        E0 = self.submat(E0)
        E1 = self.submat(E1)
        A0 = self.submat(A0)
        A1 = self.submat(A1)
        self.E = [E0, E1]
        self.A = [A0, A1]
        EE0 = self.toMat([[E1[0][0], -E0[0][1]],
         [
          E1[1][0], -E0[1][1]]])
        EE1 = self.toMat([[E0[0][0], -E1[0][1]],
         [
          E0[1][0], -E1[1][1]]])
        AA = self.toMat([[A0[0][0], A0[0][0] * 0],
         [
          A1[0][0] * 0, A1[1][1]]])
        TR = EE0 ** (-1) * EE1 * AA
        TR = self.submat(np.array(TR))
        self.Td = TR[0][0]
        self.Rdu = TR[1][0]
        self.Rud = TR[0][1]
        self.Tu = TR[1][1]
        if self.isTop:
            self.Rud = -E1[1][0] ** (-1) * E1[1][1] * A1[1][1]
            self.Td = self.Rud * 0
            self.Tu = self.Rud * 0

    @jit
    def toMat(self, l):
        shape0 = len(l)
        shape1 = len(l[0])
        shape = np.zeros(2).astype(np.int64)
        shape[0] = l[0][0].shape[0]
        shape[1] = l[0][0].shape[1]
        SHAPE = shape + 0
        SHAPE[0] *= shape0
        SHAPE[1] *= shape1
        M = np.zeros(SHAPE, np.complex)
        for i in range(shape0):
            for j in range(shape1):
                i0 = i * shape[0]
                i1 = (i + 1) * shape[0]
                j0 = j * shape[1]
                j1 = (j + 1) * shape[1]
                M[i0:i1, j0:j1] = l[i][j]

        return np.mat(M)

    @jit
    def setTTRRD(self, surface1=0):
        if self.isBottom:
            RRdu1 = np.mat(self.Rdu * 0)
        else:
            RRdu1 = surface1.RRdu
        self.TTd = (np.mat(np.eye(self.Rud.shape[0])) - np.mat(self.Rud) * np.mat(RRdu1)) ** (-1) * np.mat(self.Td)
        self.RRdu = np.mat(self.Rdu) + np.mat(self.Tu) * np.mat(RRdu1) * self.TTd

    @jit
    def setTTRRU(self, surface0=0):
        if self.isTop:
            self.RRud = self.Rud
            return 0
        self.TTu = (np.mat(np.eye(self.Rud.shape[0])) - np.mat(self.Rdu) * np.mat(surface0.RRud)) ** (-1) * np.mat(self.Tu)
        self.RRud = np.mat(self.Rud) + np.mat(self.Td) * np.mat(surface0.RRud) * self.TTu


class model:
    __doc__ = "\n    class for layered media model\n    modeFile is the media parameter model File, there are tow mods\n    if layerMode == 'norm':\n       '0    18  2.80  6.0 3.5'\n       layer's top depth, layer's bottom depth, density, p velocity, svelocity\n    if layerMode =='prem':\n        '0.00       5.800     3.350       2.800    1400.0     600.0'\n        depth,  p velocity, s velocity, density,    Qp,        Qs\n    mode is for PSV and SH\n    getMode is the way to get phase velocity:\n        norm is enough to get phase velocity\n        new is to get fundamental phase velocity for PSV\n    "

    def __init__(self, modelFile, mode='PSV', getMode='norm', layerMode='prem', layerN=10000, isFlat=False, R=6371, flatM=-2, pog='p', gpdcExe=gpdcExe, doFlat=True, QMul=1):
        self.modelFile = modelFile
        self.getMode = getMode
        self.isFlat = isFlat
        self.gpdcExe = gpdcExe
        self.mode = mode
        data = np.loadtxt(modelFile)
        layerN = min(data.shape[0] + 1, layerN + 1)
        layerL = [None for i in range(layerN)]
        if layerMode == 'old':
            layerL[0] = layer(1.7, 1, 0.0001, [-100, 0])
            for i in range(1, layerN):
                layerL[i] = layer(data[(i - 1, 3)], data[(i - 1, 4)], data[(i - 1, 2)], data[i - 1, :2])

        else:
            if layerMode == 'prem' or layerMode == 'norm':
                layerL[0] = layer(1.7, 1, 0.0001, [-100, 0])
                zlast = 0
                for i in range(1, layerN):
                    vp = data[(i - 1, 1)]
                    vs = data[(i - 1, 2)]
                    rho = data[(i - 1, 3)]
                    if data.shape[1] == 6:
                        Qp = data[(i - 1, 4)]
                        Qs = data[(i - 1, 5)]
                    else:
                        Qp = 1200
                        Qs = 600
                    Qp *= QMul
                    Qs *= QMul
                    z = np.array([data[(i - 1, 0)], data[(min(i + 1 - 1, layerN - 2), 0)]])
                    if isFlat:
                        if doFlat:
                            z, vp, vs, rho = flat(z, vp, vs, rho, m=flatM, R=R)
                    layerL[i] = layer(vp, vs, rho, z, Qp, Qs)

            else:
                if layerMode in '123456':
                    index = int(layerMode)
                    layerL[0] = layer(0.85, 0.5, 0.0001, [-100, 0])
                    for i in range(1, layerN):
                        vs = data[(i - 1, index)]
                        vp = vs * 1.73
                        rho = vs / 3
                        Qp = 1200
                        Qs = 600
                        Qp *= QMul
                        Qs *= QMul
                        z = np.array([data[(i - 1, 0)], data[(min(i + 1 - 1, layerN - 2), 0)]])
                        if isFlat:
                            if doFlat:
                                z, vp, vs, rho = flat(z, vp, vs, rho, m=flatM, R=R)
                                print(vp, vs)
                        layerL[i] = layer(vp, vs, rho, z, Qp, Qs)

        surfaceL = [None for i in range(layerN - 1)]
        for i in range(layerN - 1):
            isTop = False
            isBottom = False
            if i == 0:
                isTop = True
            if i == layerN - 2:
                isBottom = True
            surfaceL[i] = surface(layerL[i], layerL[(i + 1)], mode, isTop, isBottom)

        self.layerL = layerL
        self.surfaceL = surfaceL
        self.layerN = layerN

    @jit
    def set(self, k, omega):
        for s in self.surfaceL:
            s.setTR(k, omega)

        for i in range(self.layerN - 1 - 1, -1, -1):
            s = self.surfaceL[i]
            if i == self.layerN - 1 - 1:
                s.setTTRRD(self.surfaceL[0])
            else:
                s.setTTRRD(self.surfaceL[(i + 1)])

        for i in range(self.layerN - 1):
            s = self.surfaceL[i]
            if i == 0:
                s.setTTRRU(self.surfaceL[0])
            else:
                s.setTTRRU(self.surfaceL[(i - 1)])

    @jit
    def get(self, k, omega):
        self.set(k, omega)
        RRud0 = self.surfaceL[0].RRud
        RRdu1 = self.surfaceL[1].RRdu
        if self.getMode == 'norm':
            M = np.mat(np.eye(RRud0.shape[0])) - RRud0 * RRdu1
        else:
            if self.getMode == 'new':
                M = np.mat(self.surfaceL[0].E[1][1][0]) + np.mat(self.surfaceL[0].E[1][1][1]) * np.mat(self.surfaceL[0].A[1][1][1]) * RRdu1
                MA = np.array(M)
                MA /= np.abs(MA).std()
                return np.linalg.det(np.mat(MA))
        return np.linalg.det(M)

    @jit
    def plot(self, omega, dv=0.01):
        v, k, det = self.calList(omega, dv)
        plt.plot(v, np.real(det), '-k')
        plt.plot(v, np.imag(det), '-.k')
        plt.plot(v, np.abs(det), 'r')
        plt.show()

    @jit
    def calList(self, omega, dv=0.01):
        vs0 = self.layerL[1].vs
        vp0 = self.layerL[1].vp
        v = np.arange(vs0 - 0.499, vs0 + 5, dv)
        k = omega / v
        det = k.astype(np.complex) * 0
        for i in range(k.shape[0]):
            det[i] = self.get(k[i], omega)

        return (
         v, k, det)

    @jit
    def __call__(self, omega, calMode='fast'):
        return self.calV(omega, order=0, dv=0.002, DV=0.008, calMode=calMode, threshold=0.1)

    def calV(self, omega, order=0, dv=0.001, DV=0.008, calMode='norm', threshold=0.05, vStart=-1):
        if calMode == 'norm':
            v, k, det = self.calList(omega, dv)
            iL, detL = getDetec((-np.abs(det)), minValue=(-0.1), minDelta=(int(DV / dv)))
            i0 = iL[order]
            v0 = v[i0]
            det0 = -detL[0]
        else:
            if calMode == 'fast':
                v0, det0 = self.calVFast(omega, order=order, dv=dv, DV=DV, threshold=threshold, vStart=vStart)
        return (
         v0, det0)

    @jit
    def calVFast(self, omega, order=0, dv=0.01, DV=0.008, threshold=0.05, vStart=-1):
        if self.getMode == 'new':
            v = 2.7
        else:
            v = self.layerL[1].vs + 1e-08
        if vStart > 0:
            v = vStart - 0.02
            dv = 0.0005
        v0 = v
        det0 = 1000000000.0
        for i in range(100000):
            v1 = i * dv + v
            if np.abs(v1 - self.layerL[1].vs) < 0.005:
                continue
            det1 = np.abs(self.get(omega / v1, omega))
            if det1 < threshold and det1 < det0:
                    v0 = v1
                    det0 = det1
            if det0 < threshold and det1 > det0:
                    print(v0, 2 * np.pi / omega, det0)
                    return (v0, det0)
        return (2, 0.1)

    def calByGpdc(self, order=0, pog='p', T=np.arange(1, 100, 5).astype(np.float)):
        pogStr = pog
        if pog == 'p':
            pog = ''
        else:
            pog = '-group'
        modelInPut = self.modelFile + '_gpdc' + '_' + pogStr
        resFile = self.modelFile + '_gpdc_tmp' + '_' + pogStr
        with open(modelInPut, 'w') as (f):
            count = 0
            for layer in self.layerL[1:]:
                if layer.z[1] - layer.z[0] < 0.1:
                    pass
                else:
                    count += 1

            f.write('%d' % count)
            for layer in self.layerL[1:]:
                if layer.z[1] - layer.z[0] < 0.1:
                    pass
                else:
                    f.write('\n')
                    f.write('%f %f %f %f' % ((layer.z[1] - layer.z[0]) * 1000.0,
                     layer.vp * 1000.0, layer.vs * 1000.0, layer.rho * 1000.0))

        if self.mode == 'PSV':
            cmdRL = ' -R %d ' % (order + 1)
        else:
            cmdRL = ' -R 0 -L %d ' % (order + 1)
        cmd = '%s  %s %s  %s -min %f -max %f > %s' % (
         self.gpdcExe, modelInPut, cmdRL, pog, 1 / T.max(), 1 / T.min(), resFile)
        os.system(cmd)
        data = np.loadtxt(resFile)
        print(data)
        return (data[:, 0], 0.001 / data[:, -1])

    @jit
    def calDispersion(self, order=0, calMode='norm', threshold=0.1, T=np.arange(1, 100, 5).astype(np.float), pog='p'):
        if calMode == 'gpdc':
            return self.calByGpdc(order, pog, T)
        else:
            f = 1 / T
            omega = 2 * np.pi * f
            v = omega * 0
            v00 = 3
            for i in range(omega.size):
                if pog == 'p':
                    V = np.abs(self.calV((omega[i]), order=order, calMode=calMode, threshold=threshold, vStart=(v00 - 0.2)))[0]
                    v00 = V
                    v[i] = np.abs(self.calV((omega[i]), order=order, calMode=calMode, threshold=threshold, vStart=v00))[0]
                else:
                    if pog == 'g':
                        omega0 = omega[i] * 0.98
                        omega1 = omega[i] * 1.02
                        V = np.abs(self.calV(omega1, order=order, calMode=calMode, threshold=threshold, vStart=(v00 - 0.2)))[0]
                        v00 = V
                        v0 = np.abs(self.calV(omega0, order=order, calMode=calMode, threshold=threshold, vStart=v00))[0]
                        v1 = np.abs(self.calV(omega1, order=order, calMode=calMode, threshold=threshold, vStart=v00))[0]
                        dOmega = omega1 - omega0
                        dK = omega1 / v1 - omega0 / v0
                        v[i] = dOmega / dK

            return (
             f, v)

    def test(self):
        self.plot(2 * np.pi)

    def testDispersion(self):
        f, v = self.calDispersion()
        plt.plot(f, v)
        plt.show()

    def compare(self, dv=0.01):
        self.getMode = 'norm'
        v, k, det = self.calList(6.28, dv)
        plt.plot(v, np.abs(det) / np.abs(det).max(), 'k')
        self.getMode = 'new'
        v, k, det = self.calList(6.28, dv)
        plt.plot(v, np.abs(det) / np.abs(det).max(), 'r')
        plt.show()

    def covert2Fk(self, fkMode=0):
        if fkMode == 0:
            filename = self.modelFile + 'fk0'
        else:
            filename = self.modelFile + 'fk1'
        if self.isFlat:
            filename += '_flat'
        with open(filename, 'w+') as (f):
            for i in range(1, self.layerN):
                layer = self.layerL[i]
                thickness = layer.z[1] - layer.z[0]
                vp = layer.vp.copy()
                vs = layer.vs.copy()
                rho = layer.rho
                if fkMode == 0:
                    vp /= vs
                print('%.2f %.2f %.2f %.2f %d %d' % (thickness, vs, vp, rho, layer.Qp, layer.Qs))
                f.write('%.2f %.2f %.2f %.2f %d %d' % (thickness, vs, vp, rho, layer.Qp, layer.Qs))
                if i != self.layerN - 1:
                    f.write('\n')
    def outputZV(self):
        layerN = len(self.layerL)
        z = np.zeros(layerN * 2 - 2)
        vp = np.zeros(layerN * 2 - 2)
        vs = np.zeros(layerN * 2 - 2)
        for i in range(1, layerN):
            iNew = i - 1
            z[iNew * 2:iNew * 2 + 2] = self.layerL[i].z
            vp[iNew * 2:iNew * 2 + 2] = np.array([self.layerL[i].vp, self.layerL[i].vp])
            vs[iNew * 2:iNew * 2 + 2] = np.array([self.layerL[i].vs, self.layerL[i].vs])

        return (
         z, vp, vs)

    def outputZVRho(self):
        layerN = len(self.layerL)
        z = np.zeros(layerN * 2 - 2)
        vp = np.zeros(layerN * 2 - 2)
        vs = np.zeros(layerN * 2 - 2)
        rho = np.zeros(layerN * 2 - 2)
        for i in range(1, layerN):
            iNew = i - 1
            z[iNew * 2:iNew * 2 + 2] = self.layerL[i].z
            vp[iNew * 2:iNew * 2 + 2] = np.array([self.layerL[i].vp, self.layerL[i].vp])
            vs[iNew * 2:iNew * 2 + 2] = np.array([self.layerL[i].vs, self.layerL[i].vs])
            rho[iNew * 2:iNew * 2 + 2] = np.array([self.layerL[i].rho, self.layerL[i].rho])

        return (
         z, vp, vs, rho)

    def __call__(self, z):
        z0, vp0, vs0, rho0 = self.outputZVRho()
        interpP = interpolate.interp1d(z0, vp0, kind='linear', bounds_error=False,
          fill_value=(vp0[(-1)]))
        interpS = interpolate.interp1d(z0, vs0, kind='linear', bounds_error=False,
          fill_value=(vs0[(-1)]))
        interpRho = interpolate.interp1d(z0, rho0, kind='linear', bounds_error=False,
          fill_value=(rho0[(-1)]))
        return (interpP(z), interpS(z), interpRho(z))

    def to2D(self, x, z):
        nz = z.size
        nx = x.size
        zM = np.zeros([nz, nx]) + z.reshape([-1, 1])
        return self(zM)


class disp:
    __doc__ = '\n    traditional method to calculate the dispersion curve\n    then should add some sac to handle time difference\n    '

    def __init__(self, nperseg=300, noverlap=298, fs=1, halfDt=150, xcorrFunc=xcorrComplex):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.fs = fs
        self.halfDt = halfDt
        self.halfN = np.int(halfDt * self.fs)
        self.xcorrFunc = xcorrFunc

    @jit
    def cut(self, data):
        maxI = np.argmax(data)
        i0 = max(maxI - self.halfN, 0)
        i1 = min(maxI + self.halfN, data.shape[0])
        return (
         data[i0:i1], i0, i1)

    @jit
    def xcorr(self, data0, data1,isCut=True,fromI=0):
        if isCut:
            data1, i0, i1 = self.cut(data1)
        i0 = 0
        i1 = 0
        xx = self.xcorrFunc(data0, data1,fromI=fromI)
        return (xx, i0, i1)

    @jit
    def stft(self, data):
        F, t, zxx = scipy.signal.stft((np.real(data)), fs=(self.fs), nperseg=(self.nperseg), noverlap=(self.noverlap))
        F, t, zxxj = scipy.signal.stft((np.imag(data)), fs=(self.fs), nperseg=(self.nperseg), noverlap=(self.noverlap))
        zxx = zxx + zxxj * complex(0.0, 1.0)
        zxx /= np.abs(zxx).max(axis=1, keepdims=True)
        return (F, t, zxx)

    def show(self, F, t, zxx, data, timeL, isShow=True):
        plt.subplot(2, 1, 1)
        plt.pcolor(t, F, np.abs(zxx))
        plt.subplot(2, 1, 2)
        plt.plot(timeL, data)
        if isShow:
            plt.show()

    def sacXcorr(self, sac0, sac1, isCut=False, maxCount=-1,fromT=0, **kwags):
        fs = sac0.stats['sampling_rate']
        self.fs = fs
        self.halfN = np.int(self.halfDt * self.fs)
        data0 = sac0.data
        time0 = sac0.stats.starttime.timestamp
        dis0 = sac0.stats['sac']['dist']
        data1 = sac1.data
        time1 = sac1.stats.starttime.timestamp
        dis1 = sac1.stats['sac']['dist']
        fromI = int(fromT/sac0.stats['sac']['delta'])
        #if fromI>0:
        #    data0=data0[fromI:]
        #if fromI<0:
        #    data1=data1[-fromI:]
        xx, i0, i1 = self.xcorr(data0, data1, isCut,fromI=fromI)
        time1New = time1 + i0 / fs
        dTime = time0 - time1New
        timeL = np.arange(xx.size) / fs + dTime+fromT
        dDis = dis0 - dis1
        return corr(np.real(xx), timeL, dDis, fs, maxCount=maxCount, **kwags)

    def test(self, data0, data1, isCut=True):
        xx = self.xcorr(data0, data1, isCut=isCut)
        F, t, zxx = self.stft(xx)
        self.show(F, t, zxx, xx)

    def testSac(self, sac0, sac1, isCut=True, fTheor=[], vTheor=[]):
        xx, timeL, dDis, fs = self.sacXcorr(sac0, sac1, isCut=True).output()
        F, t, zxx = self.stft(xx)
        print(t)
        t = t + timeL[0] + 0 * self.nperseg / self.fs
        self.show(F, t, zxx, xx, timeL, isShow=False)
        if len(fTheor) > 0:
            timeTheorL = dDis / vTheor
            plt.subplot(2, 1, 1)
            plt.plot(timeTheorL, fTheor)
        return (
         xx, zxx, F, t)



class fv:
    __doc__ = '\n    class for dispersion result\n    it have two attributes f and v, each element in v accosiate with an \n     element in v \n     用于插值的本身就必须是好的数\n    '

    def __init__(self, input, mode='num', threshold=0.06):
        self.mode = mode
        if mode == 'num':
            self.f = input[0]
            self.v = input[1]
            self.std = self.f * 0 + 99
            if len(input) > 2:
                self.std = input[2]
        if mode == 'dis':
            self.f = input[0]
            self.v = input[1]
            self.std = self.f * 0 + 99
            if len(input) > 2:
                self.std = input[2]
        if mode == 'file':
            fvM = np.loadtxt(input)
            self.f = fvM[:, 0]
            self.v = fvM[:, 1]
            self.std = self.f * 0 + 99
        if mode == 'NEFile':
            T = []
            v = []
            std = []
            with open(input) as (f):
                lines = f.readlines()
            for line in lines[3:]:
                tmp = line.split()
                T.append(float(tmp[0]))
                v.append(float(tmp[1]))
                std.append(float(tmp[2]))

            f = 1 / np.array(T)
            v = np.array(v)
            std = np.array(std)
            f = f[(std <= threshold)]
            v = v[(std <= threshold)]
            if len(f) <= 1:
                f = np.array([-1, 0])
                v = np.array([-1, 0])
                std = np.array([-1, 0])
            self.f = f
            self.v = v
            self.std = std[(std <= threshold)]
        if mode == 'NEFileNew':
            T = []
            v = []
            std = []
            with open(input) as (f):
                lines = f.readlines()
            for line in lines:
                tmp = line.split()
                T.append(float(tmp[0]))
                v.append(float(tmp[1]))
                std.append(float(tmp[2]))

            f = 1 / np.array(T)
            v = np.array(v)
            std = np.array(std)
            f = f[(std <= threshold)]
            v = v[(std <= threshold)]
            if len(f) <= 1:
                f = np.array([-1, 0])
                v = np.array([-1, 0])
            self.f = f
            self.v = v
            self.std = std[(std <= threshold)]
        if mode == 'fileP':
            print(input + '_/*')
            fileL = glob(input + '_/*')
            distL = []
            vL = []
            fL = []
            for file in fileL:
                distL.append(float(file.split('/')[(-1)]))
                fvM = np.loadtxt(file)
                fL.append(fvM[:, 0])
                vL.append(fvM[:, 1:2])

            self.f = fL[0]
            self.dist = np.array(distL)
            self.v = np.concatenate(vL, axis=1)
            iL = self.dist.argsort()
            self.dist = self.dist[iL]
            self.v = self.v[:, iL]
        if len(self.f) < 2:
            self.f = np.array([1, 2])
            self.v = np.array([1e-13, 1e-13])
            self.std = np.array([99, 99])
        self.f = self.f[(self.v > 2)]
        self.std = self.std[(self.v > 2)]
        self.v = self.v[(self.v > 2)]
        if len(self.f) < 2:
            self.f = np.array([1, 2])
            self.v = np.array([1e-13, 1e-13])
            self.std = np.array([99, 99])
        self.interp = self.genInterp()

    def genInterp(self):
        if self.mode == 'fileP':
            return interpolate.interp2d((self.dist), (self.f), (self.v), kind='linear', bounds_error=False,
              fill_value='extrapolate')
        else:
            if self.mode == 'dis':
                pass
            else:
                if len(self.std) > 0:
                    self.STD = interpolate.interp1d((self.f), (self.std), kind='linear', fill_value='extrapolate',
                      bounds_error=False)
            return interpolate.interp1d((self.f), (self.v), kind='linear', bounds_error=False,
              fill_value='extrapolate')

    def __call__(self, f, dist0=0, dist1=0, threshold=0.08, N=1000, randA=0.1):
        shape0 = f.shape
        f = f.reshape([-1])
        if self.mode == 'fileP':
            dDist = (dist1 - dist0) / 10000
            distL = np.arange(dist0, dist1 + 0.0001, dDist)
            vL = self.interp(distL, f)
            iL = f.argsort()
            vL = 1 / (1 / vL).mean(axis=1)
            vLNew = vL.copy()
            for i in range(iL.size):
                vLNew[i] = vL[iL[i]]

            vL = vLNew
        else:
            if len(self.f) > 2:
                vL = self.interp(f)
            else:
                return f.reshape(shape0) * 0 + 1e-08
        df0 = f.reshape([-1, 1]) - self.f[:-1].reshape([1, -1])
        df1 = f.reshape([-1, 1]) - self.f[1:].reshape([1, -1])
        dfR = (np.abs(df0) / 2 + np.abs(df1) / 2).min(axis=1) / f
        df01R = (df0 * df1).min(axis=1) / (f * f)
        vL[dfR > threshold] = 1e-08
        vL[df01R > 0.0001] = 1e-08
        return vL.reshape(shape0)

    def save(self, filename, mode='num',quake='',station0='',station1='',writeType='a'):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        if mode == 'num':
            np.savetxt(filename, np.concatenate([self.f.reshape([-1, 1]),
                self.v.reshape([-1, 1])],
                axis=1))
            return
        if mode == 'NEFile':
            with open(filename, 'w+') as (f):
                f.write('remain\nremain\nremain\n')
                for i in range(len(self.f)):
                    T = 1 / self.f[i]
                    v = self.v[i]
                    std = -1
                    if len(self.std) > 0:
                        std = self.std[i]
                    f.write('%f %f %f\n' % (T, v, std))
        if mode=='pair':
            with open(filename, writeType) as (f):
                time = quake['time']
                la = quake['la']
                lo = quake['lo']
                ml = quake['ml']
                dep = quake['dep']
                dist0 = station0.distaz(quake)
                dist1 = station1.distaz(quake)
                f.write('%s %s\n' % (station1['sta'], station0['sta']))
                f.write('%s 5\n' % station1['comp'][(-1)])
                f.write('%s %.5f\n' % (obspy.UTCDateTime(time).strftime('%Y %m %d %H %M'), time % 60))
                f.write('%f %f\n' % (station1['la'], station1['lo']))
                f.write('%.5f %.5f %.5f %.5f 0\n' % (la, lo,dep,ml))
                f.write('%f %f 0 0 \n' % (dist0.getDelta(), dist0.getAz()))
                f.write('%s 5\n' % station0['comp'][(-1)])
                f.write('%s %.5f\n' % (obspy.UTCDateTime(time).strftime('%Y %m %d %H %M'), time % 60))
                f.write('%f %f\n' % (station0['la'], station0['lo']))
                f.write('%.5f %.5f %.5f %.5f 0\n' % (la, lo,dep,ml))
                f.write('%f %f 0 0 \n' % (dist1.getDelta(), dist1.getAz()))
                f.write('2 %d\n' % len(self.v))
                for F in self.f:
                    f.write('%f\n' % F)
                for i in range(len(self.v)):
                    f.write('%f %f\n' % (self.v[i],self.std[i]))

    def update(self, self1, isReplace=True, threshold=0.1,isControl=True):
        v = self1(self.f).reshape([-1])
        dv = np.abs(v - self.v)
        if (dv <= threshold * v).sum() > 1:
            if not isControl:
                dv[v<1]=-1
            self.f = self.f[(dv <= threshold * v)]
            self.v = self.v[(dv < threshold * v)]
            vNew   = v[(dv < threshold * v)]
            if isReplace:
                self.v[vNew>1] = vNew[vNew>1]
            self.std = self.std[(dv <= threshold * v)]
        else:
            self.v[:] = 1e-09
        if len(self.f) > 2:
            self.interp = self.genInterp()

    def qc(self, threshold=0.08):
        v = self.v.copy()
        if threshold < 0:
            v = v * 0 + 1
        self.f = self.f[(self.std <= threshold * v)]
        self.v = self.v[(self.std <= threshold * v)]
        self.std = self.std[(self.std <= threshold * v)]
        if len(self.f)>2:
            self.interp = self.genInterp()
    def disQC_(self, fvRef, dis, randA=0.15, maxR=2):
        v = fvRef(self.f)
        T = 1 / self.f
        t = dis / v
        randT = t * randA
        maxT = t * maxR
        print(((T > randT) * (T < maxT)).sum(), len(t))
        self.f = self.f[((T > randT) * (T < maxT))]
        self.v = self.v[((T > randT) * (T < maxT))]
        self.std = self.std[((T > randT) * (T < maxT))]
        if len(self.f) > 2:
            self.interp = self.genInterp()
    def disQC(self, fvRef, dis, randA=0.15, maxR=2):
        #v = fvRef(self.f)
        T = 1 / self.f
        t = dis / self.v
        randT = t * randA
        maxT = t * maxR
        #print(((T > randT) * (T < maxT)).sum(), len(t))
        self.f = self.f[((T > randT) * (T < maxT))]
        self.v = self.v[((T > randT) * (T < maxT))]
        self.std = self.std[((T > randT) * (T < maxT))]
        if len(self.f) > 2:
            self.interp = self.genInterp()
    def replaceF(self, f):
        f = f.copy()
        v = self(f)
        std = self.STD(f)
        self.f = f[(v > 1)]
        self.v = v[(v > 1)]
        self.std = std[(v > 1)]
        if len(self.f) > 2:
            self.interp = self.genInterp()
    def coverQC(self, dis, minI, maxI, R=0.001):
        if len(self.f) <= 2:
            return
        minDis = minI(self.f) * (1 - R)
        maxDis = maxI(self.f) * (1 + R)
        self.f = self.f[((dis > minDis) * (dis < maxDis))]
        self.v = self.v[((dis > minDis) * (dis < maxDis))]
        self.std = self.std[((dis > minDis) * (dis < maxDis))]
        if len(self.f) > 2:
            self.interp = self.genInterp()
    def limit(self, self1, threshold=2):
        v = self(self1.f)
        std = self.STD(self1.f)
        vmin = v - threshold * std
        vmax = v + threshold * std
        valid = -(self1.v - vmin) * (self1.v - vmax)
        self1.f = self1.f[(valid > 0)]
        self1.v = self1.v[(valid > 0)]
        self1.std = self1.std[(valid > 0)]
        if len(self1.f) > 2:
            self1.genInterp()

    def compare(self, self1, fL=''):
        if len(fL) == 0:
            fL = self.fL
        v0 = self(fL)
        v1 = self1(fL)
        dv = v1 - v0
        dv[np.isnan(v1)] = -100
        return dv
    def copy(self):
        return fv([self.f,self.v,self.std],mode='num')
    def getDVK(self,f):
        v = self(f)
        N = len(v)
        DV = []
        K  = []
        for i in range(N):
            if i ==0:
                vL = [v[0],v[1],v[2]]
                fL = [f[0],f[1],f[2]]
                vRef = v[0]
                fRef = f[0]
            elif i == N-1:
                vL = [v[-1],v[-2],v[-3]]
                fL = [f[-1],f[-2],f[-3]]
                vRef = v[-1]
                fRef = f[-1]
            else:
                vL = [v[i-1],v[i],v[i+1]]
                fL = [f[i-1],f[i],f[i+1]]
                vRef = v[i]
                fRef = f[i]
            dv,k =self.calDVK(fL,vL,fRef,vRef)
            DV.append(dv)
            K.append(k)
        return DV,K
    def calDVK(self,f,v,fRef,vRef):
        v = np.array(v)
        f = np.array(f)
        if v.min()<1:
            return -999,-999
        v = v/vRef
        f = f/fRef
        dv = (v[-1]-v[0])/(f[-1]-f[0])
        dv0= (v[1]-v[0])/(f[1]-f[0])
        dv1=(v[-1]-v[1])/(f[-1]-f[1])
        ddv = (dv1-dv0)*2/(f[-1]-f[0])
        K = np.abs(ddv)/(1+dv**2)**(3/2)
        return dv,K
fvNone=fv(np.array([[0],[0]]))

def getDVK(fvD,f,fvD0={},isRight=True):
    DV=[]
    K=[]
    V=[]
    STD=[]
    for key in fvD:
        if isinstance(key,str):
            FV = fvD[key]
        else:
            FV = key
        if not isinstance(FV,fv):
            FV = FV.copy()
        dv,k=FV.getDVK(f)
        dv = np.array(dv)
        k  = np.array(k)
        v   = FV(f)
        std = -FV.STD(f)
        if key in fvD0:
            v0 = fvD0[key](f)
            if isRight:
                v[v0<1]=0
                dv[v0<1]=-999
                k[v0<1]=-999
                absDV = np.abs(v0-v)/(v0+0.00001)
                v[(absDV>0.01)*(v0>1)]=0
                dv[(absDV>0.01)*(v0>1)]=-999
                k[(absDV>0.01)*(v0>1)]=-999
                std[(absDV>0.01)*(v0>1)]=999
            else:
                v[v0>1]=0
                dv[v0>1]=-999
                k[v0>1]=-999
                absDV = np.abs(v0-v)/(v0+0.00001)
                v[(absDV<0.01)*(v0>1)]=0
                dv[(absDV<0.01)*(v0>1)]=-999
                k[(absDV<0.01)*(v0>1)]=-999
                std[(absDV<0.01)*(v0>1)]=999
        DV.append(dv)
        K.append(k)
        V.append(v)
        STD.append(std)
    return np.array(DV),np.array(K),np.array(V),np.array(STD)

def saveFVD(fvD,stations,quakes='',saveDir='',formatType='pair',isOverwrite=False):
    fileL=[]
    for key in fvD:
        quake=''
        STA0=''
        STA1=''
        if formatType == 'NEFile':
            file = '%s/%s.dat'%(saveDir,key)
        elif formatType == 'pair':
            if '_' not in key:
                continue
            time,la,lo,sta0,sta1=key.split('_')
            STA0 = stations.Find(sta0)
            STA1 = stations.Find(sta1)
            file = '%s/%s/%s_%s/Rayleigh/pvt.dat'%(saveDir,sta0,sta0,sta1)
            quake=seism.Quake(time=float(time),la=float(la),lo=float(lo),dep=0)
            if len(quakes)>0:
                index = quakes.find(quake) 
                if index>=0:
                    quake = quakes[index]
        else:
            file = '%s/%s.dat'%(saveDir,key)
        if (file not in fileL) and isOverwrite:
            writeType='w+'
            print('overwrite',file)
        else:
            writeType='a'
        fvD[key].save(file,mode=formatType,quake=quake,station0=STA0,station1=STA1,writeType=writeType)
        fileL.append(file)

def defineEdge(A,per,minA=-100):
    AL = []
    for i in range(A.shape[1]):
        a = A[:,i]
        a = a[a>minA]
        a.sort()
        index = int(len(a)*per)
        AL.append(a[index])
    return np.array(AL)

def keyDis(key, stations):
    if '_' not in key:
        return 0
    else:
        netSta0, netSta1 = key.split('_')[-2:]
        sta0 = stations.Find(netSta0)
        sta1 = stations.Find(netSta1)
        return sta0.dist(sta1)


def replaceF(fvD, f):
    for key in fvD:
        fvD[key].replaceF(f)
def disQC(fvD, stations, fvRef, randA=0.15, maxR=2):
    for key in fvD:
        fv = fvD[key]
        dis = keyDis(key, stations)
        fv.disQC(fvRef, dis, randA=randA, maxR=maxR)
def coverQC(fvD, stations, minI, maxI, R=0.1):
    for key in fvD:
        fv = fvD[key]
        dis = keyDis(key, stations)
        fv.coverQC(dis, minI, maxI, R)

def keyValue(time, la, lo):
    return float(time) + complex(0.0, 1.0) * (float(la) + float(lo))

def outputFvDist(fvD, stations, t=16 ** np.arange(0, 1.000001, 0.02040816326530612) * 10, keys=[], keysL=[],isSyn=False):
    fL = 1 / t
    fL.sort()
    vL = []
    distL = []
    nameL = []
    fvL = []
    if len(keysL) == 0:
        keysL = fvD.keys()
    KEYL = []
    for key in keysL:
        if 'prem' in key:
            pass
        elif isSyn:
            distL.append(0)
            vL.append(fvD[key](fL))
            fvL.append(fvD[key])
            KEYL.append([0, 0])
        else:
            key0 = key
            netSta0, netSta1 = key.split('_')[-2:]
            name = netSta0 + '_' + netSta1
            name1 = netSta1 + '_' + netSta0
            if len(keys) != 0:
                if name not in keys:
                    if name1 not in keys:
                        continue
            if name not in nameL:
                if name1 not in nameL:
                    nameL.append(name)
            if key not in fvD:
                if len(key.split('_')) == 2:
                    sta0, sta1 = key.split('_')
                    key = '%s_%s' % (sta1, sta0)
                    if key not in fvD:
                        distL.append(0)
                        vL.append(fL * 0 - 1)
                        KEYL.append([key, ''])
                        continue
                else:
                    time, la, lo, sta0, sta1 = key.split('_')
                    key = '%s_%s_%s_%s_%s' % (time, la, lo, sta1, sta0)
                    if key not in fvD:
                        distL.append(0)
                        vL.append(fL * 0 - 1)
                        KEYL.append([key, ''])
                        print(key)
                        continue
            sta0 = stations.Find(netSta0)
            sta1 = stations.Find(netSta1)
            dist = sta0.dist(sta1)
            distL.append(dist)
            vL.append(fvD[key](fL))
            fvL.append(fvD[key])
            KEYL.append([key, key0])
    print(len(nameL))
    return (np.array(distL).reshape([-1, 1]), np.array(vL), fL, averageFVL(fvL, fL=fL), KEYL)

disMap ='gray_r'
def plotFvDist(distL, vL, fL, filename, fStrike=1, title='', isCover=False, minDis=[], maxDis=[], fLDis=[], R=0.001):
    VL = vL.reshape([-1])
    DISTL = (distL + vL * 0).reshape([-1])
    FL = (fL.reshape([1, -1]) + vL * 0).reshape([-1])
    DISTL = DISTL[(VL > 1)]
    FL = FL[(VL > 1)]
    binF = fL[::fStrike]
    binF.sort()
    binD = np.arange(18) / 18 * 1800
    print(DISTL.max())
    plt.close()
    plt.figure(figsize=[3, 2])
    plt.hist2d(DISTL, FL, bins=(binD, binF), rasterized=True, cmap=disMap, norm=(colors.LogNorm()))
    plt.gca().set_yscale('log')
    plt.gca().set_position([0.15,0.2,0.6,0.7])
    ax=plt.gca()
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes('right', size="7%", pad="10%",)
    plt.colorbar(cax=cax,label='count')
    if isCover:
        ax.plot( minDis,fLDis, 'r',linewidth=0.5,label='cover')
        ax.plot( maxDis, fLDis,'r',linewidth=0.5)
        #plt.plot( minDis * (1 - R), fLDis,'-.k',linewidth=0.5,label='control')
        #plt.plot( maxDis * (1 + R), fLDis,'-.k',linewidth=0.5)
        ax.legend()
    ax.set_xlabel('$\Delta$(km)')
    ax.set_ylabel('$f$(Hz)')
    ax.set_xlim([0,1800])
    #plt.title(title)
    #plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plotPair(fvD, stations, filename='predict/pairDis.eps'):
    plt.figure(figsize=[4, 4])
    staL = []
    for key in fvD:
        if 'prem' in key:
            pass
        else:
            netSta0, netSta1 = key.split('_')[-2:]
            sta0 = stations.Find(netSta0)
            sta1 = stations.Find(netSta1)
            if netSta0 not in staL:
                staL.append(netSta0)
                plt.text(sta0['lo'], sta0['la'], str([sta0['la'], sta0['lo']]))
            if netSta1 not in staL:
                staL.append(netSta1)
                plt.text(sta1['lo'], sta1['la'], str([sta1['la'], sta1['lo']]))
            plt.plot([sta0['lo'], sta1['lo']], [sta0['la'], sta1['la']], '--k', linewidth=0.01)
    plt.savefig(filename)


def plotFV(vL, fL, filename, fStrike=1, title='', isAverage=True, thresL=[0.01, 0.02, 0.05], fvAverage={}, isRand=False, randA=0.03, randN=10, midV=4, randR=0.5):
    VL = vL.reshape([-1])
    FL = (fL.reshape([1, -1]) + vL * 0).reshape([-1])
    FL = FL[(VL > 1)]
    VL = VL[(VL > 1)]
    print(len(FL), len(VL))
    if isRand:
        FL0 = FL
        VL0 = VL
        FL = []
        VL = []
        for i in range(len(FL0)):
            f = FL0[i]
            v = VL0[i]
            for loop in range(randN):
                FL.append(f)
                if np.random.rand(1) < randR:
                    rand1 = 1 + randA * (2 * np.random.rand(1) - 1) * np.random.rand(1)
                    V = v / rand1
                else:
                    dT = (np.random.rand(1) - 0.5) * 2 * (1 / midV) * randA * np.random.rand(1)
                    V = 1 / (1 / v - dT)
                VL.append(V[0])

    binF = fL[::fStrike]
    binF.sort()
    binV = np.arange(2.8, 5.2, 0.02)
    plt.close()
    plt.figure(figsize=[3, 2])
    plt.hist2d(VL, FL, bins=(binV, binF), rasterized=True, cmap=disMap, norm=(colors.LogNorm()))
    #plt.colorbar(label='count')
    plt.gca().set_yscale('log')
    plt.xlabel('$v$(km/s)')
    plt.xlim([3, 5])
    plt.ylabel('$f$(Hz)')
    plt.gca().set_position([0.15,0.2,0.6,0.7])
    ax =plt.gca()
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes('right', size="7%", pad="10%",)
    plt.colorbar(cax=cax,label='count')
    #plt.title(title)
    if isAverage:
        linewidth = 0.5
        h0,=ax.plot((fvAverage.v[(fvAverage.v > 1)]), (fvAverage.f[(fvAverage.v > 1)]), '-r', linewidth=linewidth,label='mean')
        for thres in thresL:
            if thres <1:
                h1,=ax.plot((fvAverage.v[(fvAverage.v > 1)] * (1 + thres)), (fvAverage.f[(fvAverage.v > 1)]), '-.r', linewidth=linewidth,label='$\pm$1.5%')
                ax.plot((fvAverage.v[(fvAverage.v > 1)] * (1 - thres)), (fvAverage.f[(fvAverage.v > 1)]), '-.r', linewidth=linewidth)
            else:
                h1,=ax.plot(fvAverage.v[(fvAverage.v > 1)] + thres*fvAverage.std[(fvAverage.v > 1)], (fvAverage.f[(fvAverage.v > 1)]), '-.r', linewidth=linewidth,label='$\pm$%d std'%thres)
                ax.plot(fvAverage.v[(fvAverage.v > 1)] - thres*fvAverage.std[(fvAverage.v > 1)], (fvAverage.f[(fvAverage.v > 1)]), '-.r', linewidth=linewidth)
        ax.legend()
    #plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.figure(figsize=[2.5, 2])
    plt.hist(FL, bins=binF)
    plt.gca().set_xscale('log')
    plt.xlabel('$f$(Hz)')
    plt.ylabel('count')
    plt.savefig((filename[:-4] + '_f' + filename[-4:]), dpi=300)


def compareFVD(fvD, fvDGet, stations, filename, t=12 ** np.arange(0, 1.000001, 0.02040816326530612) * 10, keys=[], fStrike=1, title='err0_dis', threshold=0.015, delta=1, thresholdForGet=0, isCount=False, plotRate=0.6, fvRef={}):
    disL, vL, fL, fvAverage, keyL0 = outputFvDist(fvD, stations, t=t, keys=keys, keysL=(fvD.keys()))
    disLGet, vLGet, fLGet, fvAverageGet, keyL = outputFvDist(fvDGet, stations, t=t, keys=keys, keysL=(fvD.keys()))
    VL = vL.copy()
    DISL = disL.reshape([-1, 1]) + vL * 0
    FL = fL.reshape([1, -1]) + vL * 0
    VLGet = vLGet.copy()
    FLGet = fLGet.reshape([1, -1]) + vLGet * 0
    DISLGet = disLGet.reshape([-1, 1]) + vLGet * 0
    dv = VLGet - VL
    dvR = dv / VL
    thresholdText = '$d_r \\leq %.1f $%%; ' % (threshold * 100)
    if thresholdForGet > 0:
        thresholdText = thresholdText + '$\\sigma \\leq %.1f$%%' % (thresholdForGet * 100)
    else:
        if thresholdForGet < 0:
            thresholdText = thresholdText + '$Prob \\geq %.2f$%%' % -thresholdForGet
    print((vL > 1).sum())
    print(dvR * 100, FL)
    plt.close()
    binF = fL[-1::-fStrike].copy()
    binF.sort()
    binF[(-1)] *= 1.00000001
    binF[0] *= 0.99999999
    maxF = fL.max()
    minF = fL.min()
    maxBinF = binF.max()
    minBinF = binF.min()
    binF = binF.tolist()
    if minF <minBinF:
        binF = [binF[0]**2/binF[1]]+binF
    if maxF >maxBinF:
        binF = binF+[binF[-1]**2/binF[-2]]
    binF =np.array(binF)
    binDVR = np.arange(-0.05, 0.051, 0.001)
    binDis = np.arange(100,1700,60)
    binN = np.arange(0,20,1)
    TL = DISL / Vav
    THRESHOLD = delta / TL
    THRESHOLD[THRESHOLD <= threshold] = threshold
    plt.figure(figsize=[4, 2.25])
    plt.hist2d((dvR[((VL > 1) * (VLGet > 1))]), (FL[((VL > 1) * (VLGet > 1))]), bins=(binDVR, binF), rasterized=True, cmap='Greys')
    plt.gca().set_yscale('log')
    plt.xlabel('$dv/v_0$')
    plt.ylabel('$f$(Hz)')
    plt.yticks([1/100,1/10])
    #plt.title(title)
    #plt.tight_layout()
    #plt.text((binDVR[(-2)]), (binF[(-2)]), thresholdText, va='top', ha='right')
    plt.gca().set_position([0.15,0.2,0.6,0.7])
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes('right', size="7%", pad="10%",)
    plt.colorbar(cax=cax,label='count')
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.figure(figsize=[4, 2.25])
    #plt.hist((FL[(VL > 1)]), bins=binF)
    #plt.hist((FL[((VL > 1) * (VLGet > 1))]), bins=binF)
    #plt.hist((FL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]), bins=binF)
    label=FL[(VL > 1)]
    pick=FL[((VL > 1) * (VLGet > 1))]
    pickRight = FL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]
    plt.hist((label,pick,pickRight), bins=binF,color=['cornflowerblue','yellowgreen','lightcoral'],label=['$T_P+F_N$','$T_P+F_P$','$T_P$'])#['$T_P+F_N$','$T_P+F_P$','$T_P$']
    #plt.hist((pickRight,pick,label,), bins=binF,color=['lightcoral','yellowgreen','cornflowerblue',],label=['$T_P$','$T_P+F_P$','$T_P+F_N$',])
    countAll, a = np.histogram((FL[(VL > 1)]), bins=binF)
    countR, a = np.histogram((FL[((VL > 1) * (VLGet > 1))]), bins=binF)
    countP, a = np.histogram((FL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]), bins=binF)
    binMid = (binF[1:] + binF[:-1]) / 2
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    R =countP / countAll * 100
    P= countP / countR * 100
    F1 = 2/(1/R+1/P)
    hR, = ax2.plot(binMid, R, '-d', markersize=2, linewidth=1,label='R',color='dimgray')
    hP, = ax2.plot(binMid, P, 'o-', markersize=2, linewidth=1,label= 'P',color='seagreen')
    hF1,= ax2.plot(binMid, F1, '.-', markersize=2, linewidth=1,label= 'F1',color='coral')
    #print((countP / countAll)[-2:])
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3)
    #ax1.legend(loc='lower center',ncol=3)
    #ax2.legend(loc='upper left',ncol=3)bbox_to_anchor=(0.5, 0., 0.5, 0.5)
    #ax1.legend(loc='lower center',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    ax1.legend(loc='upper right',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    ax2.legend(loc='upper left',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    plt.gca().set_xscale('log')
    ax1.set_ylabel('count')
    ax1.set_xlabel('$f$(Hz)')
    ax2.set_ylabel('Rate(%)')
    ax1.set_ylim([0, 1.45 * countR.max()])
    #ax2.set_ylim([min(np.min(countP / countAll * 100) - 5, 50), 112])
    ax2.set_ylim([50, 112])
    #ax2.text((binF[-1]*0.96), 110, thresholdText, va='top', ha='right')
    plt.xlim([binF.min()*0.95, binF.max()*1.05])
    plt.xticks([1/100,1/10])
    #plt.title(title)
    #plt.tight_layout()
    ax1.set_position([0.15,0.2,0.6,0.7])
    ax2.set_position([0.15,0.2,0.6,0.7])
    plt.savefig((filename[:-4] + 'FCount' + filename[-4:]), dpi=300)
    plt.close()

    plt.figure(figsize=[4, 2.25])
    label=DISL[(VL > 1)]
    pick=DISL[((VL > 1) * (VLGet > 1))]
    pickRight = DISL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]
    plt.hist((label,pick,pickRight), bins=binDis,color=['cornflowerblue','yellowgreen','lightcoral'],label=['$T_P+F_N$','$T_P+F_P$','$T_P$'])#['$T_P+F_N$','$T_P+F_P$','$T_P$']
    #plt.hist((pickRight,pick,label,), bins=binF,color=['lightcoral','yellowgreen','cornflowerblue',],label=['$T_P$','$T_P+F_P$','$T_P+F_N$',])
    countAll, a = np.histogram((DISL[(VL > 1)]), bins=binDis)
    countR, a = np.histogram((DISL[((VL > 1) * (VLGet > 1))]), bins=binDis)
    countP, a = np.histogram((DISL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]), bins=binDis)
    binMid = (binDis[1:] + binDis[:-1]) / 2
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    R =countP / countAll * 100
    P= countP / countR * 100
    F1 = 2/(1/R+1/P)
    hR, = ax2.plot(binMid, R, '-d', markersize=2, linewidth=1,label='R',color='dimgray')
    hP, = ax2.plot(binMid, P, 'o-', markersize=2, linewidth=1,label= 'P',color='seagreen')
    hF1,= ax2.plot(binMid, F1, '.-', markersize=2, linewidth=1,label= 'F1',color='coral')
    #print((countP / countAll)[-2:])
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3)
    #ax1.legend(loc='lower center',ncol=3)
    #ax2.legend(loc='upper left',ncol=3)bbox_to_anchor=(0.5, 0., 0.5, 0.5)
    #ax1.legend(loc='lower center',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    ax1.legend(loc='upper right',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    ax2.legend(loc='upper left',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    #plt.gca().set_xscale('log')
    ax1.set_ylabel('count')
    ax1.set_xlabel('$\Delta$(km)')
    ax2.set_ylabel('Rate(%)')
    ax1.set_ylim([0, 1.45 * countR.max()])
    ax2.set_ylim([min(np.min(countP[countAll>0] / countAll[countAll>0] * 100) - 5, 50), 112])
    #ax2.text((binDis[-1]*0.96), 110, thresholdText, va='top', ha='right')
    plt.xlim([binDis.min()*0.95, binDis.max()*1.05])
    #plt.xticks([1/100,1/10])
    #plt.title(title)
    #plt.tight_layout()
    ax1.set_position([0.15,0.2,0.6,0.7])
    ax2.set_position([0.15,0.2,0.6,0.7])
    plt.savefig((filename[:-4] + 'DisCount' + filename[-4:]), dpi=300)
    plt.close()

    plt.figure(figsize=[4, 2.25])
    NL= DISL/(VL+0.0000001)*FL
    label=NL[(VL > 1)]
    pick=NL[((VL > 1) * (VLGet > 1))]
    pickRight = NL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]
    plt.hist((label,pick,pickRight), bins=binN,color=['cornflowerblue','yellowgreen','lightcoral'],label=['$T_P+F_N$','$T_P+F_P$','$T_P$'])#['$T_P+F_N$','$T_P+F_P$','$T_P$']
    #plt.hist((pickRight,pick,label,), bins=binF,color=['lightcoral','yellowgreen','cornflowerblue',],label=['$T_P$','$T_P+F_P$','$T_P+F_N$',])
    countAll, a = np.histogram((NL[(VL > 1)]), bins=binN)
    countR, a = np.histogram((NL[((VL > 1) * (VLGet > 1))]), bins=binN)
    countP, a = np.histogram((NL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]), bins=binN)
    binMid = (binN[1:] + binN[:-1]) / 2
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    R =countP / countAll * 100
    P= countP / countR * 100
    F1 = 2/(1/R+1/P)
    hR, = ax2.plot(binMid, R, '-d', markersize=2, linewidth=1,label='R',color='dimgray')
    hP, = ax2.plot(binMid, P, 'o-', markersize=2, linewidth=1,label= 'P',color='seagreen')
    hF1,= ax2.plot(binMid, F1, '.-', markersize=2, linewidth=1,label= 'F1',color='coral')
    #print((countP / countAll)[-2:])
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3)
    #ax1.legend(loc='lower center',ncol=3)
    #ax2.legend(loc='upper left',ncol=3)bbox_to_anchor=(0.5, 0., 0.5, 0.5)
    #ax1.legend(loc='lower center',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    ax1.legend(loc='upper right',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    ax2.legend(loc='upper left',ncol=3,handlelength=1,handletextpad=0.2,columnspacing=0.4,borderaxespad=0.2)
    #plt.gca().set_xscale('log')
    ax1.set_ylabel('count')
    ax1.set_xlabel('$t/T$')
    ax2.set_ylabel('Rate(%)')
    ax1.set_ylim([0, 1.45 * countR.max()])
    ax2.set_ylim([min(np.min(countP[countAll>0] / countAll[countAll>0] * 100) - 5, 50), 112])
    #ax2.text((binN[-1]*0.96), 110, thresholdText, va='top', ha='right')
    plt.xlim([binN.min()*0.95, binN.max()*1.05])
    #plt.xticks([1/100,1/10])
    #plt.title(title)
    #plt.tight_layout()
    ax1.set_position([0.15,0.2,0.6,0.7])
    ax2.set_position([0.15,0.2,0.6,0.7])
    plt.savefig((filename[:-4] + 'NCount' + filename[-4:]), dpi=300)
    plt.close()

    if isCount:
        vMean = fL * 0
        for i in range(len(fL)):
            vMean[i] = VL[(VL[:, i] > 1, i)].mean()

        resDir = os.path.dirname(filename) + '/countByF/'
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        count = np.zeros([len(fL), len(stations), len(stations)])
        countHit = np.zeros([len(fL), len(stations), len(stations)])
        countRight = np.zeros([len(fL), len(stations), len(stations)])
        for i in range(len(keyL)):
            key = keyL[i][0].split('_')[-2:]
            i0 = (stations.index)(*key[0].split('.'))
            i1 = (stations.index)(*key[1].split('.'))
            if i1 < i0:
                i1, i0 = i0, i1
            v = VL[i]
            vGet = VLGet[i]
            count[(v > 1, i0, i1)] += 1
            countHit[((v > 1) * vGet > 0, i0, i1)] += 1
            countRight[((v > 1) * (vGet > 1) * (np.abs(dvR[i]) <= THRESHOLD[i]), i0, i1)] += 1

        perHit = countHit / count
        perHit[count == 0] = np.nan
        perRight = countRight / countHit
        perRight[countHit == 0] = np.nan
        plt.close()
        #if not isinstance(fvRef, dict):
        #    vMean = fvRef(fL)
        for i in range(len(fL)):
            plt.figure(figsize=[8, 6])
            plt.subplot(2, 3, 1)
            plt.pcolor((perHit[i]), vmax=(2 * plotRate), vmin=0, cmap='bwr')
            plt.colorbar()
            plt.subplot(2, 3, 2)
            plt.plot([0.25, 0.25], [0, 1], 'r', linewidth=0.5)
            plt.plot([12, 12], [0, 1], 'r', linewidth=0.5)
            for j in range(len(stations)):
                for k in range(j, len(stations)):
                    if perHit[(i, j, k)] > -1:
                        VMean= vMean[i]
                        plt.subplot(2, 3, 2)
                        plt.plot((stations[j].dist(stations[k]) / VMean * fL[i]), (perHit[(i, j, k)]), '.k', markersize=0.5)
                    if perHit[(i, j, k)] < plotRate:
                        plt.subplot(2, 3, 3)
                        plt.plot([stations[j]['lo'], stations[k]['lo']], [stations[j]['la'], stations[k]['la']], '-.k', linewidth=0.2)

            plt.subplot(2, 3, 4)
            plt.pcolor((perRight[i]), vmax=(2 * plotRate), vmin=0, cmap='bwr')
            plt.colorbar()
            plt.subplot(2, 3, 5)
            plt.plot([0.25, 0.25], [0, 1], 'r', linewidth=0.5)
            plt.plot([12, 12], [0, 1], 'r', linewidth=0.5)
            for j in range(len(stations)):
                for k in range(j, len(stations)):
                    if perRight[(i, j, k)] > -1:
                        VMean= vMean[i]
                        plt.subplot(2, 3, 5)
                        plt.plot((stations[j].dist(stations[k]) / VMean * fL[i]), (perRight[(i, j, k)]), '.k', markersize=0.5)
                    if perRight[(i, j, k)] < plotRate:
                        plt.subplot(2, 3, 6)
                        plt.plot([stations[j]['lo'], stations[k]['lo']], [stations[j]['la'], stations[k]['la']], '-.k', linewidth=0.2)

            plt.savefig(('%s/%.2fmHZ.jpg' % (resDir, fL[i] * 1000)), dpi=300)
            plt.close()
    RSum =countP.sum() / countAll.sum() 
    PSum= countP.sum() / countR.sum() 
    F1Sum = 2/(1/RSum+1/PSum)
    mean = dvR[np.abs(dvR)<3*threshold].mean()
    std = dvR[np.abs(dvR)<3*threshold].std()
    return  '%.3f&%.4f&%.4f&%.4f&%.3f&%.3f'%(threshold*100,RSum,PSum,F1Sum,mean*100,std*100)


def compareInF(fvD0, fvD1, stations, fL, isSave=True, saveDir='predict/compareInF/', R=[]):
    M = np.zeros([len(fL), len(stations), len(stations)])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    V0 = M * 0
    V1 = M * 0
    for key in fvD0:
        if '_' not in key:
            pass
        else:
            fv0 = fvD0[key]
            netSta0, netSta1 = key.split('_')[-2:]
            key0 = '%s_%s' % (netSta0, netSta1)
            key1 = '%s_%s' % (netSta1, netSta0)
            net0, sta0 = netSta0.split('.')
            net1, sta1 = netSta1.split('.')
            i0 = stations.index(net0, sta0)
            i1 = stations.index(net1, sta1)
            v0 = fv0(fL)
            if key0 in fvD1:
                fv1 = fvD1[key0]
            elif key1 in fvD1:
                fv1 = fvD1[key1]
            else:
                continue
            v1 = fv1(fL)
            V0[:, i0, i1] = v0
            V1[:, i0, i1] = v1
            M[:, i0, i1] = (v1 - v0) / v0
            M[(v1 < 1, i0, i1)] = 0
            M[(v0 < 1, i0, i1)] = 0

    MS = np.abs(M).sum(axis=0) / (np.abs(M) > 0).sum(axis=0)
    if isSave:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        with open(saveDir + 'anormStation', 'w+') as (F):
            for i in range(len(fL)):
                plt.close()
                plt.figure(figsize=[3, 6])
                plt.subplot(2, 1, 1)
                plt.pcolor((np.abs(M[i])), vmax=0.05, vmin=0, cmap='bwr')
                plt.title('%.3f mHz' % (fL[i] * 1000))
                plt.subplot(2, 1, 2)
                for j in range(len(stations)):
                    for k in range(len(stations)):
                        if np.abs(M[(i, j, k)]) > 0.03:
                            plt.plot([stations[j]['lo'], stations[k]['lo']], [stations[j]['la'], stations[k]['la']], '.-k', linewidth=1)
                            F.write('%.1f mHz %.2f  %d %s %s %d %s %s\n' % (fL[i] * 1000, M[(i, j, k)] * 100, j, stations[j]['net'], stations[j]['sta'], k, stations[k]['net'], stations[k]['sta']))

                if len(R) > 0:
                    plt.ylim(R[:2])
                    plt.xlim(R[2:])
                plt.savefig('%s/%.1fmHz.eps' % (saveDir, fL[i] * 1000))
                plt.close()

            plt.close()
            plt.figure(figsize=[3, 3])
            plt.pcolor(MS, vmax=0.05, vmin=0, cmap='bwr')
            plt.title('%s' % 'mean')
            plt.savefig('%s/%s.eps' % (saveDir, 'mean'))
            plt.close()
            plt.close()
            plt.figure(figsize=[3, 3])
            for i in range(len(stations)):
                for j in range(len(stations)):
                    if MS[(i, j)] > 0.025:
                        plt.plot([stations[i]['la'], stations[j]['la']], [stations[i]['lo'], stations[j]['lo']], '-.k', linewidth=1)

            plt.savefig('%s/%s.eps' % (saveDir, 'two-station'))
            plt.close()
    return (
     M, V0, V1)

def keyConvert(key):
    tmp=key.split('_')
    if len(tmp)>=2:
        if tmp[-2]>tmp[-1]:
            tmp[-1],tmp[-2]=tmp[-2:]
        key=tmp[0]
        for t in tmp[1:]:
            key=key+'_'+t
    return key
def keyPair(key):
    key = keyConvert(key)
    tmp=key.split('_')
    if len(tmp)>=2:
        return tmp[-2]+'_'+tmp[-1]

def getFVFromPairFile(file, fvD={}, quakeD=seism.QuakeL(), isPrint=False, isAll=False, isGetQuake=True,stations='',isCheck=True,time0=0,staD='',dRatioD={}):
    with open(file) as (f):
        lines = f.readlines()
    stat = 'next'
    fileName = file.split('/')[(-1)]
    if fileName[:3] == 'pvt':
        staDir = file.split('/')[(-3)]
    else:
        staDir = fileName.split('-')[0]
    pairKey = staDir
    largeCount=0
    allCount=0
    staName0,staName1 = pairKey.split('_')
    if isinstance(staD,dict):
        if staName0 not in staD:
            staD[staName0]=[]
        if staName1 not in staD:
            staD[staName1]=[]
    STA0 = stations.Find(staName0)
    STA1 = stations.Find(staName1)
    loc0 = np.array([STA0['la'],STA0['lo']])
    loc1 = np.array([STA1['la'],STA1['lo']])
    for line in lines:
        if len(line.split()) == 0:
            if isPrint:
                print(file, line)
        elif stat == 'next':
            sta0 = line.split()[0]
            sta1 = line.split()[1]
            stat = 'comp0'
        elif stat == 'comp0':
            stat = 'quakeTime0'
        elif stat == 'quakeTime0':
            timeL = [int(tmp) for tmp in line.split()[:-1]]
            timeL.append(float(line.split()[(-1)]))
            time = (obspy.UTCDateTime)(*timeL)
            stat = 'sta1Loc'
        elif stat == 'sta1Loc':
            sta1La, sta1Lo = line.split()[:2]
            if len(line.split()) > 2:
                stat = 'deltaLoc0'
                sta1La = float(sta1La)
                sta1Lo = float(sta1Lo[:10])
                la = float(line.split()[1][10:])
                lo = float(line.split()[2])
                dep = float(line.split()[3])
                ml = float(line.split()[4])
                continue
            sta1La = float(sta1La)
            sta1Lo = float(sta1Lo)
            stat = 'QuakeInfo0'
        elif stat == 'QuakeInfo0':
            la = float(line.split()[0])
            lo = float(line.split()[1])
            dep = float(line.split()[2])
            ml = float(line.split()[3])
            stat = 'deltaLoc0'
        elif stat == 'deltaLoc0':
            Dist0 = float(line.split()[0])
            stat = 'comp1'
        elif stat == 'comp1':
            stat = 'quakeTime1'
        elif stat == 'quakeTime1':
            stat = 'sta0Loc'
        elif stat == 'sta0Loc':
            sta0La, sta0Lo = line.split()[:2]
            if len(line.split()) > 2:
                stat = 'deltaLoc1'
                sta0La = float(sta0La)
                sta0Lo = float(sta0Lo[:10])
                la = float(line.split()[1][10:])
                lo = float(line.split()[2])
                dep = float(line.split()[3])
                ml = float(line.split()[4])
                continue
            sta0La = float(sta0La)
            sta0Lo = float(sta0Lo)
            stat = 'QuakeInfo1'
        elif stat == 'QuakeInfo1':
            stat = 'deltaLoc1'
        elif stat == 'deltaLoc1':
            Dist1 = float(line.split()[0])
            stat = 'fNum'
        elif stat == 'fNum':
            fNum = int(line.split()[1])
            if fNum == 0:
                stat = 'next'
                continue
            f = np.zeros(fNum)
            v = np.zeros(fNum)
            std = np.zeros(fNum)
            stat = 'f'
            i = 0
            if isAll:
                f = [[]] * fNum
                std = [[]] * fNum
        elif stat == 'f':
            f[i] = float(line)
            i += 1
            if i == fNum:
                stat = 'v'
                i = 0
        elif stat == 'v':
            lineS = line.split()
            v[i] = float(lineS[0])
            if len(lineS) > 1:
                std[i] = float(lineS[1])
            if isAll:
                v_prob = np.array([float(tmp) for tmp in lineS]).reshape([-1, 2])
                v[i] = v_prob[:, 0]
                std[i] = v_prob[:, 1]
            i += 1
            if i == fNum:
                if isinstance(staD,dict):
                    d0 = (loc0[0]-sta0La)**2+(loc0[1]-sta0Lo)**2+ (loc1[0]-sta1La)**2+(loc1[1]-sta1Lo)**2
                    d1 = (loc1[0]-sta0La)**2+(loc1[1]-sta0Lo)**2+ (loc0[0]-sta1La)**2+(loc0[1]-sta1Lo)**2
                    if d0<d1:
                        staD[staName0].append([time.timestamp,sta0La,sta0Lo])
                        staD[staName1].append([time.timestamp,sta1La,sta1Lo])
                    else:
                        if d0<d1:
                            staD[staName1].append([time.timestamp,sta0La,sta0Lo])
                            staD[staName0].append([time.timestamp,sta1La,sta1Lo])
                stat = 'next'
                az0 = DistAz(la, lo, sta0La, sta0Lo).getAz()
                dis0 = DistAz(la, lo, sta0La, sta0Lo).getDelta()
                #v0 = dis0*111.19/(IASP91(dep,dis0))
                az1 = DistAz(la, lo, sta1La, sta1Lo).getAz()
                dis1 = DistAz(la, lo, sta1La, sta1Lo).getDelta()
                #v1 = dis1*111.19/(IASP91(dep,dis1))
                baz0 = DistAz(la, lo, sta0La, sta0Lo).getBaz()
                az01 = DistAz(sta0La, sta0Lo, sta1La, sta1Lo).getAz()
                #vMin = min(v0,v1)
                if (az0 - az1 + 10) % 360 > 20:
                    continue
                if (baz0 - az01 + 10) % 180 > 20:
                    continue
                quake = seism.Quake(time=time, la=la, lo=lo, dep=dep, ml=ml)
                if isGetQuake:
                    index = quakeD.find(quake)
                    if index < 0:
                        quakeD.append(quake)
                    else:
                        quake = quakeD[index]

                name = quake.name('_', fmt='%.5f')
                key = keyConvert('%s_%s' % (name, pairKey))
                if len(f) < 2:
                    continue
                f = np.array(f)
                v= np.array(v)
                std= np.array(std)
                if len(stations)>0:
                    #STA0 =stations.findByLoc(sta0La,sta0Lo)
                    #STA1 =stations.findByLoc(sta1La,sta1Lo)
                    DDist_cal = np.abs(Dist1-Dist0)
                    DDist_real = np.abs(quake.distaz(STA0).getDelta()-quake.distaz(STA1).getDelta())
                    Ratio = DDist_real/DDist_cal
                    allCount+=1
                    if time.timestamp<time0:
                        print('ealier than time0',time0)
                        continue
                    if np.abs(Ratio-1)>0.001:
                        print('**********to large    differnt',time.strftime('%Y:%m:%d-%H:%M'),sta0,sta1,Ratio,allCount,largeCount)
                        print(sta0La,sta0Lo,sta1La,sta1Lo)
                        print(STA0['la'],STA0['lo'],STA1['la'],STA1['lo'])
                    else:
                        pass
                        #print('**********to a little differnt',time.strftime('%Y:%m:%d-%H:%M'),sta0,sta1,Ratio,allCount,largeCount)
                    if isCheck and np.abs(Ratio-1)>0.015:
                        largeCount+=1
                        print('remove large difference')
                        continue
                    if time.timestamp<time0:
                        print('ealier than time0',time0)
                        continue
                    if isCheck:
                        v = np.array(v)*Ratio
                #std=std[v<vMin]
                #v=v[v<vMin]
                dRatio=np.abs(Ratio-1)
                if key in dRatioD and isCheck:
                    if dRatio>dRatioD[key]:
                        print('lager; pass')
                        continue
                dRatioD[key] = dRatio
                if isAll:
                    fvD[key] = fv([f, v, std], mode='dis')
                else:
                    fvD[key] = fv([f, v, std])
        if isPrint:
            print(len(quakeD.keys()))
    if largeCount/(allCount+0.001)>0.2 and allCount>0:
        print('badPair',largeCount,allCount,sta0,sta0La,sta0Lo,sta1,sta1La,sta1Lo)

def getFVFromPairFileDis(file, fvD={}, quakeD={}, isPrint=True):
    with open(file) as (f):
        lines = f.readlines()
    stat = 'next'
    fileName = file.split('/')[(-1)]
    if fileName[:3] == 'pvt':
        staDir = file.split('/')[(-3)]
    else:
        staDir = fileName.split('-')[0]
    pairKey = staDir
    for line in lines:
        if len(line.split()) == 0:
            print(file, line)
        elif stat == 'next':
            sta0 = line.split()[0]
            sta1 = line.split()[1]
            stat = 'comp0'
        elif stat == 'comp0':
            stat = 'quakeTime0'
        elif stat == 'quakeTime0':
            timeL = [int(tmp) for tmp in line.split()[:-1]]
            timeL.append(float(line.split()[(-1)]))
            time = (obspy.UTCDateTime)(*timeL)
            stat = 'sta1Loc'
        elif stat == 'sta1Loc':
            sta1La, sta1Lo = line.split()[:2]
            if len(line.split()) > 2:
                stat = 'deltaLoc0'
                sta1La = float(sta1La)
                sta1Lo = float(sta1Lo[:10])
                la = float(line.split()[1][10:])
                lo = float(line.split()[2])
                dep = float(line.split()[3])
                ml = float(line.split()[4])
                continue
            sta1La = float(sta1La)
            sta1Lo = float(sta1Lo)
            stat = 'QuakeInfo0'
        elif stat == 'QuakeInfo0':
            la = float(line.split()[0])
            lo = float(line.split()[1])
            dep = float(line.split()[2])
            ml = float(line.split()[3])
            stat = 'deltaLoc0'
        elif stat == 'deltaLoc0':
            stat = 'comp1'
        elif stat == 'comp1':
            stat = 'quakeTime1'
        elif stat == 'quakeTime1':
            stat = 'sta0Loc'
        elif stat == 'sta0Loc':
            sta0La, sta0Lo = line.split()[:2]
            if len(line.split()) > 2:
                stat = 'deltaLoc1'
                sta0La = float(sta0La)
                sta0Lo = float(sta0Lo[:10])
                la = float(line.split()[1][10:])
                lo = float(line.split()[2])
                dep = float(line.split()[3])
                ml = float(line.split()[4])
                continue
            sta0La = float(sta0La)
            sta0Lo = float(sta0Lo)
            stat = 'QuakeInfo1'
        elif stat == 'QuakeInfo1':
            stat = 'deltaLoc1'
        elif stat == 'deltaLoc1':
            stat = 'fNum'
        elif stat == 'fNum':
            fNum = int(line.split()[1])
            if fNum == 0:
                stat = 'next'
                continue
            f = np.zeros(fNum)
            v = []
            p = []
            std = np.zeros(fNum)
            stat = 'f'
            i = 0
        elif stat == 'f':
            f[i] = float(line)
            i += 1
            if i == fNum:
                stat = 'v'
                i = 0
        elif stat == 'v':
            lineS = line.split()
            NS = len(lineS)
            v.append([float(s) for s in lineS[0::2]])
            p.append([float(s) for s in lineS[1::2]])
            i += 1
            if i == fNum:
                stat = 'next'
                az0 = DistAz(la, lo, sta0La, sta0Lo).getAz()
                az1 = DistAz(la, lo, sta1La, sta1Lo).getAz()
                baz0 = DistAz(la, lo, sta0La, sta0Lo).getBaz()
                az01 = DistAz(sta0La, sta0Lo, sta1La, sta1Lo).getAz()
                if (az0 - az1 + 10) % 360 > 20:
                    continue
                if (baz0 - az01 + 10) % 180 > 20:
                    continue
                quake = seism.Quake(time=time, la=la, lo=lo, dep=dep, ml=ml)
                name = quake.name('_')
                if name not in quakeD:
                    quakeD[name] = quake
                key = keyConvert('%s_%s' % (name, pairKey))
                if len(f) < 2:
                    continue
                fvD[key] = fv([f, v, p], mode='dis')
        if isPrint:
            print(len(quakeD.keys()))

def qcFvD(fvD, threshold=-2, minCount=3, delta=-1, stations=''):
    keyL = []
    threshold0 = np.array(threshold).copy()
    for key in fvD:
        if threshold0 > 0:
            threshold = threshold0
            if delta > 0:
                if '_' in key:
                    dis = keyDis(key, stations)
                    T = dis / Vav
                    threshold = max(delta / T, threshold0)
                    if threshold > threshold0:
                        print(dis, delta / T, threshold)
        if threshold0 > -1:
            fvD[key].qc(threshold)
        if (fvD[key].v>0.1).sum() < minCount:
            if key == '1300195681.40000_37.67000_142.13000_JL.CBT_HE.KAB':
                print('1300195681.40000_37.67000_142.13000_JL.CBT_HE.KAB')
            keyL.append(key)

    for key in keyL:
        fvD.pop(key)

def qcFvL(fvL, minCount=3):
    FVL = []
    count = 0
    for fv in fvL:
        if len(fv.f) < minCount:
            FVL.append(count)
        count += 1
    for fv in FVL[-1::-1]:
        fvL.pop(fv)

def wFunc(w, weightType):
    w = w.copy()
    if weightType == 'std':
        w[w != 0] = 0.03 / w[(w != 0)]
    else:
        if weightType == 'prob':
            w[w != 0] = np.exp(-w[(w != 0)] * 3)
    return w

def getFVByCluster(VDVK,f,minSta=5):
    vL =[]
    fL = []
    stdL =[]
    for i in range(VDVK.shape[2]):
        v  = VDVK[:,0,i]
        dv = VDVK[:,1,i]
        k  = VDVK[:,2,i]
        v  = v[k>=0]
        dv  = dv[k>=0]
        k  = k[k>=0]
        if len(v)<minSta:
            continue
        nv = v/v.std()
        ndv = dv/dv.std()
        nk = k/k.std()
        c,l=cluster.mean_shift(np.array([nv,ndv,nk]).transpose())
        vL.append( v[l==0].mean())
        stdL.append( v[l==0].std())
        fL.append(f[i])
    return fv(np.array([fL,vL,stdL]))

def averageFVL(fvL, minSta=5, threshold=0.82285, minThreshold=0.02, fL=[], isWeight=False, weightType='std',it=1,isCluster=False):
    qcFvL(fvL)
    if len(fvL) < minSta:
        return fv([np.array([-1]), np.array([-1]), np.array([10])])
    else:
        if len(fL) == 0:
            fL = []
            for FV in fvL:
                f = FV.f
                for F in f:
                    if F not in fL:
                        fL.append(F)
            fL.sort()
            fL = np.array(fL)
        if isCluster:
            VDVK = np.array([[fv(f)]+list(fv.getDVK(f)) for fv in fvL])
            return getFVByCluster(VDVK,f,minSta=minSta)
        vM = np.zeros([len(fL), len(fvL)])
        if isWeight:
            wM = np.zeros([len(fL), len(fvL)])
        for i in range(len(fvL)):
            vM[:, i] = fvL[i](fL)
            if isWeight:
                wM[:, i] = wFunc(fvL[i].STD(fL), weightType)
        vCount = (vM > 1).sum(axis=1)
        #print(vCount)
        f = fL[(vCount >= minSta)]
        vMNew = vM[(vCount >= minSta)]
        if isWeight:
            wMNew = wM[(vCount >= minSta)]
        std = f * 0
        v = f * 0
        for i in range(len(f)):
            if isWeight:
                #print('w')
                MEAN, STD, vN = QC((vMNew[i][(vMNew[i] > 1)]), threshold=threshold, minThreshold=minThreshold, minSta=minSta, wL=(wMNew[i][(vMNew[i] > 1)]), it=it)
            else:
                MEAN, STD, vN = QC((vMNew[i][(vMNew[i] > 1)]), threshold=threshold, minThreshold=minThreshold, minSta=minSta, it=it)
            v[i] = MEAN
            std[i] = STD
            #print(MEAN,STD,vMNew[i][(vMNew[i] > 1)])
        return fv([f, v, std])

def averageFVDis(fvL, minSta=5, threshold=2.5):
    fL = []
    VL = np.arange(1.5, 6, 0.005)
    for FV in fvL:
        f = FV.f
        for F in f:
            if F not in fL:
                fL.append(F)
    FL = np.array(fL)
    vM = np.zeros[(len(FL), len(VL))]
    for fv in fvL:
        for i in range(fv.f):
            f = fv.f[i]
            vL = fv.v[i]
            prob = fv.std[i]
            fIndex = np.abs(FL - f).argmin()
            for j in range(len(vL)):
                vIndex = np.abs(VL - vL[i]).argmin()
                if np.abs(VL - vL[i]).min() > 0.005:
                    continue
                vM[(fIndex, vIndex)] += prob[j]
                if vIndex > 0:
                    vM[(fIndex, vIndex - 1)] += prob[j] / 2
                if vIndex < len(VL) - 1:
                    vM[(fIndex, vIndex + 1)] += prob[j] / 2

    fL = []
    vL = []
    std = []
    for i in range(len(FL)):
        v = VL[vM[i].argmax()]
        prob = vM[i].max()
        stdProb = vM[i].std()
        mul = prob / stdProb
        if mul > 5:
            fL.append(FL[i])
            vL.append(v)
            std.append(-mul)
    return fv([np.array(f), np.array(v), np.array(std)])


def fvD2fvM(fvD, isDouble=False):
    fvM = {}
    N = len(fvD)
    count = 0
    for key in fvD:
        count += 1
        if count % 1000 == 0:
            print('toM: %d/%d' % (count, N))
        time, la, lo, sta0, sta1 = key.split('_')
        keyNew = sta0 + '_' + sta1
        if isDouble:
            if sta0 > sta1:
                keyNew = sta1 + '_' + sta0
        if keyNew not in fvM:
            fvM[keyNew] = []
        fvM[keyNew].append(fvD[key])

    return fvM


def fvM2Av(fvM, **kwags):
    fvD = {}
    N = len(fvM)
    count = 0
    for key in fvM:
        count += 1
        if count % 100 == 0:
            print('Average: %d/%d' % (count, N))
        fvD[key] = averageFVL((fvM[key]), **kwags)

    return fvD


def plotFVM(fvM,fvD={}, fvDRef={}, resDir='test/', isDouble=False, stations=[], keyL=[],fvMRef={},format='jpg', **kwags):
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    if len(keyL) == 0:
        keyL = fvM
    for key in keyL:
        key0 = key
        filename = resDir + key + '.'+format
        dist = -1
        fvRef = None
        if key in fvM:
            fvL = fvM[key]
        else:
            sta1, sta0 = key.split('_')
            key = sta0 + '_' + sta1
            if key in fvM:
                fvL = fvM[key]
            else:
                continue
        sta0, sta1 = key.split('_')
        dist = keyDis(key, stations)
        if isDouble:
            keyNew = sta1 + '_' + sta0
            if keyNew in fvM:
                fvL += fvM[keyNew]
        if key in fvD:
            fvMean = fvD[key]
            if key in fvMRef:
                fvLRef = fvMRef[key]
            else:
                fvLRef = []
            if key in fvDRef:
                fvRef = fvDRef[key0]
                plotFVL(fvL, fvMean, fvRef, filename=filename, title=key, dist=dist,fvLRef=fvLRef, **kwags)
            else:
                plotFVL(fvL, fvMean, None, filename=filename, title=key, dist=dist,fvLRef=fvLRef, **kwags)

def plotFVL(fvL, fvMean=None, fvRef=None, filename='test.jpg', thresholdL=[1],threshold=0.015, title='fvL', fL0=[], dist=-1,fvLRef=[]):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.close()
    hL = []
    lL = []
    plt.figure(figsize=[4, 2.5])
    plt.subplots_adjust(left=0.15,bottom=0.21,wspace=0.25)
    plt.subplot(1, 2, 1)
    for fv in fvL:
        if isinstance(fvL, dict):
            fv = fvL[fv]
        if len(fv.f) > 2:
            if len(fL0) > 0:
                fL = fL0.copy()
            else:
                fL = fv.f
            v = fv(fL)
            v[v < 1] = np.nan
            h, = plt.plot(v, fL, 'k', linewidth=0.1, label='single')
    hL.append(h)
    lL.append('single')
    for fv in fvLRef:
        if isinstance(fvL, dict):
            fv = fvL[fv]
        if len(fv.f) > 2:
            if len(fL0) > 0:
                fL = fL0.copy()
            else:
                fL = fv.f
            v = fv(fL)
            v[v < 1] = np.nan
            h, = plt.plot(v, fL, '--b', linewidth=0.1, label='manual')

    hL.append(h)
    lL.append('manual')
    if fvMean != None:
        if len(fL0) > 0:
            fL = fL0.copy()
        else:
            fL = fv.f
        v = fvMean(fL)
        std = fvMean.STD(fL)
        std[v < 1] = np.nan
        v[v < 1] = np.nan
        for thres in thresholdL:
            plt.plot((v - thres * std), fL, '-.r', linewidth=0.3, label='$\\pm$std')
            h1, = plt.plot((v + thres * std), fL, '-.r', linewidth=0.3)

        h2, = plt.plot(v, fL, 'r', linewidth=0.3, label='mean')
        hL.append(h2)
        lL.append('mean')
        hL.append(h1)
        lL.append('$\\pm$std')
    figSet()
    plt.legend(hL, lL)
    if dist > 0:
        plt.suptitle('%s %.1f km' % (title, dist))
    else:
        plt.suptitle('%s ' % (title,))
    plt.xlabel('$v$(km/s)')
    plt.ylabel('$f$(Hz)')
    plt.ylim([fL.min(), fL.max()])
    plt.xlim([3, 5])
    #plt.tight_layout()
    plt.subplot(1, 2, 2)
    if fvRef != None:
        f = fL
        fv = fvMean
        if len(fvRef.f) > 2:
            v = fv(f)
            std = fv.STD(f)
            vRef = fvRef(f)
            v[v < 1] = np.nan
            vRef[vRef < 1] = np.nan
            hRef, = plt.plot(vRef, f, 'b', linewidth=0.3, label='manual')
            hGet, = plt.plot(v, f, 'r', linewidth=0.3, label='predict')
            hD, = plt.plot((vRef * 0.985), f, '-.b', linewidth=0.3, label='$\\pm$ %.1f %%'%(threshold*100))
            plt.plot((vRef * 1.015), f, '-.b', linewidth=0.3)
            plt.legend()
            plt.gca().set_yscale('log')
            plt.ylim([f.min(), f.max()])
            plt.xlim([3, 5])
            plt.xlabel('$v$(km/s)')
    #plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def figSet():
    plt.xlim([3, 5])
    plt.ylim([0.00625, 0.1])
    plt.gca().semilogy()
    plt.xlabel('v/(km/s)')
    plt.ylabel('f/Hz')


def fvD2fvL(fvD, stations, f, isRef=False, fvRef=None,threshold=-1):
    indexL = [[] for station in stations]
    vL = [[] for station in stations]
    for i in range(len(stations)):
        for j in range(i, len(stations)):
            sta0 = stations[i]
            sta1 = stations[j]
            key = '%s_%s' % (sta0.name('.'), sta1.name('.'))
            key1 = '%s_%s' % (sta1.name('.'), sta0.name('.'))
            isIn = False
            if key in fvD:
                isIn = True
            else:
                if key1 in fvD:
                    isIn = True
                    key = key1
            if isIn:
                indexL[i].append(j)
                v = fvD[key](f)
                if threshold>0:
                    std = fvD[key].STD(f)
                    v[std/v>threshold]=0
                if isRef:
                    vRef = fvRef(f)
                    v[v > 1] = vRef[(v > 1)]
                vL[i].append(v)
    return (
     indexL, vL)


def getDisCover(fvD, stations, fL):
    minDis = fL * 1000000000.0
    maxDis = fL * 0
    for key in fvD:
        if '_' not in key:
            pass
        else:
            dis = keyDis(key, stations)
            v = fvD[key](fL)
            minDis[(v > 1) * (dis < minDis)] = dis
            maxDis[(v > 1) * (dis > maxDis)] = dis
    maxDis[0]=maxDis[1]
    maxDis[-1]=maxDis[-2]
    minDis[0]=minDis[1]
    minDis[-1]=minDis[-2]
    minI = interpolate.interp1d(fL, minDis)
    maxI = interpolate.interp1d(fL, maxDis)
    return (minDis, maxDis, minI, maxI)


def replaceByAv(fvD, fvDAv, threshold=0, delta=-1, stations='',isAv=True, **kwags):
    notL = []
    threshold0 = threshold
    for modelName in fvD:
        if len(modelName.split('_')) >= 2 and isAv:
            name0 = modelName.split('_')[(-2)]
            name1 = modelName.split('_')[(-1)]
            modelName0 = '%s_%s' % (name0, name1)
            modelName1 = '%s_%s' % (name1, name0)
        elif not isAv:
            modelName0 = modelName
            modelName1 = modelName
        else:
            notL.append(modelName)
            continue
        threshold = threshold0
        if delta >0 and '_' in modelName:
            dis = keyDis(modelName,stations)
            T = dis / Vav
            threshold = max(delta / T, threshold0)
            if threshold > threshold0:
                print(dis,delta / T, threshold)
        if modelName0 in fvDAv:
            (fvD[modelName].update)(fvDAv[modelName0], threshold=threshold, **kwags)
        else:
            if modelName1 in fvDAv:
                (fvD[modelName].update)(fvDAv[modelName1], threshold=threshold, **kwags)
            else:
                notL.append(modelName)
                continue
        if len(fvD[modelName].f)<2:
            notL.append(modelName)
    for name in notL:
        fvD.pop(name)
        print(name)
def compareFvD(fvD, fvDRef, f, resDir='results/', keyL=[], stations=[]):
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    if len(keyL) == 0:
        keyL = fvDRef
    for key in keyL:
        if '_' not in key:
            pass
        else:
            sta0, sta1 = key.split('_')
            key0 = key
            keyNew = sta1 + '_' + sta0
            if key in fvD or keyNew in fvD:
                if keyNew in fvD:
                    key = keyNew
                if isinstance(fvD, dict):
                    fv = fvD[key]
                if isinstance(fvDRef, dict):
                    fvRef = fvDRef[key0]
                if len(fvRef.f) > 2:
                    plt.figure(figsize=[3, 3.5])
                    v = fv(f)
                    std = fv.STD(f)
                    vRef = fvRef(f)
                    v[v < 1] = np.nan
                    vRef[vRef < 1] = np.nan
                    hRef, = plt.plot(vRef, f, 'k', linewidth=0.3, label='manual')
                    hGet, = plt.plot(v, f, 'r', linewidth=0.3, label='predict')
                    hD, = plt.plot((vRef * 0.99), f, '-.k', linewidth=0.3, label='$\\pm$ 1%')
                    plt.plot((vRef * 1.01), f, '-.k', linewidth=0.3)
                    plt.legend()
                    if len(stations) == 0:
                        plt.title(key)
                    else:
                        dis = keyDis(key, stations)
                        plt.title('%s dist: %.1f km' % (key, dis))
                    plt.gca().set_yscale('log')
                    plt.ylim([f.min(), f.max()])
                    plt.xlim([3, 5])
                    plt.xlabel('v(km/s)')
                    plt.xlabel('f(Hz)')
                    plt.tight_layout()
                    plt.savefig((resDir + 'compare_' + key + '.eps'), dpi=300)
                    plt.savefig((resDir + 'compare_' + key + '.jpg'), dpi=300)
                    plt.close()
class areas:
    __doc__ = 'docstring for  areas'

    def __init__(self, laL=[], loL=[], stations=[]):
        n = len(stations)
        M = 5
        if len(laL) == 0:
            laLo = np.zeros([n, 2])
            for i in range(n):
                laLo[(i, 0)] = stations[i]['la']
                laLo[(i, 1)] = stations[i]['lo']

            k = cluster.k_means(laLo, M)[0]
            self.la = k[:, 0]
            self.lo = k[:, 1]
        else:
            self.la = np.array(laL)
            self.lo = np.array(loL)
        N = len(self.la)
        self.fvM = [[[] for j in range(N)] for i in range(N)]
        self.avM = [[None for j in range(N)] for i in range(N)]
        self.stations = stations
        self.N = N

    def R2(self, la, lo):
        R2 = (self.la - la) ** 2 + (self.lo - lo) ** 2
        return R2

    def index(self, la, lo):
        return self.R2(la, lo).argmin()

    def Index(self, staStr):
        sta = self.stations.Find(staStr)
        return self.index(sta['la'], sta['lo'])

    def INDEX(self, key):
        staStr0, staStr1 = key.split('_')
        i0 = self.Index(staStr0)
        i1 = self.Index(staStr1)
        return (i0, i1)

    def insert(self, key, fv):
        i0, i1 = self.INDEX(key)
        self.fvM[i0][i1].append(fv)
        if i0 != i1:
            self.fvM[i1][i0].append(fv)

    def Insert(self, fvD):
        for key in fvD:
            self.insert(key, fvD[key])

    def getAv(self, threshold=2.5):
        for i in range(self.N):
            for j in range(self.N):
                if len(self.fvM[i][j]) > 30:
                    self.avM[i][j] = averageFVL((self.fvM[i][j]), threshold=threshold)

    def limit(self, fvD, threshold=2):
        keys = fvD.keys()
        for key in list(keys):
            i0, i1 = self.INDEX(key)
            if isinstance(self.avM[i0][i1], type(None)):
                fvD.pop(key)
            else:
                self.avM[i0][i1].limit((fvD[key]), threshold=threshold)

    def std20(self):
        for i0 in range(self.N):
            for i1 in range(self.N):
                if not isinstance(self.avM[i0][i1], type(None)):
                    self.avM[i0][i1].std[self.avM[i0][i1].std > 0.1] = 0
                    self.avM[i0][i1].genInterp()

    def plot(self, resDir='test/'):
        N = len(self.la)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        for i in range(N):
            for j in range(N):
                plotFVL((self.fvM[i][j]), (self.avM[i][j]), ('%s/fvM%.2f_%.2f+%.2f_%.2f.jpg' % (
                 resDir, self.la[i], self.lo[i], self.la[j], self.lo[j])),
                  thresholdL=[1, 2, 3, 4], title=('between (%.2f $^\\circ$N %.2f$^\\circ$E) and (%.2f $^\\circ$N %.2f$^\\circ$W)' % (
                 self.la[i], self.lo[i], self.la[j], self.lo[j])))


def saveFvD(fvD, fileDir='./'):
    for key in fvD:
        fvD[key].save((fileDir + '/' + key + '-pvt.dat'), mode='NEFile')


h5Str = h5py.special_dtype(vlen=str)


corrType=np.float16
class corr:
    sigCount = 1

    def __init__(self, xx=np.arange(0, dtype=(np.float32)), timeL=np.arange(0), dDis=0, fs=0, az=np.array([0, 0]), dura=0, M=np.array([0, 0, 0, 0, 0, 0, 0]), dis=np.array([0, 0]), dep=10, modelFile='', name0='', name1='', srcSac='', x0=np.arange(1), x1=np.arange(1), quakeName='', maxCount=-1, up=1,isSyn=False):
        self.maxCount = -1
        maxCount0 = xx.shape[0]
        if maxCount < maxCount0:
            if maxCount > 0:
                xx = xx[:maxCount]
                x0 = x0[:maxCount]
                x1 = x1[:maxCount]
                timeL = timeL[:maxCount]
        maxCount = xx.shape[0]
        self.xx = xx.astype(np.float32)
        self.timeL = timeL.astype(np.float32)
        self.dDis = dDis
        self.fs = fs
        self.az = az
        self.dura = dura
        self.M = M
        self.up = up
        if up > 1:
            time0 = self.timeL[0]
            delta = (self.timeL[1] - self.timeL[0]) / up
            self.timeLOut = np.arange(self.timeL.shape[0] * up) * delta + time0
        else:
            self.timeLOut = self.timeL
        self.dis = dis
        self.dep = dep
        self.modelFile = modelFile
        self.name0 = name0
        self.name1 = name1
        self.srcSac = srcSac
        self.x0 = x0.astype(np.float32)
        self.x1 = x1.astype(np.float32)
        self.quakeName = quakeName
        self.dtype = self.getDtype(maxCount)
        self.isSyn=isSyn
        self.af = []
        self.aF = []
    def copy(self):
        self1 = corr()
        self1.maxCount = self.maxCount
        self1.xx = self.xx
        self1.timeL = self.timeL
        self1.dDis = self.dDis
        self1.fs = self.fs
        self1.az = self.az
        self1.dura = self.dura
        self1.M = self.M
        self1.up = self.up
        self1.timeLOut = self.timeLOut
        self1.dis = self.dis
        self1.dep = self.dep
        self1.modelFile = self.modelFile
        self1.name0 = self.name0
        self1.name1 = self.name1
        self1.srcSac = self.srcSac
        self1.x0 = self.x0
        self1.x1 = self.x1
        self1.quakeName = self.quakeName
        self1.dtype = self.dtype
        self1.isSyn=self.isSyn
        return self1
    def getDtype(self, maxCount):
        self.maxCount = maxCount
        corrType = np.dtype([('xx', np.float32, maxCount),
         (
          'time0', np.float32, 1),
          (
          'delta', np.float32, 1),
         (
          'dDis', np.float64, 1),
         (
          'fs', np.float64, 1),
         (
          'az', np.float64, 2),
         (
          'dura', np.float64, 1),
         (
          'M', np.float64, 7),
         (
          'dis', np.float64, 2),
         (
          'dep', np.float64, 1),
         (
          'modelFile', np.str, 200),
         (
          'name0', np.str, 200),
         (
          'name1', np.str, 200),
         (
          'srcSac', np.str, 200),
         (
          'x0', np.float32, 1),
         (
          'x1', np.float32, 1),
         (
          'quakeName', np.str, 200)])
        return corrType
    def getDtypeH5(self, maxCount):
        self.maxCount = maxCount
        corrType = np.dtype([('xx', np.float32, maxCount),
         (
          'time0', np.float32, 1),
          (
          'delta', np.float32,1),
         (
          'dDis', np.float64, 1),
         (
          'fs', np.float64, 1),
         (
          'az', np.float64, 2),
         (
          'dura', np.float64, 1),
         (
          'M', np.float64, 7),
         (
          'dis', np.float64, 2),
         (
          'dep', np.float64, 1),
         (
          'modelFile', h5Str, 1),
         (
          'name0', h5Str,1),
         (
          'name1', h5Str, 1),
         (
          'srcSac', h5Str, 1),
         (
          'x0', np.float32, 1),
         (
          'x1', np.float32, 1),
         (
          'quakeName', h5Str, 1)])
        return corrType
    def outputTimeDis(self, FV, T=np.array([5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300]), sigma=2, byT=False, byA=False, rThreshold=1e-2, set2One=False, move2Int=False, noY=False, randMove=False,dtG=[],disRandA=1/15,disMaxR=1,**kwargs):
        self.T = T
        f = 1 / T
        t0 = self.timeLOut[0]
        delta = self.timeLOut[1] - self.timeLOut[0]
        dim = [
         self.timeLOut.shape[0], T.shape[0]]
        timeDis = np.zeros(dim)
        if noY:
            return (timeDis, t0)
        else:
            f = f.reshape([1, -1])
            timeL = self.timeLOut.reshape([-1, 1])
            v = FV(f, self.dis[0], self.dis[1])
            t = self.dDis / v
            if self.isSyn:
                t[t>1/f/disRandA]=1e9
                t[t<1/f/disMaxR]=1e9
            if len(dtG)!=0:
                t += dtG.reshape([1, -1])
                if np.random.rand() < 0.0005:
                    print('##random in corr', np.abs(dtG).max(),self.dDis)
            else:
                dtG = t*0
                #if np.random.rand() < 0.001:
                #    print('##random in corr', dt[dt<100])
            if move2Int:
                dt = np.abs(t - timeL)
                minT = dt.min(axis=0)
                indexT = dt.argmin(axis=0)
                t[(0, minT < delta)] = timeL[(indexT, 0)][(minT < delta)]
            tmpSigma = sigma
            if byT:
                minSigma = 0.5
                if corr.sigCount > 0:
                    corr.sigCount -= 1
                    print(minSigma)
                percent = sigma / 100
                tmpSigma = percent * t
                tmpSigma[tmpSigma < minSigma] = minSigma
                tmpSigma[tmpSigma > 4 * sigma] = 4 * sigma
            tmpSigma = tmpSigma.reshape([1, -1])
            timeDis = ne.evaluate('exp(-((timeL - t) / tmpSigma) ** 2)')
            halfV = np.exp(-(delta * 0.5 / tmpSigma) ** 2)
            if set2One:
                timeDis[timeDis > halfV] = 1
                #timeDis[timeDis > 1 / np.e] = 1
                #timeDis[timeDis <= 1 / np.e] = 0
            if byA or self.isSyn:
                if len(self.aF)==0:
                    self.aF = np.abs(self.getA(f[0])).astype(np.float32)/f[0]**0.5
                timeDis[:, self.aF < rThreshold] = 0
            return (
             timeDis, t0)
    def synthetic(self,FV,rThreshold=1e-2,isFv=False,F=[],disRandA=1/15,disMaxR=1,isNoise=False,**kwargs):
        #tiqian suanhao 
        fL = np.fft.fftfreq(len(self.xx),self.timeL[1]-self.timeL[0]).astype(np.float32)
        fL=fL[fL>1/200]
        fL=fL[fL<1/6]
        v = FV(fL)
        dt=(self.dDis / v).astype(np.float32)
        if len(self.af)==0:
            self.af =np.abs(self.getA(fL).reshape([1,-1])).astype(np.float32)
        timeL = self.timeL.reshape([-1,1])
        dt = dt.reshape([1,-1])
        fL = fL.reshape([1,-1])
        af = self.af
        pi = np.pi
        data=ne.evaluate('cos(-pi*2*(timeL-dt)*fL)*af').sum(axis=1)
        #data=ne.evaluate(np.cos(-np.pi*2*(timeL-dt)*fL)*af).sum(axis=1)
        if isNoise:
            TMIN = self.dDis / 5
            TMax = self.dDis / 2.8
            fMin=fL[af>rThreshold].min()
            minDT = 1/fMin*0.75
            A = (np.random.rand()-0.5)*0.1
            if np.random.rand()<0.01:
                print('Noise',A,'minDT',minDT)
            if np.random.rand()>0.5:
                dt = TMIN*(0.3+np.random.rand()*0.6)
                dt = max(minDT,dt)
                dIndex =int(dt/(self.timeL[1]-self.timeL[0]))
                data[:-dIndex]+=data[dIndex:]*A
            else:
                dt = TMax*(0.2+np.random.rand()*0.6)
                dt = max(minDT,dt)
                dIndex =int(dt/(self.timeL[1]-self.timeL[0]))
                data[dIndex:]+=data[:-dIndex]*A
        if isFv:
            if len(self.aF)==0:
                self.aF = np.abs(self.getA(F)).astype(np.float32)/F**0.5
            v = FV(F).astype(np.float32)
            DT=self.dDis / v
            v[self.aF < rThreshold]=0
            v[DT>1/F/disRandA]=0
            v[DT<1/F/disMaxR]=0
            return data,fv([F,v])
        return data

    def outputTimeDisNew(self, FV, sigma=2, byT=False, byA=False, rThreshold=1e-2, set2One=False, move2Int=False, noY=False, T=[]):
        t0 = self.timeLOut[0]
        delta = self.timeLOut[1] - self.timeLOut[0]
        halfV = np.exp(-(delta * 0.5 / sigma) ** 2)
        f = FV.f.reshape([1, -1])
        v = FV.v.reshape([1, -1])
        if len(T) > 0:
            f = 1 / T
            v = FV(f, self.dis[0], self.dis[1])
        f = f[(v > 0.3)]
        v = v[(v > 0.3)]
        f = f.reshape([1, -1])
        v = v.reshape([1, -1])
        if v.size < 2:
            return ([], [], [], [])
        else:
            dim = [
             self.timeLOut.shape[0], f.shape[(-1)]]
            t = self.dDis / v
            timeDis = np.zeros(dim)
            timeL = self.timeLOut.reshape([-1, 1])
            dim = [self.timeLOut.shape[0], f.shape[(-1)]]
            timeDis = np.zeros(dim)
            if noY:
                return (
                 timeDis, t0)
            t = self.dDis / v
            if move2Int:
                dt = np.abs(t - timeL)
                minT = dt.min(axis=0)
                indexT = dt.argmin(axis=0)
                t[(0, minT < delta)] = timeL[(indexT, 0)][(minT < delta)]
            tmpSigma = sigma
            if byT:
                tMax = max(300, t.max())
                tmpSigma = sigma / 300 * tMax
            timeDis = np.exp(-((timeL - t) / tmpSigma) ** 2)
            if set2One:
                if byT == False:
                    timeDis[timeDis > halfV] = 1
            if byA:
                spec = np.abs(np.fft.fft(self.xx))
                minf = 1 / (len(self.timeLOut) * (self.timeLOut[1] - self.timeOutL[0]))
                indexF = (f.reshape([-1]) / minf).astype(np.int)
                maxIndexF = indexF.max()
                spec /= spec[:maxIndexF + 1].mean()
                aF = spec[indexF]
                timeDis[:, aF < rThreshold] = timeDis[:, aF < rThreshold] * 0
            return (
             timeDis.transpose(), t0, (timeL * f).transpose(), f.transpose())
    def compareSpec(self, N=40):
        spec0 = self.toFew(np.abs(np.fft.fft(self.x0)), N)
        spec1 = self.toFew(np.abs(np.fft.fft(self.x1)), N)
        midN = max(1, int(0.01 * N))
        spec0Low = spec0[:midN].max()
        spec1Low = spec1[:midN].max()
        spec0High = spec0[midN:-midN].max()
        spec1High = spec1[midN:-midN].max()
        if spec0Low > spec0High * 1000 or spec1Low > spec1High * 1000:
            return 0
        else:
            return (spec0 * spec1).sum()

    def toFew(self, spec, N=20):
        spec[:1] *= 0
        spec[-1:] *= 0
        N0 = len(spec)
        d = int(N0 / (N - 1))
        i0 = np.arange(0, N0 - d, d)
        i1 = np.arange(d, N0, d)
        specNew = np.zeros(len(i0))
        for i in range(len(i0)):
            specNew[i] = spec[i0[i]:i1[i]].mean()
        specNew /= (specNew ** 2).sum() ** 0.5
        return specNew
    def compareInOut(self, yin, yout, t0, threshold=0.5):
        delta = self.timeLOut[1] - self.timeLOut[0]
        aIn = yin.max(axis=0)
        aOut = yout.max(axis=0)
        posIn = yin.argmax(axis=0) * delta + t0 + 1e-08
        posOut = yout.argmax(axis=0) * delta + t0 + 1e-08
        dPos = posOut - posIn
        dPosR = dPos / posIn
        dPos[aOut <= threshold] = dPos[(aOut <= threshold)] * 0 - 100000
        dPosR[aOut <= threshold] = dPosR[(aOut <= threshold)] * 0 - 10000
        return (dPos, dPosR, dPos * 0 + self.dDis)

    def getV(self, yout):
        posOut = yout.argmax(axis=0).reshape([-1])
        prob = yout.max(axis=0).reshape([-1])
        tOut = self.timeOutL[posOut]
        v = self.dis / tOut
        return (v, prob)

    def getStaName(self):
        netSta0 = os.path.basename(self.name0).split('.')[:2]
        netSta1 = os.path.basename(self.name1).split('.')[:2]
        return ('%s.%s' % (netSta0[0], netSta0[1]), '%s.%s' % (netSta1[0], netSta1[1]))

    def getA(self,f):
        t0 = self.dDis/6
        t1 = self.dDis/2
        i0 =int((t0-self.timeL[0])/(self.timeL[1]-self.timeL[0])) 
        i1 =int((t1-self.timeL[0])/(self.timeL[1]-self.timeL[0]))
        xx = self.xx[i0:i1]
        return calSpec(xx/xx.std()/len(xx),self.timeL[i0:i1],f)
    def show(self, d, FV):
        linewidth = 0.3
        F, t, zxx = d.stft(self.xx)
        t = t + self.timeL[0]
        ylim = [0, 0.2]
        xlim = [t[0], t[(-1)]]
        ax = plt.subplot(3, 2, 1)
        plt.plot((self.timeL), (np.real(self.xx)), 'b', linewidth=linewidth)
        plt.plot((self.timeL), (np.imag(self.xx)), 'r', linewidth=linewidth)
        plt.xlabel('t/s')
        plt.ylabel('corr')
        plt.xlim(xlim)
        ax = plt.subplot(3, 2, 3)
        plt.pcolor(t, F, np.abs(zxx))
        fTheor = FV.f
        timeTheorL = self.dDis / FV.v
        plt.plot(timeTheorL, fTheor, 'r')
        plt.xlabel('t/s')
        plt.ylabel('f/Hz')
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax = plt.subplot(3, 2, 2)
        sac0 = obspy.read(self.name0)[0]
        sac1 = obspy.read(self.name1)[0]
        plt.plot((getSacTimeL(sac0)), sac0, 'b', linewidth=linewidth)
        plt.plot((getSacTimeL(sac1)), sac1, 'r', linewidth=linewidth)
        plt.xlabel('time/s')
        ax = plt.subplot(3, 2, 4)
        mat = np.loadtxt(self.modelFile)
        ax.invert_yaxis()
        plt.plot((mat[:, 1]), (mat[:, 0]), 'b', linewidth=linewidth)
        plt.plot((mat[:, 2]), (mat[:, 0]), 'r', linewidth=linewidth)
        plt.ylim([900, -10])
        plt.subplot(3, 2, 5)
        timeDis = self.outputTimeDis(FV)
        plt.pcolor(self.timeLOut, self.T, timeDis.transpose())

    def output(self):
        return (
         self.xx, self.timeL, self.dDis, self.fs)

    def toDict(self):
        return {'xx':self.xx,  'time0':self.timeL[0],'delta':self.timeL[1]-self.timeL[0],  'dDis':self.dDis,  'fs':self.fs,  'az':self.az, 
         'dura':self.dura,  'M':self.M,  'dis':self.dis,  'dep':self.dep,  'modelFile':self.modelFile, 
         'name0':self.name0,  'name1':self.name1,  'srcSac':self.srcSac, 
         'x0':self.x0,  'x1':self.x1,  'quakeName':self.quakeName}
    def toMat(self):
        self.dtype = self.getDtype(self.xx.shape[0])
        return np.array((self.xx, self.timeL[0],self.timeL[1]-self.timeL[0], self.dDis, self.fs, self.az, self.dura,
         self.M, self.dis, self.dep, self.modelFile, self.name0, self.name1,
         self.srcSac, self.x0, self.x1, self.quakeName), self.dtype)
    def toMatH5(self):
        self.dtype = self.getDtypeH5(self.xx.shape[0])
        return np.array((self.xx, self.timeL[0],self.timeL[1]-self.timeL[0], self.dDis, self.fs, self.az, self.dura,
         self.M, self.dis, self.dep, self.modelFile, self.name0, self.name1,
         self.srcSac, self.x0, self.x1, self.quakeName), self.dtype)

    def setFromFile(self, file):
        mat = scipy.io.loadmat(file)
        self.setFromDict((mat['corr']), isFile=True)

    def setFromDict(self, mat, isFile=False):
        if not isFile:
            self.xx = mat['xx']
            self.timeL = mat['time0']+np.arange(len(mat['xx']))*mat['delta']
            self.dDis = mat['dDis']
            self.fs = mat['fs']
            self.az = mat['az']
            self.dura = mat['dura']
            self.M = mat['M']
            self.dis = mat['dis']
            self.dep = mat['dep']
            self.modelFile = str(mat['modelFile'])
            self.name0 = str(mat['name0'])
            self.name1 = str(mat['name1'])
            self.srcSac = str(mat['srcSac'])
            self.x0 = mat['x0']
            self.x1 = mat['x1']
            self.quakeName = str(mat['quakeName'])
        else:
            self.xx = mat['xx'][0][0][0]
            self.timeL = mat['time0'][0][0][0][0]+np.arange(len(self.xx))*mat['delta'][0][0][0][0]
            self.dDis = mat['dDis'][0][0][0]
            self.fs = mat['fs'][0][0][0][0]
            self.az = mat['az'][0][0][0]
            self.dura = mat['dura'][0][0][0][0]
            self.M = mat['M'][0][0][0]
            self.dis = mat['dis'][0][0][0]
            self.dep = mat['dep'][0][0][0][0]
            self.modelFile = str(mat['modelFile'][0][0][0])
            self.name0 = str(mat['name0'][0][0][0])
            self.name1 = str(mat['name1'][0][0][0])
            self.srcSac = str(mat['srcSac'][0][0][0])
            self.x0 = mat['x0'][0][0][0]
            self.x1 = mat['x1'][0][0][0]
            self.quakeName = str(mat['quakeName'][0][0][0])
        return self

    def save(self, fileName):
        sio.savemat(fileName, {'corr': self.toMat()})
    def getFV(self,f,fvRef,minDV,maxDV,maxK,minPer,maxPer,minSNR,N=50,v0=1.5,v1=5.,k=0,isControl=True,isByLoop=False,isStr=False,isRand=False, randA=0,midV=4,isNoMove=False,strD=0.02):
        data = self.xx.copy()
        Loop = np.arange(-5,N)
        d    = np.abs(self.dis[0]-self.dis[1])
        #print(d)
        #i0 = np.abs(d/v1-120-self.timeL).argmin()
        i1 = -256# np.abs(d/v0+512-self.timeL).argmin()
        i0 = 256
        #i1 = data.shape[0]
        data = data[i0:i1].copy()
        #data -= data.mean()
        #data=signal.detrend(data)
        timeL = self.timeL[i0:i1].copy()
        '''TMax = timeL[-1]-timeL[0]
        fMin = 1/TMax
        f = np.round(f[f>=fMin]/fMin)*fMin
        f =np.unique(f)'''
        vRef=fvRef(f)
        if not isStr:
            dvRef,KRef = fvRef.getDVK(f)
        tRef = d/vRef
        #print(len(data),f,tRef)
        Phi,std = calPhi(data,timeL,f,tRef,isNoMove=isNoMove)
        Phi = np.unwrap(Phi)
        t = (-Phi/np.pi/2/(f)).reshape([-1,1])+Loop.reshape([1,-1])/f.reshape([-1,1])
        if isByLoop:
            V = d/t
            dVR=V/vRef.reshape([-1,1])-1
            inCount = (np.abs(dVR)<0.2).sum(axis=0)
            loopIndex= inCount.argmax()
            v = V[:,loopIndex]
        else:
            dt = t- tRef.reshape([-1,1])
            v = f*0
            T=-10000
            for i in range(len(t)-1,-1,-1):
                if isStr:
                    index = np.abs(dt[i]**2).argmin()
                    T = t[i,index]
                    if np.abs(T/tRef[i]-1)>strD and np.abs(T-tRef[i])>1:
                        v[i]=0
                        #print(1/f[i],'wrong')
                    else:
                        v[i]=d/T
                    continue
                if tRef[i]*f[i]<1/5:
                    continue
                if T<0 or tRef[i]*f[i]<4:
                    index = np.abs(dt[i]**2).argmin()
                else:
                    index = np.abs(dt[i]**2+(t[i]-T)**2).argmin()
                T = t[i,index]
                v[i]=d/T
        #print(d/t[i],v[i])
        FV = fv([f.copy()[v>1],v[v>1],-std[v>1]],mode='num')
        if not isControl:
            return FV
        dv,K = FV.getDVK(f)
        #print(f,v,dv,K)
        per = v/vRef-1
        if False:#isRand:
            rand = np.random.rand()*(np.random.rand(*v.shape)*2-1)*randA
            if np.random.rand()>0.5:
                rand=v*rand/midV
                dv*=1+rand
            vRand*=1+rand
            per = vRand/vRef-1
            #dv*=1+rand
        F =[]
        V =[] 
        STD = []
        for i in range(len(t)):
            if dv[i]-dvRef[i]>minDV[i]-(maxDV[i]-minDV[i])*k/2*0 and dv[i]-dvRef[i]<maxDV[i]+(maxDV[i]-minDV[i])*k/2*0 and K[i]>=0 and  K[i]<maxK[i]*(1+k*0) and per[i]>minPer[i]-(maxPer[i]-minPer[i])*k*0 and per[i]<maxPer[i]+(maxPer[i]-minPer[i])*k*0 and std[i]>minSNR[i]:
                 #print('yes')
                 F.append(f[i])
                 V.append(v[i])
                 STD.append(std[i])
        return fv([np.array(F),np.array(V),-np.array(STD)],mode='num')
    def getFVSimple(self,f,fvRef,N=50,v0=1.5,v1=5.,k=0,isControl=True,isByLoop=False,isStr=False,isRand=False, randA=0,midV=4,isNoMove=False):
        data = self.xx.copy()
        Loop = np.arange(-5,N)
        d    = np.abs(self.dis[0]-self.dis[1])
        #print(d)
        #i0 = np.abs(d/v1-120-self.timeL).argmin()
        i1 = -1# np.abs(d/v0+512-self.timeL).argmin()
        i0 = 0
        #i1 = data.shape[0]
        data = data[i0:i1].copy()
        #data -= data.mean()
        #data=signal.detrend(data)
        timeL = self.timeL[i0:i1].copy()
        '''TMax = timeL[-1]-timeL[0]
        fMin = 1/TMax
        f = np.round(f[f>=fMin]/fMin)*fMin
        f =np.unique(f)'''
        vRef=fvRef(f)
        if not isStr:
            dvRef,KRef = fvRef.getDVK(f)
        tRef = d/vRef
        #print(len(data),f,tRef)
        Phi,std = calPhi(data,timeL,f,tRef,isNoMove=isNoMove)
        Phi = np.unwrap(Phi)
        t = (-Phi/np.pi/2/(f)).reshape([-1,1])+Loop.reshape([1,-1])/f.reshape([-1,1])
        if isByLoop:
            V = d/t
            dVR=V/vRef.reshape([-1,1])-1
            inCount = (np.abs(dVR)<0.2).sum(axis=0)
            loopIndex= inCount.argmax()
            v = V[:,loopIndex]
        else:
            dt = t- tRef.reshape([-1,1])
            v = f*0
            T=-10000
            for i in range(len(t)-1,-1,-1):
                if isStr:
                    index = np.abs(dt[i]**2).argmin()
                    T = t[i,index]
                    if np.abs(T/tRef[i]-1)>0.015 and np.abs(T-tRef[i])>1:
                        v[i]=0
                        #print(1/f[i],'wrong')
                    else:
                        v[i]=d/T
                    continue
                if tRef[i]*f[i]<1/5:
                    continue
                if T<0 or tRef[i]*f[i]<4:
                    index = np.abs(dt[i]**2).argmin()
                else:
                    index = np.abs(dt[i]**2+(t[i]-T)**2).argmin()
                T = t[i,index]
                v[i]=d/T
        #print(d/t[i],v[i])
        FV = fv([f.copy()[v>1],v[v>1],-std[v>1]],mode='num')
        return FV

def calPhi_(data,timeL,f,sigmaF=8,deltaF=0/100,sigmaT=4,deltaT=0):
    spec0 = np.fft.fft(data)
    delta= timeL[1]-timeL[0]
    fMax  = 1/(delta)
    N = len(timeL)
    fL = fMax*np.arange(N)/N
    fL[N-1:int((N-1)/2):-1]=fL[0:int((N)/2)]
    Phi = f*0
    for i in range(len(f)):
        F = f[i]
        spec = spec0*np.exp(-((F-fL)/(fL[1]*sigmaF+deltaF))**2)
        dataF = np.abs(np.fft.ifft(spec))
        iMax  = dataF.argmax()
        timeF = timeL[iMax]
        w     = np.exp(-((timeF-timeL)/(1/F*sigmaT+deltaT))**2)
        i0    = max(0,int(iMax-sigmaT/F/delta-deltaT))
        i1    = min(len(data),int(iMax+sigmaT/F/delta+deltaT))
        #w0    = signal.hilbert(w)
        #data = data*w
        Phi[i] =np.angle(calSpec(data[i0:i1],timeL[i0:i1],np.array([F])))[0]
        #print(timeF,Phi)
    return Phi

def calPhi(data,timeL,f,tRef,gamma=4,deltaF=0/100,gammaW=20,deltaT=0,isS=False,isNoMove=False):
    delta= timeL[1]-timeL[0]
    #print(timeL[0],timeL[-1])
    fMax  = 1/(delta)
    N = len(timeL)
    fMin = fMax/N
    Phi = f*0
    std = f*0
    emin=7
    if isS:
        S0=[]
        S1=[]
    #f= np.round(f/fMin)*fMin
    #f=np.unique(f)
    for i in range(len(f)):
        TRef = tRef[i]
        i0 = np.abs(timeL-TRef*0.75+0.25/f[i]).argmin()
        i1 = np.abs(timeL-TRef*1.25-0.25/f[i]).argmin()
        iN0 = np.abs(timeL-TRef*1.8).argmin()
        iN1 = np.abs(timeL-TRef*2.5).argmin()
        iN0 = int(0.75*len(timeL))
        iN1 = -1
        F = f[i]
        T = 1/f[i]
        if  T<25:
            gammaW=25
        elif  T<40:
            gammaW=24
        elif T<80:
            gammaW=22
        elif T<100:
            gammaW=20
        elif T<150:
            gammaW=18
        else:
            gammaW=18
        gammaW=20
        wn = 2*np.pi*F
        alpha = gamma**2 * wn
        t0 = np.sqrt( emin * np.log(10) * 4 * alpha) / wn
        it0 =  t0/delta;
        it0 = round(it0/2) * 2
        nfil = it0 * 2 - 1;

        tfil = delta *  ( np.arange(nfil) - it0+1 )
        yfil = wn / (2 * np.sqrt( np.pi * alpha )) * np.exp( - wn**2 / ( 4 * alpha ) * tfil**2)
        yfil = yfil * np.cos( wn * tfil )
        x1x2h = np.convolve(data,yfil)
        x1x2h = x1x2h[it0-1:len(data)+it0-1]
        x1x2h = signal.detrend(x1x2h)
        if isS:
            #print(len(x1x2h))
            S0.append(x1x2h.copy())
        #x1x2h -=x1x2h.mean()
        if i0>=i1:
            i1=i0+1
        indexMax= np.abs(x1x2h[i0:i1]).argmax()+i0
        #print(timeL[indexMax]/TRef)
        if isNoMove:
            indexMax= np.abs(timeL-TRef).argmin()
        aMax= np.abs(x1x2h[i0:i1]).max()
        noise=(x1x2h[iN0:iN1]).std()
        alpha = (gammaW)**2 * wn
        h = ((np.arange(len(data))-indexMax)*delta)**2 * wn**2 / (4*alpha);
        h = np.exp(-h);
        x1x2h = x1x2h * h
        Phi[i] =np.angle(calSpec(x1x2h,timeL,np.array([F])))[0]
        std[i] = aMax/noise
        if isS:
            S1.append(x1x2h.copy())
    if isS:
        return Phi,std,np.array(S0),np.array(S1)
    return Phi,std

def showS(timeL,data,f,S,filename):
    plt.figure(figsize=[4,4])
    S = S/np.abs(S).max(axis=1,keepdims=True)
    SH = signal.hilbert(S,axis=1)
    SH = SH/np.abs(SH).max(axis=1,keepdims=True)
    plt.subplot(3,1,1)
    #plt.gca().set_position([0.2,0.7,0.6,0.2])
    plt.plot(timeL,data,'k')
    plt.xlabel('t/s')
    #plt.xticks([])
    plt.xlim([timeL[0],timeL[-1]])
    plt.subplot(3,1,2)
    #plt.gca().set_position([0.2,0.4,0.6,0.2])
    pc=plt.pcolor(timeL,1/f,S,vmin=-1,vmax=1,cmap='bwr',rasterized=True)
    plt.xlim([timeL[0],timeL[-1]])
    #plt.xlabel('t/s')
    plt.ylabel('T/s')
    #plt.colorbar(label='$S(f)$',orientation="horizontal")
    #plt.xticks([])
    #figureSet.setColorbar(pc,'$S(f)$',pos='right')
    #plt.colorbar(label='$S(f)$')
    #plt.yscale('log')
    plt.gca().set_yscale('log')
    plt.subplot(3,1,3)
    #plt.gca().set_position([0.2,0.1,0.6,0.2])
    pc=plt.pcolor(timeL,1/f,np.abs(SH)**2,vmin=0,vmax=1,cmap='hot',rasterized=True)
    plt.xlabel('t/s')
    plt.ylabel('T/s')
    #plt.colorbar(label='$|A|^2$',orientation="horizontal")
    #figureSet.setColorbar(pc,'$|A|^2$',pos='right')
    plt.xlim([timeL[0],timeL[-1]])
    #plt.yscale('log')
    plt.gca().set_yscale('log')
    plt.savefig(filename,dpi=300)
    plt.close()


def calSpec(data,timeL,f):
    timeLT=timeL.reshape([1,-1])
    fLT = f.reshape([-1,1])
    dataT = data.reshape([1,-1])
    #print(dataT.shape,fLT.shape,timeLT.shape)
    pi = np.pi
    #MS  = (dataT*np.sin(-fLT*timeLT*np.pi*2)).sum(axis=1)
    #MC  = (dataT*np.cos(-fLT*timeLT*np.pi*2)).sum(axis=1)
    MS  = ne.evaluate('dataT*sin(-fLT*timeLT*pi*2)').sum(axis=1)
    MC  = ne.evaluate('dataT*cos(-fLT*timeLT*pi*2)').sum(axis=1)
    return 1j*MS+MC
def compareList(i0, i1):
    di = np.array(i0) - np.array(i1)
    return np.sum(np.abs(di)) < 0.1

class corrL(list):

    def __init__(self, *argv, **kwargs):
        super().__init__()
        if len(argv) > 0:
            for tmp in argv[0]:
                if 'fvD' in kwargs:
                    fvD = kwargs['fvD']
                    modelName = tmp.modelFile
                    if len(modelName.split('_')) >= 2:
                        name0 = modelName.split('_')[(-2)]
                        name1 = modelName.split('_')[(-1)]
                        modelName0 = '%s_%s' % (name0, name1)
                        modelName1 = '%s_%s' % (name1, name0)
                        if modelName not in fvD:
                            if modelName0 not in fvD:
                                if modelName1 not in fvD:
                                    continue
                    else:
                        continue
                if 'specThreshold' in kwargs:
                    if kwargs['specThreshold']>0.1:
                        if tmp.compareSpec(N=40) < kwargs['specThreshold']:
                            print('not match')
                            continue
                if 'maxCount' in kwargs:
                    pass
                self.append(tmp)
        self.iL = np.arange(0)
        self.timeDisArgv = ()
        self.timeDisKwarg = {}

    def reSetUp(self, up=1):
        for corr in self:
            corr.up = up
            if up > 1:
                time0 = corr.timeL[0]
                delta = (corr.timeL[1] - corr.timeL[0]) / up
                corr.timeLOut = np.arange(corr.timeL.shape[0] * up) * delta + time0
            else:
                corr.timeLOut = corr.timeL

    def checkIn(self, new, isCheckin=True):
        if not isCheckin:
            return True
        else:
            #new.x0 = new.x0.astype(np.float32)
            #new.x1 = new.x1.astype(np.float32)
            for x in [new.xx]:
                if np.isnan(x).sum() > 0 or np.isinf(x).sum() > 0:
                    print('bad record')
                    return False
                else:
                    if x.std() == 0:
                        return False
                    if x.max() == 0:
                        return False
                x /= x.max()
                if (x > 0.001).sum() < 5:
                    return False
            return self.checkNorm(new)

    def checkNorm(self, new, timeMax=5000, threshold=1, minA=0):
        xx = new.xx.copy()
        dDis = np.abs(new.dis[0] - new.dis[1])
        timeEnd = dDis / 2
        i0 = int((timeEnd - new.timeL[0]) / (new.timeL[1] - new.timeL[0]))
        xxAf = xx[i0:]
        std = xxAf[(np.abs(xxAf) > 1e-15)].std()
        A = xx.max()
        if A / std < minA:
            return False
        else:
            index = xx.argmax()
            time = new.timeL[index]
            v = dDis / time
            return True

    def append(self, new, isCheckIn=True):
        if self.checkIn(new, isCheckin=isCheckIn):
            if not isinstance(new, corr):
                new1 = corr()
                new1.setFromDict(new.toDict())
                new = new1
            super().append(new)
        else:
            print('data not right')

    def shuffle(self):
        count = len(self)
        ori = list(self)
        np.random.shuffle(ori)
        self = corrL(ori)

    def plotPickErro(self, yout, T, iL=[], fileName='erro.jpg', threshold=0.5):
        plt.close()
        N = yout.shape[0]
        if len(iL) == 0:
            iL = self.iL
        dPosL = np.zeros([N, len(T)])
        dPosRL = np.zeros([N, len(T)])
        fL = np.zeros([N, len(T)])
        dDisL = np.zeros([N, len(T)])
        for i in range(N):
            index = iL[i]
            tmpCorr = self[index]
            tmpYin = self.y[i, :, 0]
            tmpYOut = yout[i, :, 0]
            t0 = self.t0L[i]
            dPos, dPosR, dDis = tmpCorr.compareInOut(tmpYin, tmpYOut, t0, threshold=threshold)
            f = 1 / T
            dPosL[i, :] = dPos
            dPosRL[i, :] = dPosR
            fL[i, :] = f
            dDisL[i, :] = dDis
        bins = np.arange(-50, 50, 2) / 4
        res = np.zeros([len(T), len(bins) - 1])
        for i in range(len(T)):
            res[i, :], tmp = np.histogram((dPosL[(dPosL[:, i] > -1000, i)]), bins, density=True)
        plt.pcolor((bins[:-1]), (1 / T), res, cmap='viridis')
        plt.xlabel('erro/s')
        plt.ylabel('f/Hz')
        plt.colorbar()
        plt.gca().semilogy()
        plt.gca().invert_yaxis()
        plt.title(fileName[:-4] + '_%.2f' % threshold)
        plt.savefig((fileName[:-4] + '_%.2f.jpg' % threshold), dpi=300)
        plt.close()
        bins = np.arange(-100, 100, 1) / 800
        res = np.zeros([len(T), len(bins) - 1])
        for i in range(len(T)):
            res[i, :], tmp = np.histogram((dPosRL[(dPosRL[:, i] > -1000, i)]), bins, density=True)
        plt.pcolor((bins[:-1]), (1 / T), res, cmap='viridis')
        plt.xlabel('erro Ratio /(s/s)')
        plt.ylabel('f/Hz')
        plt.colorbar()
        plt.gca().semilogy()
        plt.gca().invert_yaxis()
        plt.title(fileName[:-4] + '_%.2f_R' % threshold)
        plt.savefig((fileName[:-4] + '_%.2f_R.jpg' % threshold), dpi=300)
        plt.close()
    def plotPickErroSq(self, yout, iL=[], fileName='erro.jpg', threshold=0.5):
        plt.close()
        N = yout.shape[0]
        dPos = yout.argmax(axis=1) - self.y.argmax(axis=1)
        bins = np.arange(-50, 50, 2) / 4
        res = np.zeros([len(T), len(bins) - 1])
        plt.histogram2d(dPos, (1 / self.fL), [bins, unique((1 / self.fL).tolist()).sort()], density=True)
        plt.xlabel('erro/s')
        plt.ylabel('f/Hz')
        plt.colorbar()
        plt.gca().semilogy()
        plt.gca().invert_yaxis()
        plt.title(fileName[:-4] + '_%.2f' % threshold)
        plt.savefig((fileName[:-4] + '_%.2f.jpg' % threshold), dpi=300)
        plt.close()
        plt.histogram2d((dPos / self.y.argmax(axis=1)), (1 / self.fL), [bins, unique((1 / self.fL).tolist()).sort()], density=True)
        plt.xlabel('erro/s')
        plt.ylabel('f/Hz')
        plt.colorbar()
        plt.gca().semilogy()
        plt.gca().invert_yaxis()
        plt.title(fileName[:-4] + '_%.2f' % threshold)
        plt.savefig((fileName[:-4] + '_%.2f_Rela.jpg' % threshold), dpi=300)
        plt.close()

    def loadByDirL(self, heads):
        for head in heads:
            for matFile in glob(head + '*.mat'):
                tmp = corr()
                tmp.setFromFile(matFile)
                self.append(tmp)
    def loadByPairsH5(self,pairNames,h5):
        if isinstance(h5,h5py.File):
            for pairName in pairNames:
                if pairName in h5['data']:
                    pair = h5['data'][pairName][: h5['count'][pairName][0]]
                    for Corr in pair:
                        tmp = corr()
                        tmp.setFromDict(np.array(Corr))
                        self.append(tmp)
        else:
            with h5py.File(h5,'r') as h5:
                self.loadByPairsH5(pairNames,h5)
    def loadByNamesH5(self, names,h5):
        if isinstance(h5,h5py.File):
            for name in names:
                if '_' not in name:
                    pass
                else:
                    sta0, sta1 = name.split('_')[-2:]
                    if sta0 > sta1:
                        sta1, sta0 = sta0, sta1
                    pairName = sta1+'_'+sta1
                    if pairName in h5['data']:
                        pair = h5['data'][pairName]
                        modelNames= h5['modelName'][pairName][:]
                        if name in modelNames:
                            i = modelNames.index(name)
                            tmp = corr()
                            tmp.setFromDict(np.array(pair[i]))
                            self.append(tmp)
        else:
            with h5py.File(h5,'r') as h5:
                self.loadByNamesH5(names,h5)
    def loadByH5(self, h5):
        if isinstance(h5,h5py.File):
            for pairName in h5['data']:
                pair = h5['data'][pairName][:h5['count'][pairName][0]]
                for corrNp in pair:
                    tmp = corr()
                    tmp.setFromDict(corrNp)
                    self.append(tmp)
        else:
            with h5py.File(h5,'r') as h5:
                self.loadByH5(h5)
    def loadByNames(self, matDir, names):
        for name in names:
            if '_' not in name:
                pass
            else:
                sta0, sta1 = name.split('_')[-2:]
                if sta0 > sta1:
                    sta1, sta0 = sta0, sta1
                tmpDir = matDir + sta0 + '/' + sta1 + '/'
                fileName = tmpDir + name + '.mat'
                if os.path.exists(fileName):
                    print(fileName)
                    tmp = corr()
                    tmp.setFromFile(fileName)
                    self.append(tmp)

    def save(self, head):
        if 'h5'==head[-2:]:
            return self.saveH5(head)
        fileDir = os.path.dirname(head)
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)
        for i in range(len(self)):
            tmp = self[i]
            print(i)
            modelName = tmp.modelFile
            if '_' not in modelName:
                pass
            else:
                sta0, sta1 = modelName.split('_')[-2:]
                if sta0 > sta1:
                    sta1, sta0 = sta0, sta1
                tmpDir = fileDir + '/' + sta0 + '/' + sta1 + '/'
                if not os.path.exists(tmpDir):
                    os.makedirs(tmpDir)
                fileName = tmpDir + modelName + '.mat'
                mat = tmp.toMat()
                sio.savemat(fileName, {'corr': mat})
    def saveH5(self, f,maxEvents=500):
        if isinstance(f,h5py.File):
            for i in range(len(self)):
                if 'data' not in f:
                    f.create_group('data')
                if 'count' not in f:
                    f.create_group('count')
                if 'modelName' not in f:
                    f.create_group('modelName')
                tmp = self[i]
                if i%1000==0:
                    print(i)
                modelName = tmp.modelFile
                if '_' not in modelName:
                    pass
                else:
                    sta0, sta1 = modelName.split('_')[-2:]
                    if sta0 > sta1:
                        sta1, sta0 = sta0, sta1
                    pairName =  sta0 + '_' + sta1
                    mat = tmp.toMatH5()
                    if pairName not in f['data']:
                        f['data'].create_dataset(pairName,dtype=mat.dtype, shape=(maxEvents,),maxshape=(None,))
                    if pairName not in f['count']:
                        f['count'].create_dataset(pairName,dtype=np.int,data=0,shape=(1,))
                    if pairName not in f['modelName']:
                        f['modelName'].create_dataset(pairName,dtype=h5Str,shape=(maxEvents,),maxshape=(None,))
                    count=int(f['count'][pairName][0])
                    if count==len(f['data'][pairName]):
                        f['data'][pairName].reshape([count+20])
                        f['modelName'][pairName].reshape([count+20])
                    f['data'][pairName][count]=mat
                    f['modelName'][pairName][count]=mat['modelFile']
                    f['count'][pairName][0]=count+1
                    if False:
                        tmpGroup =  sta0 + '_' + sta1
                        if tmpGroup not in f:
                            f.create_group(tmpGroup)
                        fileName = modelName
                        mat = tmp.toMatH5()
                        if fileName in f[tmpGroup]:
                            f[tmpGroup].pop(fileName)
                        f[tmpGroup][fileName]=mat
        else:
            with h5py.File(f,'a') as f:
                self.saveH5(f)
    def setTimeDis(self, *argv, **kwargs):
        self.timeDisArgv = argv
        self.timeDisKwarg = kwargs
        self.iL = np.arange(0)

    def __call__(self, iL):
        (self.getTimeDis)(iL, *(self.timeDisArgv), **self.timeDisKwarg)
        return (self.x, self.y, self.t0L)

    def newCall(self, iL):
        (self.getTimeDisNew)(iL, *(self.timeDisArgv), **self.timeDisKwarg)
        return (self.x, self.y, self.n, self.t0L)

    def __str__(self):
        return '%d %s' % (len(self), str(self.timeDisKwarg))

    def getTimeDis(self, iL, fvD={}, T=[], sigma=2, maxCount=512, noiseMul=0, byT=False, byA=False, rThreshold=1e-2, byAverage=False, set2One=False, move2Int=False, modelNameO='', noY=False, randMove=False, randA=0.03, midV=4, randR=0.5, mul=1,isGuassianMove=False,disAmp=1,fromT=0,fvDSyn={},**kwags):
        if len(iL) == 0:
            iL = np.arange(len(self))
        if not isinstance(iL, np.ndarray):
            iL = np.array(iL).astype(np.int)
        if iL.size == self.iL.size:
            if compareList(iL, self.iL):
                print('already done')
                return
        self.iL = iL
        dtypeX = np.float32
        dtypeY = np.float32
        maxCount0 = maxCount
        fLFV = 1/np.array(T)
        up = self[0].up
        x = np.zeros([len(iL), maxCount, 1, 2], dtype=dtypeX)
        y = np.zeros([len(iL), maxCount * up, 1, len(T)], dtype=dtypeY)
        t0L = np.zeros(len(iL))
        t0FL= np.zeros([len(iL),len(T)])
        dDisL = np.zeros(len(iL))
        deltaL = np.zeros(len(iL))
        randIndexL = np.zeros(len(iL))
        keyLSyn = list(fvDSyn.keys())
        for ii in range(len(iL)):
            if ii % mul == 0:
                randN = np.random.rand(1)
                Rand0 = np.random.rand(1)
                Rand1 = np.random.rand(1)
                Rand2 = np.random.rand(1)
                Rand3 = np.random.rand(1)
            i = iL[ii]
            delta0 = self[i].timeL[1] - self[i].timeL[0]
            fL0 = np.fft.fftfreq(maxCount0)/delta0
            maxCount = min(maxCount0, self[i].xx.shape[0])
            if self[i].isSyn==False or len(fvDSyn)==0:
                if modelNameO == '':
                    modelName = self[i].modelFile
                    if byAverage:
                        if len(modelName.split('_')) >= 2:
                            name0 = modelName.split('_')[(-2)]
                            name1 = modelName.split('_')[(-1)]
                            modelName0 = '%s_%s' % (name0, name1)
                            modelName1 = '%s_%s' % (name1, name0)
                            if modelName0 in fvD:
                                modelName = modelName0
                            if modelName1 in fvD:
                                modelName = modelName1
                else:
                    modelName = modelNameO
                if modelName in fvD:
                    fv=fvD[modelName]
                else:
                    fv = fvNone 
            else:
                fv = fvDSyn[random.choice(keyLSyn)]
            if isGuassianMove and randMove and randN > randR and (self[i].isSyn==False): 
                AT=(Rand0 - 0.5) * 2 * (self[i].dDis / midV) * randA * Rand1,
                f0=1/((30**Rand2)*8)
                sigmaG= 40**(Rand3**2)*0.2
                dtG = AT*np.exp(-((np.abs(fL0/f0)-1)**2/sigmaG**2))
                dtGFV=AT*np.exp(-((np.abs(fLFV/f0)-1)**2/sigmaG**2))
                if np.random.rand() < 0.001:
                    print('##random ', 'AT',AT, 'f0',f0, 'sigma',sigmaG,'dDis',self[i].dDis)
            else:
                dtGFV=fLFV*0
            
            tmpy, t0,= self[i].outputTimeDis((fv), T=T,
                sigma=sigma,
                byT=byT,
                byA=byA,
                rThreshold=rThreshold,
                set2One=set2One,
                move2Int=move2Int,
                noY=noY,
                randMove=randMove,
                dtG=dtGFV,**kwags)
            iP, iN = self.ipin(t0-fromT, self[i].fs)
            y[ii, iP * up:maxCount * up + iN * up, 0, :] = tmpy[-iN * up:maxCount * up - iP * up]
            if not self[i].isSyn:
                xx=self[i].xx
            else:
                xx=self[i].synthetic(fv,**kwags)
            x[ii, iP:maxCount + iN, 0, 0] = xx.reshape([
                -1])[-iN:maxCount - iP]
            if isGuassianMove and randMove and randN > randR and (self[i].isSyn==False):
                specX = np.fft.fft(x[ii,:, 0, 0])
                spexXNew = specX*np.exp(-fL0*np.pi*2*dtG*1j)
                x[ii,:, 0, 0] =np.fft.ifft(spexXNew)
            dDis = self[i].dDis
            if randMove and (self[i].isSyn==False):
                if randN <= randR:
                    rand1 = 1 + randA * (2 * Rand0 - 1) * Rand1
                    if np.random.rand(1) < 0.001:
                        print('mul', mul)
                        print('******* rand1:', rand1)
                    dDis = dDis * rand1
            timeMin = dDis / 5.0-fromT
            timeMax = dDis / 3.0-fromT
            I0 = int(np.round(timeMin / delta0).astype(np.int))
            I1 = int(np.round(timeMax / delta0).astype(np.int))
            x[ii, I0:I1, 0, 1] = disAmp
            t0L[ii] = t0 - iN / self[i].fs - iP / self[i].fs
            dt = np.random.rand() * 5 - 2.5
            iP, iN = self.ipin(t0 + dt, self[i].fs)
            if randMove and (self[i].isSyn==False):
                if randN > randR and (not isGuassianMove):
                    dT = (Rand0 - 0.5) * 2 * (self[i].dDis / midV) * randA * Rand1
                    dN = int(np.round(dT * self[i].fs).astype(np.int))
                    t0L[ii]-=float(dN)/self[i].fs
                    if np.random.rand() < 0.001:
                        print('##random ', dT, dN, self[i].dDis)
                    if dN > 0:
                        for channel in (0, ):
                            x[ii, dN:, 0, channel] = x[ii, :-dN, 0, channel]
                            x[ii, :dN, 0, channel] = 0

                        y[ii, dN * up:, 0, :] = y[ii, :-dN * up, 0, :]
                        y[ii, :dN * up, 0, :] = 0
                    if dN < 0:
                        for channel in (0, ):
                            x[ii, :dN, 0, channel] = x[ii, -dN:, 0, channel]
                            x[ii, dN:, 0, channel] = 0

                        y[ii, :dN * up, 0, :] = y[ii, -dN * up:, 0, :]
                        y[ii, dN * up:, 0, :] = 0
            dDisL[ii] = self[i].dDis
            t0FL[ii] = t0L[ii]-dtGFV
            deltaL[ii] = self[i].timeLOut[1] - self[i].timeLOut[0]
        xStd = x.std(axis=1, keepdims=True)
        self.x = x
        if noiseMul > 0:
            self.x += noiseMul * ((np.random.rand)(*list(x.shape)).astype(dtype) - 0.5) * xStd
        self.y = y
        self.randIndexL = randIndexL
        self.t0L = t0L
        self.t0FL = t0FL
        self.dDisL = dDisL
        self.deltaL = deltaL
    def clear(self):
        self.x=0
        self.y=0
        self.t0L=0
        self.iL=np.arange(0)
        gc.collect()
    def getTimeDisNew(self, iL, fvD={}, T=[], sigma=2, maxCount=512, noiseMul=0, byT=False, byA=False, rThreshold=1e-2, byAverage=False, set2One=False, move2Int=False, modelNameO='', noY=False, randMove=False, up=1):
        if len(iL) == 0:
            iL = np.arange(len(self))
        if not isinstance(iL, np.ndarray):
            iL = np.array(iL).astype(np.int)
        if iL.size == self.iL.size:
            if compareList(iL, self.iL):
                print('already done')
                return
        self.iL = iL
        dtype = np.float32
        maxCount0 = maxCount
        x = []
        y = []
        n = []
        t0L = []
        dDisL = []
        deltaL = []
        randIndexL = []
        indexL = []
        fL = []
        for ii in range(len(iL)):
            i = iL[ii]
            maxCount = min(maxCount0, self[i].xx.shape[0])
            if modelNameO == '':
                modelName = self[i].modelFile
                if byAverage:
                    if len(modelName.split('_')) >= 2:
                        name0 = modelName.split('_')[(-2)]
                        name1 = modelName.split('_')[(-1)]
                        modelName0 = '%s_%s' % (name0, name1)
                        modelName1 = '%s_%s' % (name1, name0)
                        if modelName0 in fvD:
                            modelName = modelName0
                        if modelName1 in fvD:
                            modelName = modelName1
            else:
                modelName = modelNameO
            if len(fvD[modelName].f) < 2:
                pass
            else:
                tmpy, t0, tmpn, tmpf = self[i].outputTimeDisNew((fvD[modelName]), T=T,
                  sigma=sigma,
                  byT=byT,
                  byA=byA,
                  rThreshold=rThreshold,
                  set2One=set2One,
                  move2Int=move2Int,
                  noY=noY)
                if len(tmpy) == 0:
                    pass
                else:
                    up = self[i].up
                    X = np.zeros([tmpy.shape[0], maxCount0, 4], dtype=dtype)
                    Y = np.zeros([tmpy.shape[0], maxCount0 * up, 1], dtype=dtype)
                    N = np.zeros([tmpy.shape[0], maxCount0 * up, 1], dtype=dtype)
                    iP, iN = self.ipin(t0, self[i].fs)
                    Y[:, iP * up:maxCount * up + iN * up, 0] = tmpy[:, -iN * up:maxCount * up - iP * up]
                    N[:, :, 0] = tmpn[:, -iN * up:-iN * up + 1] + (np.arange(maxCount0 * up) - iP * up).reshape([1, -1]) * (tmpn[:, 1:2] - tmpn[:, 0:1])
                    X[:, iP:maxCount + iN, 0] = self[i].xx.reshape([
                     -1])[-iN:maxCount - iP]
                    X[:, iP:maxCount + iN, 1] = 0
                    dt = np.random.rand() * 5 - 2.5
                    iP, iN = self.ipin(t0 + dt, self[i].fs)
                    dDisL += [self[i].dDis] * len(tmpf)
                    deltaL += [self[i].timeLOut[1] - self[i].timeLOut[0]] * len(tmpf)
                    x.append(X)
                    y.append(Y)
                    n.append(N)
                    t0L += [t0] * len(tmpf)
                    randIndexL = []
                    indexL = [i] * len(tmpf)
                    fL += tmpf.reshape([-1]).tolist()
        self.x = np.concatenate(x, axis=0)
        self.n = np.concatenate(n, axis=0)
        self.y = np.concatenate(y, axis=0)
        self.randIndexL = randIndexL
        self.t0L = np.array(t0L)
        self.dDisL = np.array(dDisL)
        self.deltaL = np.array(deltaL)
        self.indexL = np.array(indexL)
        self.fL = np.array(fL)
    def getV(self, yout, isSimple=True, D=0.1, isLimit=False, isFit=False,maxN=10):
        if isSimple:
            maxDis = self.dDisL.max()
            minDis = self.dDisL.min()
            tmin = minDis / 6 - 5
            tmax = maxDis / 2 + 10
            i0 = 0# int(max(1, tmin / self.deltaL[0]))
            i1 = yout.shape[1]-1#int(min(yout.shape[1] - 1, tmax / self.deltaL[0]))
            #pos,prob = yout[:, i0:i1, 0, :].argmax(axis=1) + i0
            #prob = pos.astype(np.float64) * 0
            pos,prob = mathFunc.Max(yout[:, i0:i1, 0, :],N=maxN) 
            pos+= i0
            #prob = pos.astype(np.float64) * 0
            vM = []
            probM = []
            for i in range(pos.shape[0]):
                vM.append([])
                probM.append([])
                for j in range(pos.shape[1]):
                    POS = np.where(yout[i, i0:i1, 0, j] > 0.5)[0] + i0
                    time = self.t0FL[i,j] + POS * self.deltaL[i]
                    vM[(-1)].append(self.dDisL[i] / time)
                    probM[(-1)].append(yout[(i, POS, 0, j)])
            '''
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    prob[(i, j)] = yout[(i, pos[(i, j)], 0, j)]
            '''
            #time = self.t0L.reshape([-1, 1]) + pos * self.deltaL.reshape([-1, 1])
            time = self.t0FL + pos * self.deltaL.reshape([-1, 1])
            v = self.dDisL.reshape([-1, 1]) / time
        else:
            if isLimit:
                yout *= self.y.max(axis=1, keepdims=True) > 0.5
            N = yout.shape[0]
            M = yout.shape[(-1)]
            v = np.zeros([N, M])
            prob = np.zeros([N, M])
            fvD = self.timeDisArgv[0]
            T = self.timeDisArgv[1]
            isO = False
            if 'modelNameO' in self.timeDisKwarg:
                if self.timeDisKwarg['modelNameO'] != '':
                    modelFile = self.timeDisKwarg['modelNameO']
                    fv = fvD[modelFile]
                    vL = fv(1 / T)
                    minVL = (1 - D) * vL
                    maxVL = (1 + D) * vL
                    isO = True
            for i in range(N):
                index = self.iL[i]
                if not isO:
                    modelFile = self[index].modelFile
                    fv = fvD[modelFile]
                    vL = fv(1 / T)
                    minVL = (1 - D) * vL
                    maxVL = (1 + D) * vL
                for j in range(M):
                    v0 = vL[j]
                    minV = minVL[j]
                    maxV = maxVL[j]
                    maxT = self.dDisL[i] / minV
                    minT = self.dDisL[i] / maxV
                    i0 = max(0, int((minT - self.t0L[0]) / self.deltaL[i]))
                    i1 = min(yout.shape[1] - 1, int((maxT - self.t0L[0]) / self.deltaL[i]))
                    if i0 >= i1:
                        continue
                    pos = yout[i, i0:i1, 0, j].argmax() + i0
                    prob[(i, j)] = yout[(i, pos, 0, j)]
                    if prob[(i, j)] > 0.5:
                        if isFit:
                            pass
                    time = self.t0L[i] + pos * self.deltaL[i]
                    v[(i, j)] = self.dDisL[i] / time

        return (
         v, prob, vM, probM)

    def saveV(self, v, prob, T, iL=[], stations=[], minProb=0.7, resDir='models/predict/'):
        if len(iL) == 0:
            iL = self.iL
        for i in range(v.shape[0]):
            index = iL[i]
            corr = self[index]
            sta0, sta1 = corr.getStaName()
            station0 = stations.Find(sta0)
            station1 = stations.Find(sta1)
            timeStr, laStr, loStr = corr.quakeName.split('_')
            time = float(timeStr)
            la = float(laStr)
            lo = float(loStr)
            fileDir = '%s/%s/%s_%s/Rayleigh/' % (resDir, sta1, sta1, sta0)
            if not os.path.exists(fileDir):
                os.makedirs(fileDir)
            file = fileDir + 'pvt_sel.dat'
            vIndex = np.where(prob[i] > minProb)[0]
            if len(vIndex) == 0:
                pass
            else:
                with open(file, 'a') as (f):
                    f.write('%s %s\n' % (station1['sta'], station0['sta']))
                    f.write('%s 5\n' % station1['comp'][(-1)])
                    f.write('%s %.5f\n' % (obspy.UTCDateTime(time).strftime('%Y %m %d %H %M'), time % 60))
                    f.write('%f %f\n' % (station1['la'], station1['lo']))
                    f.write('%.5f %.5f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[1] / 111.19, corr.az[1]))
                    f.write('%s 5\n' % station0['comp'][(-1)])
                    f.write('%s %.5f\n' % (obspy.UTCDateTime(time).strftime('%Y %m %d %H %M'), time % 60))
                    f.write('%f %f\n' % (station0['la'], station0['lo']))
                    f.write('%.5f %.5f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[0] / 111.19, corr.az[0]))
                    f.write('2 %d\n' % len(vIndex))
                    for ii in vIndex:
                        f.write('%f\n' % (1 / T[ii]))

                    for ii in vIndex:
                        f.write('%f %f\n' % (v[i][ii], -prob[i][ii]))

    def saveVAll(self, v, prob, T, iL=[], stations=[], minProb=0.7, resDir='models/predict/'):
        if len(iL) == 0:
            iL = self.iL
        for i in range(len(v)):
            index = iL[i]
            corr = self[index]
            sta0, sta1 = corr.getStaName()
            station0 = stations.Find(sta0)
            station1 = stations.Find(sta1)
            timeStr, laStr, loStr = corr.quakeName.split('_')
            time = float(timeStr)
            la = float(laStr)
            lo = float(loStr)
            fileDir = '%s/%s/%s_%s/Rayleigh/' % (resDir, sta1, sta1, sta0)
            if not os.path.exists(fileDir):
                os.makedirs(fileDir)
            file = fileDir + 'pvt_all.dat'
            vIndex = []
            for j in range(len(prob[i])):
                if (prob[i][j] > minProb).sum() > 0:
                    vIndex.append(j)

            if len(vIndex) == 0:
                pass
            else:
                with open(file, 'a') as (f):
                    f.write('%s %s\n' % (station1['sta'], station0['sta']))
                    f.write('%s 5\n' % station1['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station1['la'], station1['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[1] /111.19, corr.az[1]))
                    f.write('%s 5\n' % station0['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station0['la'], station0['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[0] / 111.19, corr.az[0]))
                    f.write('2 %d\n' % len(vIndex))
                    for ii in vIndex:
                        f.write('%f\n' % (1 / T[ii]))

                    for ii in vIndex:
                        for j in range(len(prob[i][ii])):
                            if prob[i][ii][j] > minProb:
                                f.write('%f ' % v[i][ii][j])
                                f.write('%f ' % prob[i][ii][j])

                        f.write('\n')

    def saveVAllSq(self, v, prob, T, iL=[], stations=[], minProb=0.7, resDir='models/predict/'):
        if len(iL) == 0:
            iL = self.iL
        for i in range(v.shape[0]):
            index = iL[i]
            corr = self[index]
            sta0, sta1 = corr.getStaName()
            station0 = stations.Find(sta0)
            station1 = stations.Find(sta1)
            timeStr, laStr, loStr = corr.quakeName.split('_')
            time = float(timeStr)
            la = float(laStr)
            lo = float(loStr)
            fileDir = '%s/%s/%s_%s/Rayleigh/' % (resDir, sta0, sta0, sta1)
            if not os.path.exists(fileDir):
                os.makedirs(fileDir)
            file = fileDir + 'pvt_all.dat'
            vIndex = []
            for j in range(len(prob[i])):
                if (prob[i][j] > minProb).sum() > 0:
                    vIndex.append(j)

            if len(vIndex) == 0:
                pass
            else:
                with open(file, 'a') as (f):
                    f.write('%s %s\n' % (station0['sta'], station1['sta']))
                    f.write('%s 5\n' % station0['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station0['la'], station0['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[0], corr.az[0]))
                    f.write('%s 5\n' % station1['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station1['la'], station1['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[1], corr.az[1]))
                    f.write('2 %d\n' % len(vIndex))
                    for ii in vIndex:
                        f.write('%f\n' % (1 / T[ii]))

                    for ii in vIndex:
                        for j in range(prob[i][ii]):
                            if prob[i][ii][j] > minProb:
                                f.write('%f ' % v[i][ii][j])
                                f.write('%f ' % prob[i][ii][j])
                            f.write('\n')

    def saveVByPair(self, v, prob, T, iL=[], stations=[], minProb=0.7, resDir='models/predict/'):
        if len(iL) == 0:
            iL = self.iL
        sta0, sta1 = self[iL[0]].getStaName()
        station0 = stations.Find(sta0)
        station1 = stations.Find(sta1)
        fileDir = '%s/%s/%s_%s/Rayleigh/' % (resDir, sta0, sta0, sta1)
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)
        file = fileDir + 'pvt_sel.dat'
        with open(file, 'a') as (f):
            for i in range(v.shape[0]):
                index = iL[i]
                corr = self[index]
                timeStr, laStr, loStr = corr.quakeName.split('_')
                time = float(timeStr)
                la = float(laStr)
                lo = float(loStr)
                vIndex = np.where(prob[i] > minProb)[0]
                if len(vIndex) == 0:
                    pass
                else:
                    f.write('%s %s\n' % (station0['sta'], station1['sta']))
                    f.write('%s 5\n' % station0['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station0['la'], station0['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[0], corr.az[0]))
                    f.write('%s 5\n' % station1['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station1['la'], station1['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[1], corr.az[1]))
                    f.write('2 %d\n' % len(vIndex))
                    for ii in vIndex:
                        f.write('%f\n' % (1 / T[ii]))

                    for ii in vIndex:
                        f.write('%f %f\n' % (v[i][ii], -prob[i][ii]))

    def getAndSaveOld(self, model, fileName, stations, isPlot=False, isSimple=True, D=0.2, isLimit=False, isFit=False, minProb=0.7):
        N = len(self)
        if 'T' in self.timeDisKwarg:
            T = self.timeDisKwarg['T']
        else:
            T = self.timeDisArgv[1]
        resDir = os.path.dirname(fileName)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        M = len(T)
        v = np.zeros([N, M])
        v0 = np.zeros([N, M])
        prob = np.zeros([N, M])
        prob0 = np.zeros([N, M])
        for i0 in range(0, N, 1000):
            i1 = min(i0 + 1000, N)
            print(i0, i1)
            x, y, t = self(np.arange(i0, i1))
            print('predict')
            Y = model.predict(x)
            print('calV')
            v[i0:i1], prob[i0:i1], vM, probM = self.getV(Y, isSimple=isSimple, D=D,
              isLimit=isLimit,
              isFit=isFit)

        self.saveV(v, prob, T, (np.arange(N)), stations, resDir=resDir, minProb=minProb)
        if isPlot:
            dv = np.abs(v - v0)
            dvO = v - v0
            plt.close()
            for i in range(len(dv)):
                indexL = validL((dv[i]), (prob[i]), minProb=minProb, minV=(-1), maxV=2)
                if np.random.rand() < 0.1:
                    print('validL: ', indexL)
                for iL in indexL:
                    iL = np.array(iL).astype(np.int)
                    plt.plot((v[(i, iL)]), (1 / T[iL]), 'k', linewidth=0.1, alpha=0.3)
            plt.xlim([2, 7])
            plt.gca().semilogy()
            plt.xlabel('v/(m/s)')
            plt.ylabel('f/Hz')
            plt.savefig((fileName + '.jpg'), dpi=300)
            plt.close()
            for i in range(len(dv)):
                indexL = validL((dv[i]), (prob[i]), minProb=minProb, minV=(-1), maxV=2)
                if np.random.rand() < 0.1:
                    print('validL: ', indexL)
                for iL in indexL:
                    iL = np.array(iL).astype(np.int)
                    plt.plot((dvO[(i, iL)]), (1 / T[iL]), 'k', linewidth=0.1, alpha=0.3)
            plt.xlim([-1, 1])
            plt.xlabel('dv/(m/s)')
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.savefig((fileName + '_dv.jpg'), dpi=300)
            plt.close()
    def getAndSaveOldSq(self, model, fileName, stations, isPlot=False, isSimple=True, D=0.2, isLimit=False, isFit=False, minProb=0.7):
        N = len(self)
        if 'T' in self.timeDisKwarg:
            T = self.timeDisKwarg['T']
        else:
            T = self.timeDisArgv[1]
        resDir = os.path.dirname(fileName)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        M = len(T)
        v = np.zeros([N, M])
        v0 = np.zeros([N, M])
        prob = np.zeros([N, M])
        prob0 = np.zeros([N, M])
        for i0 in range(0, N, 1000):
            i1 = min(i0 + 1000, N)
            print(i0, i1)
            x, y, t = self(np.arange(i0, i1))
            print('predict')
            Y = model.predict(x).reshape([-1, len(T), x.shape[1], 0]).transpose([0, 2, 3, 1])
            print('calV')
            v[i0:i1], prob[i0:i1], vM, probM = self.getVSq(Y, isSimple=isSimple, D=D,
              isLimit=isLimit,
              isFit=isFit)
            self.saveVAll(vM, probM, T, (self.indexL), stations, resDir=resDir, minProb=minProb)

        self.saveV(v, prob, T, (np.arange(N)), stations, resDir=resDir, minProb=minProb)
        if isPlot:
            dv = np.abs(v - v0)
            dvO = v - v0
            plt.close()
            for i in range(dv.shape[0]):
                indexL = validL((dv[i]), (prob[i]), minProb=minProb, minV=(-1), maxV=2)
                if np.random.rand() < 0.1:
                    print('validL: ', indexL)
                for iL in indexL:
                    iL = np.array(iL).astype(np.int)
                    plt.plot((v[(i, iL)]), (1 / T[iL]), 'k', linewidth=0.1, alpha=0.3)

            plt.xlim([2, 7])
            plt.gca().semilogy()
            plt.xlabel('v/(m/s)')
            plt.ylabel('f/Hz')
            plt.savefig((fileName + '.jpg'), dpi=300)
            plt.close()
            for i in range(dv.shape[0]):
                indexL = validL((dv[i]), (prob[i]), minProb=minProb, minV=(-1), maxV=2)
                if np.random.rand() < 0.1:
                    print('validL: ', indexL)
                for iL in indexL:
                    iL = np.array(iL).astype(np.int)
                    plt.plot((dvO[(i, iL)]), (1 / T[iL]), 'k', linewidth=0.1, alpha=0.3)

            plt.xlim([-1, 1])
            plt.xlabel('dv/(m/s)')
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.savefig((fileName + '_dv.jpg'), dpi=300)
            plt.close()

    def getAndSave(self, model, fileName, stations, isPlot=False, isSimple=True, D=0.2, isLimit=False, isFit=False, minProb=0.7):
        N = len(self)
        if 'T' in self.timeDisKwarg:
            T = self.timeDisKwarg['T']
        else:
            T = self.timeDisArgv[1]
        resDir = os.path.dirname(fileName)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        M = len(T)
        v = np.zeros([N, M])
        v0 = np.zeros([N, M])
        prob = np.zeros([N, M])
        prob0 = np.zeros([N, M])
        sN = len(stations)
        indexM = [[[] for i in range(sN)] for i in range(sN)]
        staL = [station['net'] + '.' + station['sta'] for station in stations]
        for i in range(N):
            sta0, sta1 = self[i].getStaName()
            index0 = staL.index(sta0)
            index1 = staL.index(sta1)
            indexM[index0][index1].append(i)

        for i in range(sN):
            for j in range(sN):
                if len(indexM[i][j]) > 0:
                    indexL = np.array(indexM[i][j]).astype(np.int)
                    print(staL[i], staL[j], len(indexL))
                    x, y, t = self(indexL)
                    V, Prob, vM, probM = self.getV((model.predict(x)), isSimple=isSimple, D=D,
                      isLimit=isLimit,
                      isFit=isFit)
                    V0, Prob0 = self.getV(y)
                    self.saveVByPair(V, Prob, T, indexL, stations, resDir=resDir, minProb=minProb)
                    for ii in range(len(indexL)):
                        iii = indexL[ii]
                        v[iii] = V[ii]
                        prob[iii] = Prob[ii]
                        v0[iii] = V0[ii]
                        prob0[iii] = Prob0[ii]

        if isPlot:
            dv = np.abs(v - v0)
            dvO = v - v0
            plt.close()
            for i in range(dv.shape[0]):
                indexL = validL((dv[i]), (prob[i]), minProb=minProb, minV=(-1), maxV=2)
                if np.random.rand() < 0.1:
                    print('validL: ', indexL)
                for iL in indexL:
                    iL = np.array(iL).astype(np.int)
                    plt.plot((v[(i, iL)]), (1 / T[iL]), 'k', linewidth=0.1, alpha=0.3)

            plt.xlim([2, 7])
            plt.gca().semilogy()
            plt.xlabel('v/(km/s)')
            plt.ylabel('f/Hz')
            plt.savefig((fileName + '.jpg'), dpi=300)
            plt.close()
            for i in range(dv.shape[0]):
                indexL = validL((dv[i]), (prob[i]), minProb=minProb, minV=(-1), maxV=2)
                if np.random.rand() < 0.1:
                    print('validL: ', indexL)
                for iL in indexL:
                    iL = np.array(iL).astype(np.int)
                    plt.plot((dvO[(i, iL)]), (1 / T[iL]), 'k', linewidth=0.1, alpha=0.3)

            plt.xlim([-1, 1])
            plt.xlabel('dv/(km/s)')
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.savefig((fileName + '_dv.jpg'), dpi=300)
            plt.close()

    def ipin(self, dt, fs):
        i0 = round(dt * fs)
        iP = 0
        iN = 0
        if i0 > 0:
            iP = i0
        else:
            iN = i0
        return (
         iP, iN)

    def copy(self):
        return corrL(self)


class corrD(dict):

    def __init__(self, corrL):
        self.corrL = corrL
        for i in range(len(corrL)):
            corr = corrL[i]
            modelName = corr.modelFile
            if '_' in modelName:
                sta0, sta1 = modelName.split('_')[-2:]
                if sta0 < sta1:
                    key = sta0 + '_' + sta1
                else:
                    key = sta0 + '_' + sta1
                if key not in self:
                    self[key] = []
                self[key].append(i)

        self.keyL = list(self.keys())

    def __call__(self, keyL=[], mul=1, N=-1,isRand=True,isSyn=False):
        N0 = N
        iL = []
        if len(keyL) == 0:
            keyL = list(self.keys())
        for key in keyL:
            if not isinstance(key, str):
                key = self.keyL[key]
            if isSyn:
                for index in self[key]:
                    if self.corrL[index].isSyn:
                        iL.append(index)
                        break
            if False:#len(self[key]) < mul:
                pass
            else:
                if N0 < 0:
                    N = int(len(self[key]) / mul + 0.99999999999) * mul
                if (N > len(self[key])) or (isRand==False):
                    for i in range(N):
                        iL.append(self[key][(i % len(self[key]))])
                else:
                    iL += random.sample(self[key], N)
        if len(iL)>0 and isRand:
            iL = np.array(iL)
            iLNew = iL.reshape([-1,mul])
            np.random.shuffle(iLNew)
            iL = iLNew.reshape([-1])
        #print(len(iL))
        x, y, t = self.corrL(np.array(iL))
        return (x.reshape([-1, mul, x.shape[1], x.shape[(-1)]]).transpose([0, 2, 1, 3]), y.reshape([-1, mul, y.shape[1], y.shape[(-1)]]).transpose([0, 2, 1, 3]), t.reshape([-1, mul]))
    def getAndSaveOld(self, model, fileName, stations, isPlot=False, isSimple=True, D=0.2, isLimit=False, isFit=False, minProb=0.7, mul=1):
        if 'T' in self.corrL.timeDisKwarg:
            T = self.corrL.timeDisKwarg['T']
        else:
            T = self.corrL.timeDisArgv[1]
        resDir = os.path.dirname(fileName)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        x, y, t = self(mul=mul)
        Y = model.predict(x)
        v, prob, vM, probM = self.corrL.getV((Y.transpose([0, 2, 1, 3]).reshape([-1, Y.shape[1], 1, Y.shape[(-1)]])), isSimple=isSimple, D=D, isLimit=isLimit, isFit=isFit)
        self.corrL.saveV(v, prob, T, (self.corrL.iL), stations, resDir=resDir, minProb=minProb)
        self.corrL.clear()
    def getAndSaveOldPer(self, model, fileName, stations, isPlot=False, isSimple=True, D=0.2, isLimit=False, isFit=False, minProb=0.7, mul=1,per=100):
        if 'T' in self.corrL.timeDisKwarg:
            T = self.corrL.timeDisKwarg['T']
        else:
            T = self.corrL.timeDisArgv[1]
        resDir = os.path.dirname(fileName)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        keyL = self.keyL
        for i in range(0,len(keyL),per):
            print(i,'in',len(keyL) )
            x, y, t = self(keyL=keyL[i:min(i+per,len(keyL))],mul=mul)
            Y = model.predict(x)
            v, prob, vM, probM = self.corrL.getV((Y.transpose([0, 2, 1, 3]).reshape([-1, Y.shape[1], 1, Y.shape[(-1)]])), isSimple=isSimple, D=D, isLimit=isLimit, isFit=isFit)
            self.corrL.saveV(v, prob, T, (self.corrL.iL), stations, resDir=resDir, minProb=minProb)
            self.corrL.clear()

def showCorrD_(x,y0,y,t,iL,corrL,outputDir,T,mul=6,number=3):
    f = 1/T
    dirName = os.path.dirname(outputDir)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    count = x.shape[1]
    cmap = plt.cm.bwr
    norm = colors.Normalize(vmin=0, vmax=1)
    for i in range(10):
        plt.close()
        axL=[]
        caxL=[]
        figL=[]
        for nn in range(number):
            fig=plt.figure(figsize=[7.5,5])
            grids=plt.GridSpec(10,4)
            #axL.append(fig.add_subplot(projection='3d'))
            axL.append(fig.add_subplot(grids[0:9,:],projection='3d'))
            caxL.append(fig.add_subplot(grids[9,1:3]))
            figL.append(fig)
        for j in range(0,mul,3):
            print(j)
            index= iL[i,j]
            if j==0:
                sta0,sta1 = corrL[index].modelFile.split('_')[-2:]
                head = sta0+'_'+sta1
            timeL    = corrL[index].timeL-corrL[index].timeL[0]+t[i,j]
            timeLOut = corrL[index].timeLOut-corrL[index].timeLOut[0]+t[i,j]
            xlim=[0,500]
            ylim=[-0.5,mul-0.5]
            zlim=[0,8]
            box = [3,6,1.25]
            elev = 40
            azim = -60
            tmpy0=y0[i,:,j,:]
            tmpy=y[i,:,j,:]
            tmpx = x[i,:,j,:]
            pos0  =tmpy0.argmax(axis=0)
            timeLOutL0=timeLOut[pos0.astype(np.int)]
            timeLOutL0[tmpy0.max(axis=0)<0.5]=np.nan
            pos  =tmpy.argmax(axis=0).astype(np.float)
            timeLOutL=timeLOut[pos.astype(np.int)]
            timeLOutL[tmpy.max(axis=0)<0.5]=np.nan
            #plt.subplot(number,1,1)
            #axL[0].title('%s%d'%(outputDir,i))
            legend = ['r s','i s',\
            'r h','i h']
            for k in range(2):
                if k==1:
                    c= 'k'
                else:
                    c= 'rrrgggbbbrrryyymmm'[j]
                axL[0].plot(timeL[timeL<xlim[1]],tmpx[timeL<xlim[1],k],c,\
                    label=legend[k],linewidth=0.5,zs=j,zdir='y')
            #plt.legend()
            axL[0].set_xlim(xlim)
            axL[0].set_ylim(ylim)
            axL[0].set_zlim(zlim)
            axL[0].set_ylabel('index')
            axL[0].set_xlabel('t/s')
            axL[0].set_zlabel('A')
            axL[0].set_yticks(np.arange(mul))
            axL[0].set_zticks([0,4])
            axL[0].set_box_aspect(box)
            axL[0].view_init(elev,azim)
            caxL[0].axis('off')
            figureSet.setColorbar(None,label='Probability',pos='bottom',isAppend=False)
            #plt.clim(0,1)
            #pc=plt.pcolormesh(timeLOut,f,tmpy0.transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True,zs=j,zdir='y')
            X,Z=np.meshgrid(timeLOut[timeLOut<xlim[1]],f)
            Y = X*0+j
            surf=axL[1].plot_surface(X,Y,Z,cmap='bwr',vmin=0,vmax=1,facecolors=cmap(norm(tmpy0[timeLOut<xlim[1]].transpose())),rasterized=True,rstride=1, cstride=1)
            surf._facecolors2d=surf._facecolors3d
            surf._edgecolors2d=surf._edgecolors3d
            axL[1].set_box_aspect(box)
            axL[1].view_init(elev,azim)
            #figureSet.setColorbar(pc,label='Probility',pos='right')
            if number==3:
                axL[1].plot(timeLOutL,f,'--k',linewidth=0.5,zs=j,zdir='y')
            axL[1].set_ylabel('index')
            axL[1].set_zlabel('f/Hz')
            axL[1].set_xlabel('t/s')
            axL[1].set_zscale('log')
            axL[1].set_xlim(xlim)
            axL[1].set_ylim(ylim)
            axL[1].set_zticks([f.max(),f.min()])
            #axL[1].set(clip_on=False)
            axL[1].set_yticks(np.arange(mul))
            pc =cm.ScalarMappable(norm=norm, cmap=cmap)
            #caxL[1].axis('off')
            figureSet.setColorbar(pc,label='Probability',pos='bottom',isAppend=False,ax=caxL[1])
            #plt.colorbar(label='Probility')
            if number==3:
                #pc=plt.pcolormesh(timeLOut,f,tmpy.transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True,zs=j,zdir='y')
                surf=axL[2].plot_surface(X,Y,Z,cmap='bwr',vmin=0,vmax=1,facecolors=cmap(norm(tmpy[timeLOut<xlim[1]].transpose())),rasterized=True,rstride=1, cstride=1)
                surf._facecolors2d=surf._facecolors3d
                surf._edgecolors2d=surf._edgecolors3d
                #plt.clim(0,1)
                axL[2].plot(timeLOutL0,f,'--k',linewidth=0.5,zs=j,zdir='y')
                axL[2].set_ylabel('index')
                axL[2].set_zlabel('f/Hz')
                axL[2].set_xlabel('t/s')
                axL[2].set_zscale('log')
                axL[2].set_box_aspect(box)
                axL[2].view_init(elev,azim)
                axL[2].set_yticks(np.arange(mul))
                axL[2].set_zticks([f.max(),f.min()])
                axL[2].set_xlim(xlim)
                axL[2].set_ylim(ylim)
                pc =cm.ScalarMappable(norm=norm, cmap=cmap)
                #caxL[2].axis('off')
                figureSet.setColorbar(pc,label='Probability',pos='bottom',isAppend=False,ax=caxL[2])
        #plt.colorbar(label='Probility')
        #plt.gca().semilogx()
        for nn in range(number):
            if number==3:
                tail = 'with'
                kind = ['wave','label','predict'][nn]
            if number==2:
                tail = 'without'
                kind = ['wave','label'][nn]
            figL[nn].savefig('%s/%s_%s_%s.svg'%(outputDir,head,kind,tail),dpi=500)

def showCorrD(x,y0,y,t,iL,corrL,outputDir,T,mul=6,number=3):
    f = 1/T
    dirName = os.path.dirname(outputDir)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    count = x.shape[1]
    cmap = plt.cm.bwr
    norm = colors.Normalize(vmin=0, vmax=1)
    for i in range(len(x)):
        plt.close()
        axL=[]
        caxL=[]
        figL=[]
        for nn in range(number):
            fig=plt.figure(figsize=[4,2])
            grids=plt.GridSpec(8,4)
            #axL.append(fig.add_subplot(projection='3d'))
            axL.append(fig.add_subplot(grids[0:4,:]))
            caxL.append(fig.add_subplot(grids[6,:]))
            figL.append(fig)
        for j in range(1):
            print(j)
            index= iL[i,j]
            if j==0:
                sta0,sta1 = corrL[index].modelFile.split('_')[-2:]
                head = sta0+'_'+sta1
            timeL    = corrL[index].timeL.copy()#-corrL[index].timeL[0]#+t[i,j]
            timeLOut = corrL[index].timeLOut.copy()#-corrL[index].timeLOut[0]#+t[i,j]
            xlim=[timeL.min(),timeL.max()]
            ylim=[-6,6]
            tmpy0=y0[i,:,j,:]
            tmpy=y[i,:,j,:]
            tmpx = x[i,:,j,:]
            pos0,A0  = mathFunc.Max(tmpy0.reshape([1,tmpy0.shape[0],-1]))
            pos0 = pos0.reshape([-1])
            A0   = A0.reshape([-1])
            timeLOutL0=timeLOut[0]+pos0*(timeLOut[1]-timeLOut[0])
            timeLOutL0[A0<0.5]=np.nan
            pos,A  = mathFunc.Max(tmpy.reshape([1,tmpy.shape[0],-1]))
            pos = pos.reshape([-1])
            A   = A.reshape([-1])
            timeLOutL=timeLOut[0]+pos*(timeLOut[1]-timeLOut[0])
            timeLOutL[A<0.5]=np.nan
            #plt.subplot(number,1,1)
            #axL[0].title('%s%d'%(outputDir,i))
            legend = ['r s','i s',\
            'r h','i h']
            for k in range(2):
                if k==1:
                    c= 'k'
                else:
                    c= 'rrrgggbbbrrryyymmm'[j]
                axL[0].plot(timeL[timeL<xlim[1]],tmpx[timeL<xlim[1],k],c,\
                    label=legend[k],linewidth=0.5)
            #plt.legend()
            axL[0].set_xlim(xlim)
            #axL[0].set_ylim(ylim)
            axL[0].set_xlabel('$t$/s')
            axL[0].set_ylabel('$A$')
            #axL[0].set_yticks(np.arange(mul))
            #axL[0].set_yticks([-2,0,2])
            caxL[0].axis('off')
            figureSet.setColorbar(None,label='Probability',pos='bottom',isAppend=False)
            #plt.clim(0,1)
            #pc=plt.pcolormesh(timeLOut,f,tmpy0.transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True,zs=j,zdir='y')
            surf=axL[1].pcolormesh(timeLOut[timeLOut<xlim[1]],f,tmpy0[timeLOut<xlim[1]].transpose(),vmin=0,vmax=1,cmap='bwr',rasterized=True)
            #figureSet.setColorbar(pc,label='Probility',pos='right')
            ##if number==3:
            #    axL[1].plot(timeLOutL,f,'--k',linewidth=0.5)
            axL[1].set_ylabel('$f$/Hz')
            axL[1].set_xlabel('$t$/s')
            axL[1].set_yscale('log')
            axL[1].set_xlim(xlim)
            #axL[1].set_ylim(ylim)
            pc =cm.ScalarMappable(norm=norm, cmap=cmap)
            #caxL[1].axis('off')
            figureSet.setColorbar(pc,label='Probability',pos='bottom',isAppend=False,ax=caxL[1])
            #plt.colorbar(label='Probility')
            if number==3:
                #pc=plt.pcolormesh(timeLOut,f,tmpy.transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True,zs=j,zdir='y')
                surf=axL[2].pcolormesh(timeLOut[timeLOut<xlim[1]],f,tmpy[timeLOut<xlim[1]].transpose(),vmin=0,vmax=1,cmap='bwr',rasterized=True)
                axL[2].plot(timeLOutL0,f,'--k',linewidth=0.5)
                axL[2].set_ylabel('$f$/Hz')
                axL[2].set_xlabel('$t$/s')
                axL[2].set_yscale('log')
                #axL[2].set_zticks([f.max(),f.min()])
                axL[2].set_xlim(xlim)
                #axL[2].set_ylim(ylim)
                pc =cm.ScalarMappable(norm=norm, cmap=cmap)
                #caxL[2].axis('off')
                figureSet.setColorbar(pc,label='Probability',pos='bottom',isAppend=False,ax=caxL[2])
        #plt.colorbar(label='Probility')
        #plt.gca().semilogx()
        for nn in range(number):
            if number==3:
                tail = 'with'
                kind = ['wave','label','predict'][nn]
            if number==2:
                tail = 'without'
                kind = ['wave','label'][nn]
            figL[nn].savefig('%s/%s_%s_%s.eps'%(outputDir,head,kind,tail),dpi=500)

def analyModel(depth, v, resDir='predict/KEA20/'):
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    V = v.reshape([-1])
    Dep = (v + depth.reshape([-1, 1, 1])).reshape(-1)
    Dep = Dep[(V > 0.1)]
    V = V[(V > 0.1)]
    Vmax = V.max()
    Vmin = V.min()
    vBins = np.arange(Vmin * 0.9, Vmax * 1.1, 0.1)
    depBins = [-1] + (depth[1:] / 2 + depth[:-1] / 2).tolist() + [depth.max() * 1.1]
    vmax = depth * 0
    vmean = depth * 0
    vmin = depth * 0
    vstd = depth * 0
    with open(resDir + 'vRange', 'w+') as (f):
        for i in range(len(depth)):
            vtmp = v[i]
            vtmp = vtmp[(vtmp > 0.1)]
            vmax[i], vmin[i], vmean[i], vstd[i] = (vtmp.max(), vtmp.min(), vtmp.mean(), vtmp.std())
            f.write('%.2f %.3f %.3f %.3f %.3f\n' % (depth[i], vmax[i], vmin[i], vmean[i], vstd[i]))

    plt.figure(figsize=[4, 4])
    plt.hist2d(V, Dep, bins=(vBins, depBins), rasterized=True, cmap='Greys', norm=(colors.LogNorm()))
    plt.ylim([depth.max() + 50, depth.min() - 10])
    plt.plot(vmean, depth, 'r', linewidth=0.5)
    plt.plot((vmean - vstd), depth, '--r', linewidth=0.5)
    plt.plot((vmean + vstd), depth, '--r', linewidth=0.5)
    plt.plot(vmin, depth, '.r', markersize=1)
    plt.plot(vmax, depth, '.r', markersize=1)
    plt.xlabel('v/(km/s)')
    plt.ylabel('depth(km)')
    plt.savefig((resDir + 'depth-v.jpg'), dpi=300)
    plt.close()


def getSacTimeL(sac):
    return np.arange(len(sac)) * sac.stats['delta'] + sac.stats['sac']['b']

def model2disp(vp,vs,z,f,rho=[],**kwags):
    if len(rho)==0:
        rho = vp * .32 + .77
    thickness = z[1:]-z[:-1]
    vp =0.5*(vp[1:]+vp[:-1])
    vs =0.5*(vs[1:]+vs[:-1])
    rho =0.5*(rho[1:]+rho[:-1])
    vp = vp[thickness!=0]
    vs = vs[thickness!=0]
    rho = rho[thickness!=0]
    thickness =thickness[thickness!=0]
    #thickness, vp, vs, rho, periods,
    #                wave='love', mode=1, velocity='group', flat_earth=True)
    #print(thickness, vp, vs, rho, 1/f,kwags)
    return surf96(thickness, vp, vs, rho, 1/f,**kwags)
 
z0=np.array([0,2.5,7.5,10,15,20,25,30,40,50,60,70,80,100,120,140,160,200,280,360,500])
vs0=np.array([3.352,3.352,3.374,3.419,3.489,3.569,3.632,3.651,4.125,4.563,4.483,4.416,4.373,4.362,4.374,4.389,4.400,4.427,4.530,4.719,5.058])
vp0=vs0*1.7
data = np.loadtxt('/home/jiangyr/Surface-Wave-Dispersion/models/prem')
z,vp,vs,rho,qp,qs=data.transpose()
z0 = np.array([0,10,20,30,40,50,60,80,100,125,150,175,200,250,300,360,420,500])
vp0 = interpolate.interp1d(z,vp)(z0)
vs0 = interpolate.interp1d(z,vs)(z0)
#vs0=np.array([3.352,3.352,3.474,3.819,3.889,38769,3.832,3.951,4.125,4.323,4.333,4.356,4.363,4.365,4.374,4.389,4.400,4.427,4.530,4.719,5.058])
#vs0=vs0*1.05
#vs0[-12:]*=1.0
#
class One:
    def __init__(self,vp=vp0,vs=vs0,z=z0,rho=[]):
        if len(rho)==0:
            rho = vp * .32 + .77
        thickness = z[1:]-z[:-1]
        vp =0.5*(vp[1:]+vp[:-1])
        vs =0.5*(vs[1:]+vs[:-1])
        rho =0.5*(rho[1:]+rho[:-1])
        self.vp = vp[thickness!=0]
        self.vs = vs[thickness!=0]
        self.rho = rho[thickness!=0]
        self.thickness =thickness[thickness!=0]
    def getDisp(self,f,wave='rayleigh', mode=1, velocity='phase', flat_earth=True,isRand=False):
        randA =  self.vp*0+1
        if isRand:
            I0 = np.random.rand()*len(self.thickness)
            sigmaI = np.random.rand()*10
            A = (np.random.rand()*2-1)*0.1
            randI = np.arange(len(self.thickness))
            randA = A*np.exp(-(randI-I0)**2/sigmaI**2)+1
        fMax = f.max()
        fMin = f.min()
        fL = fMin*(fMax/fMin)**np.arange(-0.0000000001,1.000001,1/59)
        vL=surf96(self.thickness, self.vp*randA, self.vs*randA,self.rho*randA, 1/fL[::-1],mode=mode, velocity=velocity, flat_earth=flat_earth,wave=wave)[::-1]
        if (vL>1).sum()<2:
            return f*0
        v = interpolate.interp1d(fL[vL>1],vL[vL>1],fill_value=0,bounds_error=False)(f)
        return v
    def synthetic(self,d,f0,f,sigmaF,AL,timeL,minA=0.05,**kwags):
        data = timeL.astype(np.float64)*0
        for i in range(len(AL)):
            v =  self.getDisp(f,mode=i+1,**kwags)
            A = AL[i]
            fA = np.exp(-(f-f0)**2/sigmaF**2)*A
            if i ==0:
                fValid  =  f[fA>minA]
                vValid =   v[fA>minA]
            print(v)
            for j in range(len(f)):
                if v[j]<0.5:continue
                V = v[j]
                t = d/V
                w = f[j]*np.pi*2
                
                data += fA[j]*np.cos(-(timeL-t)*w)
        return data,fValid,vValid



def model2kernel(vp0,vs0,z,f,rho0=[],**kwags):
    v0 = model2disp(vp0,vs0,z,f,rho0,**kwags)
    t = z[1:]-z[:-1]
    dv = 0.5
    GP = np.zeros([len(z),len(f)])
    GS = np.zeros([len(z),len(f)])
    GRho = np.zeros([len(z),len(f)])
    for i in range(len(z)):
        DV = dv*np.exp(-(z-z[i])**2/30**2)*dv
        DS = (t*(DV[1:]+DV[:-1])/2).sum()
        vp = vp0+DV
        vs = vs0+DV
        rho = rho0+DV
        v_p = model2disp(vp,vs0,z,f,rho0,**kwags)
        v_s = model2disp(vp0,vs,z,f,rho0,**kwags)
        v_rho = model2disp(vp0,vs0,z,f,rho,**kwags)
        GP[i]  = (v_p-v0)/DS
        GS[i]  = (v_s-v0)/DS
        GRho[i]  = (v_rho-v0)/DS
    GP[:,v0==0]=np.nan
    GS[:,v0==0]=np.nan
    GRho[:,v0==0]=np.nan
    v0[v0==0]=np.nan
    return v0,GP,GS,GRho

def genSharpW(omega,vp,vg,DOmega,x,phi,timeL):
    k = omega/vp
    k1 = 1/vg
    return np.cos(k*x-omega*timeL+phi)*np.sin((k1*x-timeL+0.00001)*DOmega)/(k1*x-timeL+0.00001)/DOmega,np.cos(k*x-omega*timeL+phi),np.sin((k1*x-timeL+0.00001)*DOmega)/(k1*x-timeL+0.00001)/DOmega
    

def corrSac(d, sac0, sac1, name0='', name1='', quakeName='', az=np.array([0, 0]), dura=0, M=np.array([0, 0, 0, 0, 0, 0, 0]), dis=np.array([0, 0]), dep=10, modelFile='', srcSac='', isCut=False, maxCount=-1,fromT=0, **kwags):
    corr = d.sacXcorr(sac0, sac1, isCut=isCut, maxCount=maxCount,fromT=fromT)
    corr.az = az
    corr.dura = dura
    corr.M = M
    corr.dis = dis
    corr.dep = dep
    corr.modelFile = modelFile
    corr.name0 = name0
    corr.name1 = name1
    corr.srcSac = srcSac
    corr.quakeName = quakeName
    return corr


iasp91 = taup(phase_list=['S', 's'])
def IASP91(dep,deg):
    tStart = iasp91(dep,deg)
    if deg > 95:
        tStart = deg*111.19 / 5.5
    return tStart

vMax = 6
vMin = 2.0
VMax = 6.0
VMin = 2.0
tailT = 300
def corrSacsL(d, sacsL, sacNamesL, dura=0, M=np.array([0, 0, 0, 0, 0, 0, 0]), dep=10, modelFile='', srcSac='', minSNR=5, isCut=False, maxDist=100000000.0, minDist=0, maxDDist=100000000.0, minDDist=0, isFromO=False, removeP=False, isLoadFv=False, fvD={}, quakeName='', isByQuake=False, specN=40, specThreshold=0.1, isDisp=False, maxCount=-1,plotDir='',isIqual=False,fromT=0, **kwags):
    modelFileO = modelFile
    if len(plotDir)!=0:
        plt.close()
        plt.figure(figsize=[10,5])
        isPlot=False
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
    TMIN=10000000
    TMAX=-10000
    if len(sacsL) != len(sacNamesL):
        return []
    else:
        corrL = []
        N = len(sacsL)
        distL = np.zeros(N)
        SNR = np.zeros(N)
        for i in range(N):
            distL[i] = sacsL[i][0].stats['sac']['dist']
            if isDisp:
                sacsL[i][0].integrate()
            to = -500
            dto = to - sacsL[i][0].stats['sac']['b']
            io = max(0, int(dto / sacsL[i][0].stats['sac']['delta']))
            te = 100
            dte= te - sacsL[i][0].stats['sac']['b']
            ie = max(0, int(dte / sacsL[i][0].stats['sac']['delta']))
            TS = IASP91(sacsL[i][0].stats['sac']['evdp'], sacsL[i][0].stats['sac']['gcarc'])
            tStart =  TS
            degree = sacsL[i][0].stats['sac']['gcarc']
            #if tStart > 100000.0 or tStart < 5:
            #    tStart = distL[i] / 5
            #t0 = max(distL[i] /5, tStart)###min->max

            t0 = distL[i]/vMax
            dt0 = t0 - sacsL[i][0].stats['sac']['b']
            i0 = max(0, int(dt0 / sacsL[i][0].stats['sac']['delta']))

            tEnd = distL[i] / vMin
            t1 =  tEnd
            dt1 = t1 - sacsL[i][0].stats['sac']['b']
            i1 = min(sacsL[i][0].data.size - 10, int(dt1 / sacsL[i][0].stats['sac']['delta']))
            
            if i1 == sacsL[i][0].data.size:
                SNR[i] = -1
                continue
            if sacsL[i][0].data[i0:i1].std() == 0 or len(sacsL[i][0].data[i0:i1])==0:
                SNR[i] = -1
                continue
            
            dTN  =int(50/sacsL[i][0].stats['sac']['delta'])
            
            #sigmaS= (data[maxI0:maxI1]**2).mean()**0.5
            #sigmaN= (data[io:ie]**2).mean()**0.5
            if isIqual:
                if 'iqual' in sacsL[i][0].stats['sac']:
                    iqual = sacsL[i][0].stats['sac']['iqual']
                    if iqual >1.5:
                        SNR[i]=10000
                    else:
                        SNR[i]=-1
                        if np.random.rand()<0.001:
                            print('control by iqual')
                else:
                    print('no iqual')
                    continue
            else:
                sacNew = sacsL[i][0].copy()
                sacNew.filter('bandpass',freqmin=1/80, freqmax=1/15, corners=3, zerophase=True)
                data=sacNew.data.copy()
                data-=data.mean()
                maxI = np.abs(data[i0:i1]).argmax()+i0
                maxI0 = int(maxI*0.75)
                maxI1 = int(maxI*1.25)
                sigmaS= (data[maxI0:maxI1]**2).mean()**0.5#np.abs(data[i0:i1]).max()#
                sigmaN= (data[io:ie]**2).mean()**0.5#data[io:ie].std()
                SNR[i] = sigmaS / sigmaN
                if SNR[i]<3:
                    if np.random.rand()<0.001:
                        print('control by snr',SNR[i])
            
            if SNR[i]>minSNR:
                DATA = data
                data=sacsL[i][0].data.copy()
                data-=data.mean()
                if len(plotDir)!=0:
                    b=sacsL[i][0].stats['sac']['b']
                    timeL = b+sacsL[i][0].stats['sac']['delta']*np.arange(len(sacsL[i][0].data))
                    data = sacsL[i][0].data
                    plt.plot(timeL,data/np.abs(data).max()+degree,'k',linewidth=0.25)
                    plt.plot(timeL,DATA/np.abs(DATA).max()+degree,'r',linewidth=0.25)
                if len(plotDir)!=0:
                    isPlot=True
                    h0,=plt.plot([tStart,tStart],[degree-0.5,degree+0.5],'r',linewidth=0.5)
                    h1,=plt.plot([tEnd,tEnd],[degree-0.5,degree+0.5],'b',linewidth=0.5)
                    h2,=plt.plot([distL[i]/5,distL[i]/5],[degree-0.5,degree+0.5],'g',linewidth=0.5)
                    h3,=plt.plot([TS,TS],[degree-0.5,degree+0.5],'y',linewidth=0.5)
                    TMAX = max(TMAX,tEnd*1.2)
                    TMIN = min(TMIN,min(tStart*0.8,distL[i] / 5*0.8))
            
            if removeP:
                '''
                sacsL[i][0].data[:i0] *= 0
                sacsL[i][0].data[i1:] *= 0
                sacsL[i][0].data[i0:i1] -= sacsL[i][0].data[i0:i1].mean()
                sacsL[i][0].data[i0:i1] = signal.detrend(sacsL[i][0].data[i0:i1])
                '''
                t0 = distL[i]/VMax
                dt0 = t0 - sacsL[i][0].stats['sac']['b']
                i0 = max(0, int(dt0 / sacsL[i][0].stats['sac']['delta']))

                tEnd = distL[i] / VMin
                t1 =  tEnd
                dt1 = t1 - sacsL[i][0].stats['sac']['b']
                i1 = min(sacsL[i][0].data.size - 10, int(dt1 / sacsL[i][0].stats['sac']['delta']))
                '''
                sacsL[i][0].data[:i0] *= 0
                sacsL[i][0].data[i1:] *= 0
                sacsL[i][0].data[i0:i1] -= sacsL[i][0].data[i0:i1].mean()
                sacsL[i][0].data[i0:i1] = signal.detrend(sacsL[i][0].data[i0:i1])
                '''
                sacsL[i][0].data -= sacsL[i][0].data.mean()
                #sacsL[i][0].data = signal.detrend(sacsL[i][0].data)
                tailCount = int(tailT/sacsL[i][0].stats['sac']['delta'])
                gL        = np.exp(-((np.arange(i0)-i0)/tailCount)**2)
                sacsL[i][0].data[:i0] *= gL
                gL        = np.exp(-((np.arange(i1,len(data))-i1)/tailCount)**2)
                sacsL[i][0].data[i1:] *= gL
            STD = sacsL[i][0].data.std()
            if STD == 0:
                SNR[i] = -1
            else:
                sacsL[i][0].data /= STD

        iL = distL.argsort()
        for ii in range(N):
            for jj in range(ii):
                i = iL[ii]
                j = iL[jj]
                sac0 = sacsL[i][0]
                sac1 = sacsL[j][0]
                name0 = sacNamesL[i][0]
                name1 = sacNamesL[j][0]
                if SNR[i] < minSNR:
                    pass
                else:
                    if SNR[j] < minSNR:
                        pass
                    else:
                        az = np.array([sac0.stats['sac']['az'], sac1.stats['sac']['az']])
                        baz0 = sac0.stats['sac']['baz']
                        az01 = DistAz(sac0.stats['sac']['stla'], sac0.stats['sac']['stlo'], sac1.stats['sac']['stla'], sac1.stats['sac']['stlo']).getAz()
                        dis01 = DistAz(sac0.stats['sac']['stla'], sac0.stats['sac']['stlo'], sac1.stats['sac']['stla'], sac1.stats['sac']['stlo']).getDelta() * 111.19
                        dis = np.array([sac0.stats['sac']['dist'], sac1.stats['sac']['dist']])
                        if dis.min() < minDist:
                            pass
                        else:
                            if dis01 < minDDist:
                                pass
                            else:
                                if dis.max() > maxDist:
                                    pass
                                else:
                                    if dis01 > maxDDist:
                                        pass
                                    else:
                                        minDis = dis.min()
                                        maxD = 250
                                        maxTheta = 10
                                        if np.random.rand() < 0.01:
                                            pass
                                        thetaE = max(10, disDegree(minDis, maxD=maxD, maxTheta=maxTheta))
                                        theta01 = max(10, disDegree(dis01, maxD=maxD, maxTheta=maxTheta))
                                        thetaE = 10
                                        theta01 = 10
                                        if np.abs((az[0] - az[1] + thetaE) % 360) > 2 * thetaE:
                                            pass
                                        else:
                                            if np.abs((baz0 - az01 + theta01) % 360) > 2 * theta01:
                                                pass
                                            else:
                                                modelFile = modelFileO
                                                staName0 = sac0.stats['network'] + '.' + sac0.stats['station']
                                                staName1 = sac1.stats['network'] + '.' + sac1.stats['station']
                                                if staName0 > staName1:
                                                    staName1, staName0 = staName0, staName1
                                                modelFile0 = staName0 + '_' + staName1
                                                modelFile1 = staName1 + '_' + staName0
                                                if isByQuake:
                                                    modelFile0 = quakeName + '_' + modelFile0
                                                    modelFile1 = quakeName + '_' + modelFile1
                                                modelFile = modelFile0
                                                if modelFile0 in fvD:
                                                    modelFile = modelFile0
                                                if modelFile1 in fvD:
                                                    modelFile = modelFile1
                                                if isLoadFv:
                                                    if modelFile0 not in fvD:
                                                        if modelFile1 not in fvD:
                                                            continue
                                                corr = corrSac(d, sac0, sac1, name0, name1, quakeName, az, dura, M, dis, dep, modelFile, srcSac, isCut=isCut, maxCount=maxCount,fromT=fromT,**kwags)
                                                corrL.append(corr)
        if len(plotDir)!=0 and isPlot:
            plt.legend([h0,h1,h2,h3],['start','end','dis/4.6','ts'])
            plt.xlim([TMIN,TMAX])
            print([TMAX,TMIN])
            plt.savefig(plotDir+quakeName+'.jpg',dpi=300)
            plt.close()
        return corrL

class fkcorr:

    def __init__(self, config=config()):
        self.config = config

    def __call__(self, index, iL, f, mul=290, depth0=-1, srcSacIndex=0, dura0=-1, rise0=-1, M0=[], azimuth=[0]):
        for i in iL:
            if i < 0:
                modelFile = '%s' % self.config.originName
            else:
                modelFile = '%s%d' % (self.config.originName, i)
            m = self.config.getModel(modelFile)
            m.covert2Fk(0)
            m.covert2Fk(1)
            if dura0 < 0:
                dura = np.random.rand() * 10 + 20
            else:
                dura = dura0
            if depth0 < 0:
                depth = int(np.random.rand() * 20 + 10) + i % 39
            else:
                depth = depth0
            print('###################################', depth)
            if len(M0) == 0:
                M = np.array([3e+25, 0, 0, 0, 0, 0, 0])
                M[1:] = np.random.rand(6)
            else:
                M = M0
            if srcSacIndex >= 0:
                srcSacIndex = int(np.random.rand() * self.config.srcSacNum * 0.999)
            if rise0 < 0:
                rise = 0.1 + 0.3 * np.random.rand()
            else:
                rise = rise0
            sacsL, sacNamesL = f.test(distance=(self.config.distance + np.round((np.random.rand(self.config.distance.size) - 0.5) * mul)),
              modelFile=modelFile,
              fok=(self.config.fok),
              dt=(self.config.delta),
              depth=depth,
              expnt=(self.config.expnt),
              dura=dura,
              dk=(self.config.dk),
              azimuth=azimuth,
              M=M,
              rise=rise,
              srcSac=getSourceSacName(srcSacIndex, (self.config.delta), srcSacDir=(self.config.srcSacDir)),
              isFlat=(self.config.isFlat))
            with open(modelFile + 'sacFile', 'w') as (ff):
                for sacNames in sacNamesL:
                    for sacName in sacNames:
                        ff.write('%s ' % sacName)

                    ff.write('\n')

                ff.write('#')
                ff.write('%s' % getSourceSacName(srcSacIndex, (self.config.delta), srcSacDir=(self.config.srcSacDir)))