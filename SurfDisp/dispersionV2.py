import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit
from sklearn import cluster
import multiprocessing
from numba import jit,float32, int64
from scipy import fftpack,interpolate
import os 
from scipy import io as sio
import obspy
from multiprocessing import Process, Manager,Pool
import random
from glob import glob
from obspy.taup import TauPyModel
from tensorflow.core.framework.attr_value_pb2 import _ATTRVALUE_LISTVALUE
from ..io.seism import taup
from ..io import seism
from .fk import FK,getSourceSacName,FKL
from ..mathTool.mathFunc import getDetec,xcorrSimple,xcorrComplex,flat,validL,randomSource,disDegree,QC,fitexp
from ..mathTool.distaz import DistAz
import gc
from matplotlib import colors
gpdcExe = '/home/jiangyr/program/geopsy/bin/gpdc'
Vav = -1

class config:
    def __init__(self,originName='models/prem',srcSacDir='/home/jiangyr/home/Surface-Wave-Dispersion/',\
        distance=np.arange(400,1500,100),srcSacNum=100,delta=0.5,layerN=1000,\
        layerMode='prem',getMode = 'norm',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=150,\
        xcorrFuncL = [xcorrSimple,xcorrComplex],isFlat=False,R=6371,flatM=-2,pog='p',calMode='fast',\
        T=np.array([0.5,1,5,10,20,30,50,80,100,150,200,250,300]),threshold=0.1,expnt=10,dk=0.1,\
        fok='/k',gpdcExe=gpdcExe,order=0,minSNR=5,isCut=False,minDist=0,maxDist=1e8,\
        minDDist=0,maxDDist=1e8,para={},isFromO=False,removeP=False,doFlat=True,\
        QMul=1,modelMode='norm', convolveSrc=False):
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
        else:
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
        else:
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
        else:
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
                  resDir=resDir)
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
                 maxCount=maxCount, **kwags)
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
                    fvD[pairKey] = fv((fvFileD[pairKey]), mode=mode)

        return fvD

    def loadQuakeNEFV(self, stations, quakeFvDir='models/QuakeNEFV', quakeD=seism.QuakeL(), isGetQuake=True):
        fvD = {}
        for i in range(len(stations)):
            print('do for %d in %d' % (i + 1, len(stations)))
            for j in range(len(stations)):
                file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt_sel.dat' % (quakeFvDir,
                 stations[i]['net'], stations[i]['sta'], stations[j]['net'], stations[j]['sta']))
                if len(file) > 0:
                    for f in file:
                        getFVFromPairFile(f, fvD, quakeD, isGetQuake=isGetQuake)

                else:
                    file = glob('%s/%s.%s/*_%s.%s/Rayleigh/pvt.dat' % (quakeFvDir,
                     stations[i]['net'], stations[i]['sta'], stations[j]['net'], stations[j]['sta']))
                    if len(file) > 0:
                        for f in file:
                            getFVFromPairFile(f, fvD, quakeD, isGetQuake=isGetQuake)

                    else:
                        file = glob('%s/%s.%s_%s.%s-pvt*' % (quakeFvDir,
                         stations[i]['net'], stations[i]['sta'], stations[j]['net'], stations[j]['sta']))
                        if len(file) > 0:
                            for f in file:
                                getFVFromPairFile(f, fvD, quakeD, isGetQuake=isGetQuake)

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
    def xcorr(self, data0, data1, isCut=True):
        if isCut:
            data1, i0, i1 = self.cut(data1)
        i0 = 0
        i1 = 0
        xx = self.xcorrFunc(data0, data1)
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

    def sacXcorr(self, sac0, sac1, isCut=False, maxCount=-1, **kwags):
        fs = sac0.stats['sampling_rate']
        self.fs = fs
        self.halfN = np.int(self.halfDt * self.fs)
        data0 = sac0.data
        time0 = sac0.stats.starttime.timestamp
        dis0 = sac0.stats['sac']['dist']
        data1 = sac1.data
        time1 = sac1.stats.starttime.timestamp
        dis1 = sac1.stats['sac']['dist']
        xx, i0, i1 = self.xcorr(data0, data1, isCut)
        time1New = time1 + i0 / fs
        dTime = time0 - time1New
        timeL = np.arange(xx.size) / fs + dTime
        dDis = dis0 - dis1
        return corr(xx, timeL, dDis, fs, x0=sac0.data, x1=sac1.data, maxCount=maxCount, **kwags)

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
                    self.STD = interpolate.interp1d((self.f), (self.std), kind='linear', bounds_error=False,
                      fill_value=1e-08)
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

    def save(self, filename, mode='num'):
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        else:
            if mode == 'num':
                np.savetxt(filename, np.concatenate([self.f.reshape([-1, 1]),
                 self.v.reshape([-1, 1])],
                  axis=1))
                return
            if mode == 'NEFile':
                with open(filename, 'w+') as (f):
                    for i in range(len(self.f)):
                        T = 1 / self.f[i]
                        v = self.v[i]
                        std = -1
                        if len(self.std) > 0:
                            std = self.std[i]
                        f.write('%f %f %f\n' % (T, v, std))

    def update(self, self1, isReplace=True, threshold=0.1):
        v = self1(self.f).reshape([-1])
        dv = np.abs(v - self.v)
        if (dv <= threshold * v).sum() > 1:
            self.f = self.f[(dv <= threshold * v)]
            if isReplace:
                self.v = v[(dv < threshold * v)]
            else:
                self.v = self.v[(dv < threshold * v)]
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
    def disQC(self, fvRef, dis, randA=0.15, maxR=2):
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
    def replaceF(self, f):
        f = f.copy()
        v = self(f)
        std = self.STD(f)
        self.f = f[(v > 1)]
        self.v = v[(v > 1)]
        self.std = std[(v > 1)]
        if len(self.f) > 2:
            self.interp = self.genInterp()
    def coverQC(self, dis, minI, maxI, R=0.1):
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


def outputFvDist(fvD, stations, t=16 ** np.arange(0, 1.000001, 0.02040816326530612) * 10, keys=[], keysL=[]):
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


def plotFvDist(distL, vL, fL, filename, fStrike=1, title='', isCover=False, minDis=[], maxDis=[], fLDis=[], R=0.1):
    VL = vL.reshape([-1])
    DISTL = (distL + vL * 0).reshape([-1])
    FL = (fL.reshape([1, -1]) + vL * 0).reshape([-1])
    DISTL = DISTL[(VL > 1)]
    FL = FL[(VL > 1)]
    binF = fL[::fStrike]
    binF.sort()
    binD = np.arange(20) / 18 * 2000
    print(DISTL.max())
    plt.close()
    plt.figure(figsize=[2.5, 2])
    plt.hist2d(DISTL, FL, bins=(binD, binF), rasterized=True, cmap='Greys', norm=(colors.LogNorm()))
    if isCover:
        plt.plot(fLDis, minDis, 'r')
        plt.plot(fLDis, maxDis, 'r')
        plt.plot(fLDis, minDis * (1 - R), '--r')
        plt.plot(fLDis, maxDis * (1 + R), '--r')
    plt.colorbar(label='count')
    plt.gca().set_yscale('log')
    plt.xlabel('distance(km)')
    plt.ylabel('frequency(Hz)')
    plt.title(title)
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
    plt.figure(figsize=[2.5, 2])
    plt.hist2d(VL, FL, bins=(binV, binF), rasterized=True, cmap='Greys', norm=(colors.LogNorm()))
    plt.colorbar(label='count')
    plt.gca().set_yscale('log')
    plt.xlabel('v(km/s)')
    plt.ylabel('frequency(Hz)')
    plt.title(title)
    if isAverage:
        linewidth = 0.5
        plt.plot((fvAverage.v[(fvAverage.v > 1)]), (fvAverage.f[(fvAverage.v > 1)]), '-r', linewidth=linewidth)
        for thres in thresL:
            plt.plot((fvAverage.v[(fvAverage.v > 1)] * (1 + thres)), (fvAverage.f[(fvAverage.v > 1)]), '-.r', linewidth=linewidth)
            plt.plot((fvAverage.v[(fvAverage.v > 1)] * (1 - thres)), (fvAverage.f[(fvAverage.v > 1)]), '-.r', linewidth=linewidth)
    plt.savefig(filename, dpi=300)
    plt.close()
    plt.figure(figsize=[2.5, 2])
    plt.hist(FL, bins=binF)
    plt.gca().set_xscale('log')
    plt.xlabel('f(Hz)')
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
    dv = VLGet - VL
    dvR = dv / VL
    thresholdText = '$d_r \\leq %.1f $%%\n' % (threshold * 100)
    if thresholdForGet > 0:
        thresholdText = thresholdText + '$\\sigma \\leq %.1f$%%' % (thresholdForGet * 100)
    else:
        if thresholdForGet < 0:
            thresholdText = thresholdText + '$Prob \\geq %.2f$%%' % -thresholdForGet
    print((vL > 1).sum())
    print(dvR * 100, FL)
    plt.close()
    binF = fL[::fStrike].copy()
    binF.sort()
    binF[(-1)] *= 1.00000001
    binF[0] *= 0.99999999
    binDVR = np.arange(-0.08, 0.08, 0.001)
    TL = DISL / Vav
    THRESHOLD = delta / TL
    THRESHOLD[THRESHOLD <= threshold] = threshold
    plt.figure(figsize=[3, 3])
    plt.hist2d((dvR[((VL > 1) * (VLGet > 1))]), (FL[((VL > 1) * (VLGet > 1))]), bins=(binDVR, binF), rasterized=True, cmap='Greys')
    plt.colorbar(label='count')
    plt.gca().set_yscale('log')
    plt.xlabel('$dv/v_0$')
    plt.ylabel('$f$(Hz)')
    plt.title(title)
    plt.tight_layout()
    plt.text((binDVR[(-2)]), (binF[(-2)]), thresholdText, va='top', ha='right')
    plt.savefig(filename, dpi=300)
    plt.figure(figsize=[3, 3])
    plt.hist((FL[(VL > 1)]), bins=binF)
    plt.hist((FL[((VL > 1) * (VLGet > 1))]), bins=binF)
    plt.hist((FL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]), bins=binF)
    countAll, a = np.histogram((FL[(VL > 1)]), bins=binF)
    countR, a = np.histogram((FL[((VL > 1) * (VLGet > 1))]), bins=binF)
    countP, a = np.histogram((FL[((VL > 1) * (VLGet > 1) * (np.abs(dvR) <= THRESHOLD))]), bins=binF)
    binMid = (binF[1:] + binF[:-1]) / 2
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(binMid, (countP / countAll * 100), 'k-o', markersize=2, linewidth=1)
    ax2.plot(binMid, (countP / countR * 100), 'r-d', markersize=2, linewidth=1)
    print((countP / countAll)[-2:])
    plt.gca().set_xscale('log')
    ax1.set_ylabel('count')
    ax1.set_xlabel('$f$(Hz)')
    ax2.set_ylabel('Rate(%)')
    ax1.set_ylim([0, 1.25 * countR.max()])
    ax2.set_ylim([min(np.min(countP / countAll * 100) - 5, 50), 106])
    plt.xlim([0.1, 0.00625])
    ax2.text((binF[2]), 105, thresholdText, va='top', ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig((filename[:-4] + 'FCount' + filename[-4:]), dpi=300)
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
        if not isinstance(fvRef, dict):
            vMean = fvRef(fL)
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
                        plt.subplot(2, 3, 2)
                        plt.plot((stations[j].dist(stations[k]) / vMean[i] * fL[i]), (perHit[(i, j, k)]), '.k', markersize=0.5)
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
                        plt.subplot(2, 3, 5)
                        plt.plot((stations[j].dist(stations[k]) / vMean[i] * fL[i]), (perRight[(i, j, k)]), '.k', markersize=0.5)
                    if perRight[(i, j, k)] < plotRate:
                        plt.subplot(2, 3, 6)
                        plt.plot([stations[j]['lo'], stations[k]['lo']], [stations[j]['la'], stations[k]['la']], '-.k', linewidth=0.2)

            plt.savefig(('%s/%.2fmHZ.jpg' % (resDir, fL[i] * 1000)), dpi=300)
            plt.close()


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


def getFVFromPairFile(file, fvD={}, quakeD=seism.QuakeL(), isPrint=False, isAll=False, isGetQuake=True):
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
                if isGetQuake:
                    index = quakeD.find(quake)
                    if index < 0:
                        quakeD.append(quake)
                    else:
                        quake = quakeD[index]
                name = quake.name('_', fmt='%.5f')
                key = '%s_%s' % (name, pairKey)
                if len(f) < 2:
                    continue
                if isAll:
                    fvD[key] = fv([f, v, std], mode='dis')
                else:
                    fvD[key] = fv([f, v, std])
        if isPrint:
            print(len(quakeD.keys()))

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
                key = '%s_%s' % (name, pairKey)
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
        if len(fvD[key].f) < minCount:
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


def averageFVL(fvL, minSta=5, threshold=2.5, minThreshold=0.02, fL=[], isWeight=False, weightType='std'):
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
        vM = np.zeros([len(fL), len(fvL)])
        if isWeight:
            wM = np.zeros([len(fL), len(fvL)])
        for i in range(len(fvL)):
            vM[:, i] = fvL[i](fL)
            if isWeight:
                wM[:, i] = wFunc(fvL[i].STD(fL), weightType)
        vCount = (vM > 1).sum(axis=1)
        f = fL[(vCount >= minSta)]
        vMNew = vM[(vCount >= minSta)]
        if isWeight:
            wMNew = wM[(vCount >= minSta)]
        std = f * 0
        v = f * 0
        for i in range(len(f)):
            if isWeight:
                MEAN, STD, vN = QC((vMNew[i][(vMNew[i] > 1)]), threshold=threshold, minThreshold=minThreshold, minSta=minSta, wL=(wMNew[i][(vMNew[i] > 1)]), it=1)
            else:
                MEAN, STD, vN = QC((vMNew[i][(vMNew[i] > 1)]), threshold=threshold, minThreshold=minThreshold, minSta=minSta, it=1)
            v[i] = MEAN
            std[i] = STD
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


def plotFVM(fvM, fvD={}, fvDRef={}, resDir='test/', isDouble=False, stations=[], keyL=[], **kwags):
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    if len(keyL) == 0:
        keyL = fvM
    for key in keyL:
        key0 = key
        filename = resDir + key + '.jpg'
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
        fvMean = fvD[key]
        fvRef = fvDRef[key0]
        plotFVL(fvL, fvMean, fvRef, filename=filename, title=key, dist=dist, **kwags)


def plotFVL(fvL, fvMean=None, fvRef=None, filename='test.jpg', thresholdL=[1], title='fvL', fL0=[], dist=-1):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.close()
    hL = []
    lL = []
    plt.figure(figsize=[4, 2])
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
    if fvMean != None:
        if len(fL0) > 0:
            fL = fL0.copy()
        else:
            fL = fv.f
        v = fvMean(fL)
        std = fvMean.STD(fL)
        std[v < 1] = np.nan
        v[v < 1] = np.nan
        for threshold in thresholdL:
            plt.plot((v - threshold * std), fL, '-.r', linewidth=0.5, label='$\\pm$std')
            h1, = plt.plot((v + threshold * std), fL, '-.r', linewidth=0.5)

        h2, = plt.plot(v, fL, 'r', linewidth=0.5, label='mean')
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
    plt.xlabel('v(km/s)')
    plt.ylabel('f(Hz)')
    plt.ylim([fL.min(), fL.max()])
    plt.xlim([3, 5])
    plt.tight_layout()
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
            hD, = plt.plot((vRef * 0.99), f, '-.b', linewidth=0.3, label='$\\pm$ 1%')
            plt.plot((vRef * 1.01), f, '-.b', linewidth=0.3)
            plt.legend()
            plt.gca().set_yscale('log')
            plt.ylim([f.min(), f.max()])
            plt.xlim([3, 5])
            plt.xlabel('v(km/s)')
            plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def figSet():
    plt.xlim([3, 5])
    plt.ylim([0.00625, 0.1])
    plt.gca().semilogy()
    plt.xlabel('v/(km/s)')
    plt.ylabel('f/Hz')


def fvD2fvL(fvD, stations, f, isRef=False, fvRef=None):
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
    minI = interpolate.interp1d(fL, minDis)
    maxI = interpolate.interp1d(fL, maxDis)
    return (minDis, maxDis, minI, maxI)


def replaceByAv(fvD, fvDAv, threshold=0, delta=-1, stations='', **kwags):
    notL = []
    threshold0 = threshold
    for modelName in fvD:
        if len(modelName.split('_')) >= 2:
            name0 = modelName.split('_')[(-2)]
            name1 = modelName.split('_')[(-1)]
            modelName0 = '%s_%s' % (name0, name1)
            modelName1 = '%s_%s' % (name1, name0)
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


class corr:
    sigCount = 1

    def __init__(self, xx=np.arange(0, dtype=(np.complex)), timeL=np.arange(0), dDis=0, fs=0, az=np.array([0, 0]), dura=0, M=np.array([0, 0, 0, 0, 0, 0, 0]), dis=np.array([0, 0]), dep=10, modelFile='', name0='', name1='', srcSac='', x0=np.arange(0), x1=np.arange(0), quakeName='', maxCount=-1, up=1):
        self.maxCount = -1
        maxCount0 = xx.shape[0]
        if maxCount < maxCount0:
            if maxCount > 0:
                xx = xx[:maxCount]
                x0 = x0[:maxCount]
                x1 = x1[:maxCount]
                timeL = timeL[:maxCount]
        maxCount = xx.shape[0]
        self.xx = xx.astype(np.complex64)
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

    def getDtype(self, maxCount):
        self.maxCount = maxCount
        corrType = np.dtype([('xx', np.complex64, maxCount),
         (
          'timeL', np.float32, maxCount),
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
          'x0', np.float32, maxCount),
         (
          'x1', np.float32, maxCount),
         (
          'quakeName', np.str, 200)])
        return corrType
    def outputTimeDis(self, FV, T=np.array([5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300]), sigma=2, byT=False, byA=False, rThreshold=0.1, set2One=False, move2Int=False, noY=False, randMove=False):
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
            timeDis = np.exp(-((timeL - t) / tmpSigma) ** 2)
            halfV = np.exp(-(delta * 0.5 / tmpSigma) ** 2)
            if set2One:
                timeDis[timeDis > halfV] = 1
                timeDis[timeDis > 1 / np.e] = 1
                timeDis[timeDis <= 1 / np.e] = 0
            if byA:
                spec = np.abs(np.fft.fft(self.xx))
                minf = 1 / (len(self.timeL) * (self.timeL[1] - self.timeL[0]))
                indexF = (f.reshape([-1]) / minf).astype(np.int)
                maxIndexF = indexF.max()
                spec /= spec[:maxIndexF + 1].mean()
                aF = spec[indexF]
                timeDis[:, aF < rThreshold] = timeDis[:, aF < rThreshold] * 0
            return (
             timeDis, t0)
    def outputTimeDisNew(self, FV, sigma=2, byT=False, byA=False, rThreshold=0.1, set2One=False, move2Int=False, noY=False, T=[]):
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
        return {'xx':self.xx,  'timeL':self.timeL,  'dDis':self.dDis,  'fs':self.fs,  'az':self.az, 
         'dura':self.dura,  'M':self.M,  'dis':self.dis,  'dep':self.dep,  'modelFile':self.modelFile, 
         'name0':self.name0,  'name1':self.name1,  'srcSac':self.srcSac, 
         'x0':self.x0,  'x1':self.x1,  'quakeName':self.quakeName}

    def toMat(self):
        self.dtype = self.getDtype(self.xx.shape[0])
        return np.array((self.xx, self.timeL, self.dDis, self.fs, self.az, self.dura,
         self.M, self.dis, self.dep, self.modelFile, self.name0, self.name1,
         self.srcSac, self.x0, self.x1, self.quakeName), self.dtype)

    def setFromFile(self, file):
        mat = scipy.io.loadmat(file)
        self.setFromDict((mat['corr']), isFile=True)

    def setFromDict(self, mat, isFile=False):
        if not isFile:
            self.xx = mat['xx']
            self.timeL = mat['timeL']
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
            self.quakeName = mat['quakeName']
        else:
            self.xx = mat['xx'][0][0][0]
            self.timeL = mat['timeL'][0][0][0]
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
                    if tmp.compareSpec(N=40) < kwargs['specThreshold']:
                        print('not match')
                        continue
                if 'maxCount' in kwargs:
                    if np.real(tmp.x0)[1:kwargs['maxCount']].std() == 0:
                        print('bad record')
                        continue
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
            new.x0 = new.x0.astype(np.float32)
            new.x1 = new.x1.astype(np.float32)
            for x in [np.real(new.xx), new.x0, new.x1]:
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
        xx = np.real(new.xx)
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

    def load(self, head):
        for matFile in glob(head + '*/*/*.mat'):
            print(matFile)
            tmp = corr()
            tmp.setFromFile(matFile)
            self.append(tmp)

    def loadByDirL(self, heads):
        for head in heads:
            for matFile in glob(head + '*.mat'):
                tmp = corr()
                tmp.setFromFile(matFile)
                self.append(tmp)

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

    def getTimeDis(self, iL, fvD={}, T=[], sigma=2, maxCount=512, noiseMul=0, byT=False, byA=False, rThreshold=0.1, byAverage=False, set2One=False, move2Int=False, modelNameO='', noY=False, randMove=False, randA=0.03, midV=4, randR=0.5, mul=1):
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
        up = self[0].up
        x = np.zeros([len(iL), maxCount, 1, 4], dtype=dtypeX)
        y = np.zeros([len(iL), maxCount * up, 1, len(T)], dtype=dtypeY)
        t0L = np.zeros(len(iL))
        dDisL = np.zeros(len(iL))
        deltaL = np.zeros(len(iL))
        randIndexL = np.zeros(len(iL))
        for ii in range(len(iL)):
            if ii % mul == 0:
                randN = np.random.rand(1)
                Rand0 = np.random.rand(1)
                Rand1 = np.random.rand(1)
            else:
                i = iL[ii]
                maxCount = min(maxCount0, self[i].xx.shape[0], self[i].x0.shape[0], self[i].x1.shape[0])
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
                tmpy, t0 = self[i].outputTimeDis((fvD[modelName]), T=T,
                    sigma=sigma,
                    byT=byT,
                    byA=byA,
                    rThreshold=rThreshold,
                    set2One=set2One,
                    move2Int=move2Int,
                    noY=noY,
                    randMove=randMove)
                iP, iN = self.ipin(t0, self[i].fs)
                y[ii, iP * up:maxCount * up + iN * up, 0, :] = tmpy[-iN * up:maxCount * up - iP * up]
                x[ii, iP:maxCount + iN, 0, 0] = np.real(self[i].xx.reshape([
                    -1]))[-iN:maxCount - iP]
                dDis = self[i].dDis
                if randMove:
                    if randN <= randR:
                        rand1 = 1 + randA * (2 * Rand0 - 1) * Rand1
                        if np.random.rand(1) < 0.001:
                            print('mul', mul)
                            print('******* rand1:', rand1)
                        dDis = dDis * rand1
                delta0 = self[i].timeL[1] - self[i].timeL[0]
                timeMin = dDis / 5.5
                timeMax = dDis / 2.5
                I0 = int(np.round(timeMin / delta0).astype(np.int))
                I1 = int(np.round(timeMax / delta0).astype(np.int))
                x[ii, I0:I1, 0, 1] = 1
                t0L[ii] = t0 - iN / self[i].fs - iP / self[i].fs
                dt = np.random.rand() * 5 - 2.5
                iP, iN = self.ipin(t0 + dt, self[i].fs)
                x[ii, iP:maxCount + iN, 0, 2] = self[i].x0.reshape([
                    -1])[-iN:maxCount - iP]
                iP, iN = self.ipin(dt, self[i].fs)
                x[ii, iP:maxCount + iN, 0, 3] = self[i].x1.reshape([
                    -1])[-iN:maxCount - iP]
                if randMove:
                    if randN > randR:
                        dT = (Rand0 - 0.5) * 2 * (self[i].dDis / midV) * randA * Rand1
                        dN = int(np.round(dT * self[i].fs).astype(np.int))
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
            deltaL[ii] = self[i].timeLOut[1] - self[i].timeLOut[0]
        xStd = x.std(axis=1, keepdims=True)
        self.x = x
        if noiseMul > 0:
            self.x += noiseMul * ((np.random.rand)(*list(x.shape)).astype(dtype) - 0.5) * xStd
        self.y = y
        self.randIndexL = randIndexL
        self.t0L = t0L
        self.dDisL = dDisL
        self.deltaL = deltaL

    def getTimeDisNew(self, iL, fvD={}, T=[], sigma=2, maxCount=512, noiseMul=0, byT=False, byA=False, rThreshold=0.1, byAverage=False, set2One=False, move2Int=False, modelNameO='', noY=False, randMove=False, up=1):
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
            maxCount = min(maxCount0, self[i].xx.shape[0], self[i].x0.shape[0], self[i].x1.shape[0])
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
                    X[:, iP:maxCount + iN, 0] = np.real(self[i].xx.reshape([
                     -1]))[-iN:maxCount - iP]
                    X[:, iP:maxCount + iN, 1] = np.imag(self[i].xx.reshape([
                     -1]))[-iN:maxCount - iP].reshape([1, -1])
                    dt = np.random.rand() * 5 - 2.5
                    iP, iN = self.ipin(t0 + dt, self[i].fs)
                    X[:, iP:maxCount + iN, 2] = self[i].x0.reshape([
                     -1])[-iN:maxCount - iP].reshape([1, -1])
                    iP, iN = self.ipin(dt, self[i].fs)
                    X[:, iP:maxCount + iN, 3] = self[i].x1.reshape([
                     -1])[-iN:maxCount - iP].reshape([1, -1])
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
    def getV(self, yout, isSimple=True, D=0.1, isLimit=False, isFit=False):
        if isSimple:
            maxDis = self.dDisL.max()
            minDis = self.dDisL.min()
            tmin = minDis / 6 - 5
            tmax = maxDis / 2 + 10
            i0 = int(max(1, tmin / self.deltaL[0]))
            i1 = int(min(yout.shape[1] - 1, tmax / self.deltaL[0]))
            pos = yout[:, i0:i1, 0, :].argmax(axis=1) + i0
            prob = pos.astype(np.float64) * 0
            vM = []
            probM = []
            for i in range(pos.shape[0]):
                vM.append([])
                probM.append([])
                for j in range(pos.shape[1]):
                    POS = np.where(yout[i, i0:i1, 0, j] > 0.5)[0] + i0
                    time = self.t0L[i] + POS * self.deltaL[i]
                    vM[(-1)].append(self.dDisL[i] / time)
                    probM[(-1)].append(yout[(i, POS, 0, j)])

            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    prob[(i, j)] = yout[(i, pos[(i, j)], 0, j)]

            time = self.t0L.reshape([-1, 1]) + pos * self.deltaL.reshape([-1, 1])
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
        """
        if len(iL) ==0:
            iL=self.iL
        for i in range(v.shape[0]):
            index = iL[i]
            corr  = self[index]
            f.write('%s %s %s %s | '%(corr.srcSac, corr.modelFile, corr.name0, corr.name1))
            for tmp in T:
                f.write(' %.3f '% (1/tmp))
            f.write('|')
            for tmp in v[i]:
                f.write(' %.3f '% tmp)
            f.write('|')
            for tmp in prob[i]:
                f.write(' %.3f '% tmp)
            f.write('
')
        """
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
                    f.write('%f %f 0 0 \n' % (corr.dis[1] / 110.7, corr.az[1]))
                    f.write('%s 5\n' % station0['comp'][(-1)])
                    f.write('%s %.5f\n' % (obspy.UTCDateTime(time).strftime('%Y %m %d %H %M'), time % 60))
                    f.write('%f %f\n' % (station0['la'], station0['lo']))
                    f.write('%.5f %.5f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[0] / 110.7, corr.az[0]))
                    f.write('2 %d\n' % len(vIndex))
                    for ii in vIndex:
                        f.write('%f\n' % (1 / T[ii]))

                    for ii in vIndex:
                        f.write('%f %f\n' % (v[i][ii], -prob[i][ii]))

    def saveVAll(self, v, prob, T, iL=[], stations=[], minProb=0.7, resDir='models/predict/'):
        """
        if len(iL) ==0:
            iL=self.iL
        for i in range(v.shape[0]):
            index = iL[i]
            corr  = self[index]
            f.write('%s %s %s %s | '%(corr.srcSac, corr.modelFile, corr.name0, corr.name1))
            for tmp in T:
                f.write(' %.3f '% (1/tmp))
            f.write('|')
            for tmp in v[i]:
                f.write(' %.3f '% tmp)
            f.write('|')
            for tmp in prob[i]:
                f.write(' %.3f '% tmp)
            f.write('
')
        """
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
                    f.write('%f %f 0 0 \n' % (corr.dis[1] / 110.7, corr.az[1]))
                    f.write('%s 5\n' % station0['comp'][(-1)])
                    f.write(obspy.UTCDateTime(time).strftime('%Y %m %d %H %M %S\n'))
                    f.write('%f %f\n' % (station0['la'], station0['lo']))
                    f.write('%f %f -1 -1 0\n' % (la, lo))
                    f.write('%f %f 0 0 \n' % (corr.dis[0] / 110.7, corr.az[0]))
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
        """
        if len(iL) ==0:
            iL=self.iL
        for i in range(v.shape[0]):
            index = iL[i]
            corr  = self[index]
            f.write('%s %s %s %s | '%(corr.srcSac, corr.modelFile, corr.name0, corr.name1))
            for tmp in T:
                f.write(' %.3f '% (1/tmp))
            f.write('|')
            for tmp in v[i]:
                f.write(' %.3f '% tmp)
            f.write('|')
            for tmp in prob[i]:
                f.write(' %.3f '% tmp)
            f.write('
')
        """
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
        """
        if len(iL) ==0:
            iL=self.iL
        for i in range(v.shape[0]):
            index = iL[i]
            corr  = self[index]
            f.write('%s %s %s %s | '%(corr.srcSac, corr.modelFile, corr.name0, corr.name1))
            for tmp in T:
                f.write(' %.3f '% (1/tmp))
            f.write('|')
            for tmp in v[i]:
                f.write(' %.3f '% tmp)
            f.write('|')
            for tmp in prob[i]:
                f.write(' %.3f '% tmp)
            f.write('
')
        """
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


class corrD(Dict):

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

    def __call__(self, keyL=[], mul=1, N=-1):
        N0 = N
        iL = []
        if len(keyL) == 0:
            keyL = list(self.keys())
        for key in keyL:
            if not isinstance(key, str):
                key = self.keyL[key]
            if len(self[key]) < mul:
                pass
            else:
                if N0 < 0:
                    N = int(len(self[key]) / mul + 0.99999999999) * mul
                if N > len(self[key]):
                    for i in range(N):
                        iL.append(self[key][(i % len(self[key]))])

                else:
                    iL += random.sample(self[key], N)

        print(len(iL))
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


def corrSac(d, sac0, sac1, name0='', name1='', quakeName='', az=np.array([0, 0]), dura=0, M=np.array([0, 0, 0, 0, 0, 0, 0]), dis=np.array([0, 0]), dep=10, modelFile='', srcSac='', isCut=False, maxCount=-1, **kwags):
    corr = d.sacXcorr(sac0, sac1, isCut=isCut, maxCount=maxCount)
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

def corrSacsL(d, sacsL, sacNamesL, dura=0, M=np.array([0, 0, 0, 0, 0, 0, 0]), dep=10, modelFile='', srcSac='', minSNR=5, isCut=False, maxDist=100000000.0, minDist=0, maxDDist=100000000.0, minDDist=0, isFromO=False, removeP=False, isLoadFv=False, fvD={}, quakeName='', isByQuake=False, specN=40, specThreshold=0.1, isDisp=False, maxCount=-1, **kwags):
    modelFileO = modelFile
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
            to = 0
            dto = to - sacsL[i][0].stats['sac']['b']
            io = max(0, int(dto / sacsL[i][0].stats['sac']['delta']))
            te = distL[i] / 1.8 * 1.5
            dte = te - sacsL[i][0].stats['sac']['b']
            ie = min(sacsL[i][0].data.size, int(dte / sacsL[i][0].stats['sac']['delta']))
            tStart = iasp91(sacsL[i][0].stats['sac']['evdp'], sacsL[i][0].stats['sac']['gcarc']) * 1.1
            if tStart > 100000.0 or tStart < 5:
                tStart = distL[i] / 5
            t0 = min(distL[i] / 4.6, tStart)
            dt0 = t0 - sacsL[i][0].stats['sac']['b']
            i0 = max(0, int(dt0 / sacsL[i][0].stats['sac']['delta']))
            tEnd = distL[i] / 1.5
            t1 = max(1, tEnd)
            dt1 = t1 - sacsL[i][0].stats['sac']['b']
            i1 = min(sacsL[i][0].data.size - 10, int(dt1 / sacsL[i][0].stats['sac']['delta']))
            if i1 == sacsL[i][0].data.size:
                SNR[i] = -1
                continue
            if sacsL[i][0].data[i0:i1].std() == 0:
                SNR[i] = -1
                continue
            SNR[i] = np.max(np.abs(sacsL[i][0].data[i0:i1])) / sacsL[i][0].data[io:io + int((i0 - io) / 5)].std()
            if removeP:
                sacsL[i][0].data[:i0] *= 0
                sacsL[i][0].data[i1:] *= 0
                sacsL[i][0].data[i0:i1] -= sacsL[i][0].data[i0:i1].mean()
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
                        dis01 = DistAz(sac0.stats['sac']['stla'], sac0.stats['sac']['stlo'], sac1.stats['sac']['stla'], sac1.stats['sac']['stlo']).getDelta() * 110.7
                        dis = np.array([sac0.stats['sac']['dist'], sac1.stats['sac']['dist']])
                        if dis.min() < minDist:
                            pass
                        else:
                            if np.abs(dis[0] - dis[1]) < minDDist:
                                pass
                            else:
                                if dis.max() > maxDist:
                                    pass
                                else:
                                    if np.abs(dis[0] - dis[1]) > maxDDist:
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
                                                if modelFile0 in fvD:
                                                    modelFile = modelFile0
                                                else:
                                                    if modelFile1 in fvD:
                                                        modelFile = modelFile1
                                                    else:
                                                        modelFile = modelFile0
                                                if isLoadFv:
                                                    if modelFile0 not in fvD:
                                                        if modelFile1 not in fvD:
                                                            continue
                                                corr = corrSac(d, sac0, sac1, name0, name1, quakeName, az, dura, M, dis, dep, modelFile, srcSac, isCut=isCut, maxCount=maxCount, **kwags)
                                                corrL.append(corr)
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