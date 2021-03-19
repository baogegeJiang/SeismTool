#python
import argparse
import matplotlib.pyplot as plt
import obspy
import math
import scipy.io as sio
import scipy
import numpy as np
from numpy import cos, sin
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras import backend as K
import h5py
import tensorflow as tf
import logging
import random
import sys
sys.path.append('/home/jiangyr/Surface-Wave-Dispersion/')
from SeismTool import deepLearning,io
from SeismTool.io import sacTool
from SeismTool.plotTool import figureSet as fs
genModel0 = deepLearning.fcn.genModel0 
os.environ["MKL_NUM_THREADS"] = "32"
fileDir='/home/jiangyr/accuratePickerV3/testNew/'
isBadPlus=1
styleKey='HBDZKX'
def loadModel(file,mode='norm',phase='p'):
    m = genModel0(mode,phase)
    m.load_weight(file)
    return m
class modelPhase:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(processX(X))


def predict(model, X):
    return model.predict(processX(X))


def predictLongData(model, x, N=2000, indexL=range(750, 1250)):
    if len(x) == 0:
        return np.zeros(0)
    N = x.shape[0]
    Y = np.zeros(N)
    perN = len(indexL)
    loopN = int(math.ceil(N/perN))
    perLoop = int(1000)
    inMat = np.zeros((perLoop, 2000, 1, 3))
    for loop0 in range(0, int(loopN), int(perLoop)):
        loop1 = min(loop0+perLoop, loopN)
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                inMat[loop-loop0, :, :, :] = processX(x[sIndex: sIndex+2000, :])\
                .reshape([2000, 1, 3])
        outMat = model.predict(inMat).reshape([-1, 2000])
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                Y[indexL[0]+sIndex: indexL[-1]+1+sIndex] = \
                outMat[loop-loop0, indexL].reshape([-1])
    return Y


def processX(X, rmean=True, normlize=False, reshape=True,isNoise=False,num=2000):
    if reshape:
        X = X.reshape(-1, num, 1, 3)
    if rmean:
        X-= X.mean(axis=1,keepdims=True)
    if normlize:
        X /=(X.std(axis=(1, 2, 3),keepdims=True))
    if isNoise:
        X+=(np.random.rand(X.shape[0],num,1,3)-0.5)*np.random.rand(X.shape[0],1,1,3)*X.max(axis=(1,2,3),keepdims=True)*0.15*(np.random.rand(X.shape[0],1,1,1)<0.1)
    return X


def processY(Y):
    return Y.reshape(-1, 2000, 1, 1)


def validStd(tmpY,tmpY0,threshold=100, minY=0.2,num=2000):
    tmpY=tmpY.reshape((-1,num))
    tmpY0=tmpY0.reshape((-1,num))
    maxY0=tmpY0.max(axis=1);
    validL=np.where(maxY0>0.9)[0]
    tmpY=tmpY[validL]
    tmpY0=tmpY0[validL]
    maxYIndex=tmpY0.argmax(axis=1)
    i0=250
    i1=1750
    if num ==2000:
        validL=np.where((maxYIndex-i0)*(maxYIndex-i1)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,2000])[:, i0:i1].argmax(axis=1)-\
                tmpY0.reshape([-1,2000])[:, i0:i1].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,2000])[:, i0:i1].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0, 0
    if num==1600:
        validL=np.where((maxYIndex-200)*(maxYIndex-1400)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,num])[:, 200:1400].argmax(axis=1)-\
                tmpY0.reshape([-1,num])[:, 200:1400].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,1600])[:, 200:1400].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0,0
    if num==1200:
        validL=np.where((maxYIndex-200)*(maxYIndex-1000)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,num])[:, 200:1000].argmax(axis=1)-\
                tmpY0.reshape([-1,num])[:, 200:1000].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,1200])[:, 200:1000].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0,0
    if num==1500:
        validL=np.where((maxYIndex-250)*(maxYIndex-1250)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,num])[:, 250:1250].argmax(axis=1)-\
                tmpY0.reshape([-1,num])[:, 250:1250].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,1500])[:, 250:1250].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0,0

    return np.size(validL)/np.size(di),di[validL].mean(),di[validL].std(),len(di)
#sio.savemat(resFile, {'out'+phase+'y': outY, 'out'+phase+'x': xTest, \
#            phase+'y'+'0': yTest})
def compY(tmpY,tmpY0,threshold=100, minY=0.2,returnDi=False):
    num = tmpY.shape[1]#get time length
    i0 = int(num/20)+25#get start index of valid time
    i1 = num - int(num/20)-25#get end index of valid time

    tmpY=tmpY.reshape((-1,num))# change shapes
    tmpY0=tmpY0.reshape((-1,num))

    maxY0=tmpY0[:,i0:i1].max(axis=1)#only time with phase 
    tmpY=tmpY[maxY0>0.9]
    tmpY0=tmpY0[maxY0>0.9]
    
    di = tmpY[:,i0-50:i1+50].argmax(axis=1)-tmpY0[:,i0-50:i1+50].argmax(axis=1)
    maxY =  tmpY.max(axis=1)
    rightN = (np.abs(di[maxY>minY])<=threshold).sum()
    N = (maxY>minY).sum()
    N0 = len(tmpY)
    P = rightN/N
    R = rightN/N0
    F1 = 2/(1/P+1/R)
    rightDi = di[maxY>minY]
    rightDi = rightDi[np.abs(rightDi)<=threshold]
    m = rightDi.mean()
    STD = rightDi.std()
    if returnDi:
        return rightDi
    return P,R,F1,m,STD,rightN


def getResFromFile(fileName,phase='p',threshold=100, minY=0.2):
    data = sio.loadmat(fileName)
    y0   = datap[phase+'y'+'0']
    yout = datap['out'+phase+'y']
    return compY(yout,y0,threshold=threshold,minY=minY)

def shuffle(l,d=100):
    L = []
    n = len(l)
    indexL = np.arange(n)
    for i in range(d):
        for index in np.where(indexL%d==i)[0]:
            L.append(l[index])
    return L


def train(modelFile, resFile, phase='p',validWN=10000,testWN=20000,\
    validNN=5000,testNN=5000,inN=1000,trainWN=10000,trainNN=2000,\
    modelType='norm',\
    waveFile='/media/jiangyr/MSSD/waveforms_11_13_19.hdf5',\
    catalogFile1='data/metadata_11_13_19.csv'\
    ,catalogFile2='phaseDir/hinetFileLstNew',\
    dtP=0.08,dtS=0.16,setting='all'):
    mode=phase
    phase=mode.split('_')[0]
    rms0=1e5
    resCount=50
    logger=logging.getLogger(__name__)
    model,dIndex,channelN= genModel0(modelType,mode)
    logger.info('model mode: %s phase: %s'%(model.config.mode,phase))
    model0 = model.get_weights()
    print(model.summary())
    if phase=='p':
        channelIndex=np.arange(0,1)
    if phase=='s':
        channelIndex=np.arange(1,2)
    if phase=='ps':
        channelIndex=np.arange(3)
    w = h5py.File(waveFile,'r')
    catalog1,d1=sacTool.getCatalogFromFile(catalogFile1,mod='STEAD')
    catalog2,d2=sacTool.getCatalogFromFile(catalogFile2,mod='hinet')
    catalog1   = shuffle(catalog1)
    catalog2   = shuffle(catalog2)
    catalogValid=[]
    catalogTest=[]
    catalogTestSTEAD=catalog1[:testWN]+catalog1[-testNN:]
    catalogTestHinet=catalog2[:testWN]+catalog2[-testNN:]
    catalogTrain=[]
    if setting=='all':
        valid_setting = [catalog1,catalog2]
        train_setting = [catalog1,catalog2,catalog2,catalog2]
    elif setting=='STEAD':
        valid_setting = [catalog1]
        train_setting = [catalog1]
    elif setting=='hinet':
        valid_setting = [catalog2]
        train_setting = [catalog2,catalog2]

    for  catalog in [catalog1,catalog2]:
        catalogTest+=catalog[:testWN]+catalog[-testNN:]
    for  catalog in valid_setting:
        catalogValid+=catalog[testWN:(testWN+validWN)]\
        +catalog[-(testNN+validNN):-testNN]
    for catalog in train_setting:
        catalogTrain+=random.sample(catalog[validWN+testWN:-(testNN+validNN)],min(len(catalog[validWN+testWN:-(testNN+validNN)]),trainWN+trainNN))
        logger.info('add Train set num: %d'\
        %min(len(catalog[validWN+testWN:-(testNN+validNN)]),trainWN+trainNN))



    logger.info('vaild num: %d   testNum: %d  trainNum: %d inN: %d'\
        %(len(catalogValid),len(catalogTest),len(catalogTrain),inN))
    logger.info('dtP: %.3f dtS: %.3f'\
        %(dtP,dtS))
    xValid,yValid,modeValid=sacTool.getXYFromCatalogP(catalogValid,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase,oIndex=-2,dtP=dtP,dtS=dtS)
    #xValid=processX(xValid,isNoise=False,num=dIndex)
    
    increaseCount = 12
    #K.set_value(model.optimizer.lr, 1e-6)
    for i in range(50000):
        catalogIn=random.sample(catalogTrain,inN)
        xTrain,yTrain,modeTrain=sacTool.getXYFromCatalogP(catalogIn,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase,dtP=dtP,dtS=dtS)
        #xTrain=processX(xTrain,isNoise=False,num=dIndex)
        ne =3
        if i >3:
            ne =1
        if i >6 and i%int(increaseCount)==0:
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) \
            * 0.96)#0.95
            increaseCount*=1
        tmpI=i%xTrain.shape[0]
        showXY(xTrain[tmpI],yTrain[tmpI],np.arange(min(yTrain.shape[-1],2)))
        plt.title(modeTrain[tmpI])
        plt.savefig('fig/train_%s/%d_train.jpg'%(phase,i),dpi=300)
        plt.close()
        model.fit(xTrain,yTrain,verbose=0)
        logger.info('loop %d runSample/allSample: %.7f'%(i,inN*(i+1)/len(catalogTrain)))
        if i%3==0:
            thresholds = [50, 25, 5]
            minYL=[0.1,0.5,0.9]
            tmpY=model.predict(xValid,verbose=0)
            tmpI=i%xValid.shape[0]
            showXY(xValid[tmpI],tmpY[tmpI],np.arange(min(yTrain.shape[-1],2)))
            plt.title(modeValid[tmpI])
            plt.savefig('fig/train_%s/out_%d_train.jpg'%(phase,i),dpi=300)
            plt.close()
            for threshold in thresholds:
                for minY in minYL:
                    for cI in channelIndex:
                        if cI==2:
                            continue
                        p,r,f1,m,s,num=compY(tmpY[:,:,:,channelIndex.tolist().index(cI)],\
                            yValid[:,:,:,channelIndex.tolist().index(cI)], threshold=\
                            threshold, minY=minY)
                        logger.info('STEAD channel: %d % 3d : minY:%.2f P:\
                            %.4f R:%.4f F1:%.4f m:%.4f s:%.4f num: %7d'%(cI,threshold,minY,p,r,f1,m,s,num))
            p,r,f1,m,s,num=compY(tmpY[:,:,:,0],yValid[:,:,:,0]\
                ,threshold=20, minY=0.5)
            rms=model.evaluate(x=xValid, y=yValid,verbose=0)
            logger.info('vaild loss: %.9f'%rms)
            rms-=p
            logger.info('vaild rms: %.9f'%rms)
            logger.info('best rms: %.9f'%rms0)
            if rms >= rms0 and p > 0.45 :
                resCount = resCount-1
                if resCount == 0:
                    model.set_weights(model0)
                    logger.info('over fit ,force to stop, set to best model')
                    break
            if rms < rms0 and p > 0.45 :
                resCount = 50
                rms0 = rms
                model0 = (model.get_weights())
                logger.info('find a better model')
    model.set_weights(model0)
    xTrain=[]
    yTrain=[]
    xValid=[]
    yValid=[]
    model.save(modelFile)
    minYL=[0.1,0.5,0.9]
    thresholds = [50, 25, 5]
    xTest,yTest,modeTest=sacTool.getXYFromCatalogP(catalogTestSTEAD,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase,oIndex=-2,dtP=dtP,dtS=dtS)
    #xTest=processX(xTest,isNoise=False,num=dIndex)
    outY = model.predict(xTest,verbose=0)
    for threshold in thresholds:
        for minY in minYL:
            for cI in channelIndex:
                if cI==2:
                    continue
                p,r,f1,m,s,num=compY(outY[:,:,:,channelIndex.tolist().index(cI)],\
                    yTest[:,:,:,channelIndex.tolist().index(cI)],\
                 threshold=threshold, minY=minY)
                logger.info('test STEAD channel: %d % 3d : minY:%.2f P:\
                  %.4f R:%.4f F1:%.4f m:%.4f s:%.4f num: %7d'%(cI,threshold,minY,p,r,f1,m,s,num))
                
    sio.savemat(resFile[:-4]+'_STEAD.mat', {'out'+phase+'y': outY, 'out'+phase+'x': xTest, \
            phase+'y'+'0': yTest})
    xTest,yTest,modeTest=sacTool.getXYFromCatalogP(catalogTestHinet,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase,oIndex=-2,dtP=dtP,dtS=dtS)
    #xTest=processX(xTest,isNoise=False,num=dIndex)
    outY = model.predict(xTest,verbose=0)
    for threshold in thresholds:
        for minY in minYL:
            for cI in channelIndex:
                if cI==2:
                    continue
                p,r,f1,m,s,num=compY(outY[:,:,:,channelIndex.tolist().index(cI)],\
                    yTest[:,:,:,channelIndex.tolist().index(cI)],\
                 threshold=threshold, minY=minY)
                logger.info('test hinet channel: %d % 3d : minY:%.2f P:\
                  %.4f R:%.4f F1:%.4f m:%.4f s:%.4f num: %7d'%(cI,threshold,minY,p,r,f1,m,s,num))
                
    sio.savemat(resFile[:-4]+'_hinet.mat', {'out'+phase+'y': outY, 'out'+phase+'x': xTest, \
            phase+'y'+'0': yTest})
    logger.info('test res save at %s and %s '%(resFile[:-4]+'_STEAD.mat',resFile[:-4]+'_hinet.mat'))

def showXY(x,y,channelL):
    A = x.std()*3
    for i in range(3):
        plt.plot(x[:,:,i]/A+1+i*3,'k',linewidth=0.3)
    for i in channelL:
        plt.plot(y[:,:,i]-i-1,linewidth=0.3)


def plotWave(mat,indexL=[0,1,2,3,4,5],N=3,M=2,phase='p',delta=0.02,width=0.3,fileDir='resDir/',num=20,figStr='(a)'):
    if not os.path.exists(os.path.dirname(fileDir)):
        os.makedirs(os.path.dirname(fileDir))
        print('plot in %s'%os.path.dirname(fileDir))
    strL='abcdefghij'
    compL = 'ENZ'
    fs.init(styleKey)
    y0= mat[phase+'y'+'0']
    maxY0=y0[:,50:-50].max(axis=(1,2,3))
    x = mat['out'+phase+'x'][maxY0>0.5]
    y = mat['out'+phase+'y'][maxY0>0.5]
    y0= y0[maxY0>0.5]
    timeL = np.arange(x.shape[1])*delta
    plt.close()
    filename ='%sres.pdf'%(fileDir)
    plt.figure(figsize=(3,1.8))
    rightDi=compY(y,y0,threshold=150, minY=0.5,returnDi=True)
    plt.hist(rightDi*delta,bins=np.arange(-2,2,0.02))
    plt.ylabel('count')
    plt.xlabel('dt/s')
    plt.xlim([-1,1])
    fs.setABC('('+figStr+')',key=styleKey)
    plt.savefig(filename,dpi=300)
    for loop in range(num):
        filename = '%s%d.pdf'%(fileDir,loop)
        plt.close()
        plt.figure(figsize=[4,6])
        count=0
        for index in indexL:
            index+=len(indexL)*loop
            count+=1
            plt.subplot(N,M,count)
            X = x[index,:,0,:]
            X /= 2.2*X.max()
            Y0= y0[index,:,0,0]
            Y = y[index,:,0,0]
            for comp in range(3):
                plt.plot(timeL,X[:,comp]+3-comp,'k',linewidth=0.3)
            plt.plot(timeL,Y0-0.5,'k',linewidth=0.3)
            plt.plot(timeL,timeL*0-0.5+0.5,'--k',linewidth=0.3)
            plt.plot(timeL,Y-1.6,'k',linewidth=0.3)
            plt.plot(timeL,timeL*0-1.6+0.5,'--k',linewidth=0.3)
            plt.xlim([timeL[0],timeL[-1]])
            plt.ylim([-1.7,4.1])
            if count%M==1:
                plt.yticks([-1.1,0,1,2,3],['q(x)','p(x)','E','N','Z'])
            else:
                plt.yticks([-1.1,0,1,2,3],['','','','',''])
            if count>M*(N-1):
                plt.xlabel('t/s')
            fs.setABC('(%s)'%strL[count-1],key=styleKey)
        plt.savefig(filename,dpi=300)
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--phase', '-p', type=str, help='train for p or s')
    parser.add_argument('--Num', '-N', type=int, help='number of JP phase')
    parser.add_argument('--modelType', '-m', type=str, help='isSoft')
    parser.add_argument('--plot','-P',type=bool, default=False, help='plot')
    parser.add_argument('--figStrL','-f',type=str, default='ab', help='figStrL')
    parser.add_argument('--setting','-s',type=str, default='all', help='setting')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, \
        format='%(asctime)s - %(name)s - %(levelname)s -\
            %(message)s')
    phase=args.phase
    WN=args.Num
    NN=int(WN/5)
    modelType=args.modelType
    figStrL = args.figStrL
    modelFile='model/%s_%s_%d_%d'%(modelType,phase,WN,NN)
    if args.setting=='all':
        resFile='resDir/res_%s_%s_%d_%d.mat'%(modelType,phase,WN,NN)
    else:
        resFile='resDir/res_%s_%s_%d_%d_%s.mat'%(modelType,phase,WN,NN,args.setting)
    logger=logging.getLogger(__name__)
    logger.info('doing train')
    logger.info('model type:%s'%modelType)
    logger.info('phase:%s'%phase)
    logger.info('Phase num:%d Noise num:%d'%(WN,NN))
    logger.info('modelFile: %s,resFile: %s'%(modelFile,resFile))
    if args.plot:
        i=0
        for testSetting in ['STEAD','hinet']:
            resFileTmp = resFile[:-4]+'_%s.mat'%testSetting
            fileDir = resFile[:-4]+'/%s_'%testSetting
            plotWave(sio.loadmat(resFileTmp),fileDir=fileDir,phase=phase,figStr=figStrL[i])
            i+=1
    else:
        train(modelFile,resFile,trainWN=WN,trainNN=NN,phase=phase,\
        modelType=modelType,setting=args.setting)
#{'out'+phase+'y': outY, 'out'+phase+'x': xTest, \
#            phase+'y'+'0': yTest}

'''
inx
inx done
(15551, 2000, 1, 1)
2020-11-09 03:02:46,309 - __main__ - INFO -            STEAD channel: 0  50 : minY:0.10 p:                            0.9869 m:0.6185 s:4.7623
2020-11-09 03:02:46,489 - __main__ - INFO -            STEAD channel: 0  50 : minY:0.50 p:                            0.9815 m:0.6016 s:4.6684
2020-11-09 03:02:46,681 - __main__ - INFO -            STEAD channel: 0  50 : minY:0.90 p:                            0.8906 m:0.4825 s:3.5416
2020-11-09 03:02:46,864 - __main__ - INFO -            STEAD channel: 0  25 : minY:0.10 p:                            0.9803 m:0.6307 s:3.6111
2020-11-09 03:02:47,047 - __main__ - INFO -            STEAD channel: 0  25 : minY:0.50 p:                            0.9753 m:0.6329 s:3.5882
2020-11-09 03:02:47,245 - __main__ - INFO -            STEAD channel: 0  25 : minY:0.90 p:                            0.8877 m:0.4929 s:2.8190
2020-11-09 03:02:47,427 - __main__ - INFO -            STEAD channel: 0   5 : minY:0.10 p:                            0.8713 m:0.1311 s:1.5278
2020-11-09 03:02:47,605 - __main__ - INFO -            STEAD channel: 0   5 : minY:0.50 p:                            0.8680 m:0.1316 s:1.5282
2020-11-09 03:02:47,790 - __main__ - INFO -            STEAD channel: 0   5 : minY:0.90 p:                            0.8165 m:0.1370 s:1.4864
15551/15551 [==============================] - 15s 964us/step
2020-11-09 03:03:04,944 - __main__ - INFO -            over fit ,force to stop, set to best model
inx
inx done
2020-11-09 03:03:58,125 - __main__ - INFO -            test STEAD channelP:0   50 : minY:0.10                     p:0.9906 m:-0.5220 s:4.7276
2020-11-09 03:03:58,411 - __main__ - INFO -            test STEAD channelP:0   50 : minY:0.50                     p:0.9873 m:-0.5170 s:4.6597
2020-11-09 03:03:58,709 - __main__ - INFO -            test STEAD channelP:0   50 : minY:0.90                     p:0.9186 m:-0.5172 s:3.5163
2020-11-09 03:03:58,998 - __main__ - INFO -            test STEAD channelP:0   25 : minY:0.10                     p:0.9825 m:-0.5602 s:3.5593
2020-11-09 03:03:59,282 - __main__ - INFO -            test STEAD channelP:0   25 : minY:0.50                     p:0.9796 m:-0.5614 s:3.5382
2020-11-09 03:03:59,581 - __main__ - INFO -            test STEAD channelP:0   25 : minY:0.90                     p:0.9156 m:-0.5439 s:2.9674
2020-11-09 03:03:59,866 - __main__ - INFO -            test STEAD channelP:0    5 : minY:0.10                     p:0.8719 m:-0.5605 s:1.6176
2020-11-09 03:04:00,154 - __main__ - INFO -            test STEAD channelP:0    5 : minY:0.50                     p:0.8703 m:-0.5615 s:1.6162
2020-11-09 03:04:00,485 - __main__ - INFO -            test STEAD channelP:0    5 : minY:0.90                     p:0.8329 m:-0.5678 s:1.5980
'''
