from tensorflow import keras
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,\
  Dropout, Dense,Softmax,Conv1D,Reshape,DenseFeatures,LayerNormalization,UpSampling2D,UpSampling1D,Multiply,SeparableConv2D,SeparableConv1D,DepthwiseConv2D,Permute,Flatten
from tensorflow.keras.layers import BatchNormalization
from layer import MBatchNormalization
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.keras import initializers, regularizers, constraints, activations
#LayerNormalization = keras.layers.BatchNormalization
import numpy as np
#from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt   
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
import random
import os
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numexpr as ne
K.set_floatx('float32')
def defProcess():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
# this module is to construct deep learning network
    
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
'''
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#config.gpu_options.allow_growth = False
#session =tf.compat.v1.Session(config=config)
#K.set_session(session)
#defProcess()
'''
class LayerNormalization(Layer):
    """Layer Normalization Layer.
    
    # References
        [Layer Normalization](http://arxiv.org/abs/1607.06450)
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps
    
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config.update({
            'eps': self.eps,
        })
        return config
'''
#默认是float32位的网络
isUnlabel=1#1
W0=0.04
def getW(T,W0):
    return W0


class lossFuncMSE:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        disRandA=1/15
        disMaxR=1
        #self.w = K.ones([1,maxCount,1,len(TL)])
        w = np.ones([1,maxCount,1,len(TL)])*0.05
        for i in range(len(TL)):
            i0 = int(TL[i]/disMaxR/delta*(1+randA))
            i1 = min(int(TL[i]/disRandA/delta*(1-randA)),maxCount)
            w[0,:,0,i]=getW(TL[i],W0)
            #self.w[0,:,0,i]=10/TL[i]
            w[0,:i0,0,i]=0
            w[0,i1:,0,i]=0
        #self.w[:]=w
        #K.set_value(self.w,w)
        self.w = K.variable(w,dtype='float32')
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = (K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return K.mean(\
                         (\
                             (y0-yout0)**2
                                                                     )*\
                         (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample)),\
                                                                         )
class lossFuncMSENP:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        K=np
        self.__name__ = 'lossFuncSoft'
        disRandA=1/15
        disMaxR=1
        self.w = K.ones([1,maxCount,1,len(TL)])
        w = np.ones([1,maxCount,1,len(TL)])*0.05
        for i in range(len(TL)):
            i0 = int(TL[i]/disMaxR/delta*(1+randA))
            i1 = min(int(TL[i]/disRandA/delta*(1-randA)),maxCount)
            w[0,:,0,i]=getW(TL[i],W0)
            #self.w[0,:,0,i]=10/TL[i]
            w[0,:i0,0,i]=0
            w[0,i1:,0,i]=0
        self.w=w
    def __call__(self,y0,yout0):
        return callMSE(y0,yout0,self.w)
        
                                              
def callMSE(y0,yout0,w,isUnlabel=isUnlabel):
    K=np
    #y1 = 1-y0
    #yout1 = 1-yout0
    #yout0 = K.clip(yout0,1e-7,1)
    #yout1 = K.clip(yout1,1e-7,1)
    maxChannel  = ((K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2)
    maxSample   = K.max(maxChannel,axis=3, keepdims=True)
    isUnlabel=np.array(isUnlabel,dtype=np.float32)
    return ne.evaluate('(y0-yout0)**2*(maxChannel + isUnlabel*w*(1-maxChannel)*maxSample+(1-maxSample) )').mean()

def hitRate(yin,yout,maxD=10):
    yinPos  = K.argmax(yin ,axis=1)
    youtPos = K.argmax(yout,axis=1)
    d       = K.abs(yinPos - youtPos)
    count   = K.sum(K.sign(d+0.1))
    hitCount= K.sum(K.sign(-d+maxD))
    return hitCount/count

up=-1
def rateNp(yinPos,youtPos,yinMax,youtMax,maxD=0.03,K=np,minP=0.5,fromI=-384*2):
    threshold = (yinPos+fromI)*maxD
    threshold[threshold<up]=up
    d0      = youtPos-yinPos
    d       = K.abs(yinPos - youtPos)
    count0   = K.sum(yinMax>0.5)
    count1   = K.sum((yinMax>0.5)*(youtMax>minP))
    hitCount= K.sum((d<=threshold)*(yinMax>0.5)*(youtMax>minP))
    recall = hitCount/count0
    right  = hitCount/count1
    F = 2/(1/recall+1/right)
    res = d0[(d<threshold*3)*(yinMax>0.5)*(youtMax>minP)]/(yinPos[(d<threshold*3)*(yinMax>0.5)*(youtMax>minP)]+fromI)
    return recall, right,F,res.mean(),res.std()


globalFL = [0]


def printRes_old(yin, yout,resL=globalFL,isUpdate=False,N=10):
    #       0.01  0.8  0.36600 0.9996350
    strL   = 'maxD   minP hitRate rightRate F mean  std old'
    strfmt = '\n%5.3f&%3.1f&%7.5f&%7.5f&%7.5f&%7.3f&%7.3f'
    yinPos,yinMax  = mathFunc.Max(yin,N=N)
    youtPos,youtMax = mathFunc.Max(yout,N=N)
    for maxD in [0.03,0.015,0.01]:
        for minP in [0.5,0.6,0.8]:
            hitRate,rightRate,F,mean,std = rateNp(\
                yinPos,youtPos,yinMax,youtMax,maxD=maxD,minP=minP)
            tmpStr=strfmt%(maxD, minP, hitRate, rightRate,F,mean*100,std*100)
            strL += tmpStr
            if maxD == 0.015 and minP==0.5 and (not np.isnan(F)) and isUpdate:
                resL.append(F)
        strL+='\n'+('-'*len(tmpStr))
    print(strL)
    return strL


def printRes(yin, yout):
    #       0.01  0.8  0.36600 0.9996350
    strL   = 'maxD   minP hitRate rightRate F'
    strfmt = '\n%5.3f %3.1f %7.5f %7.5f %7.5f'
    try:
        yinPos, yinMax = findPos(yin)
        youtPos, youtMax = findPos(yout) 
        for maxD in [0.03,0.02,0.01,0.005]:
            for minP in [0.5,0.7,0.8,0.9]:
                hitRate,rightRate, F = rateNp(\
                    yinPos,youtPos,yinMax,youtMax,maxD=maxD,minP=minP)
                strL += strfmt%(maxD, minP, hitRate, rightRate,F)
    except:
        print('cannot find')
    else:
        pass
    print(strL)
    return strL

def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    last    = inputs
    CA = -1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
    BNA =CA
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    upL  = [None for i in range(depth+1)]
    momentum=0.95
    renorm_momentum=0.95
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2,data_format=config.data_format)(last)
        else:
            if config.data_format=='channels_first':
                wave = last[:,:1,:,:]
                dist = last[:,1:,:,:]
            else:
                wave = last[:,:,:,:1]
                dist = last[:,:,:,1:]
            #last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
            last = K.concatenate([DepthwiseConv2D(kernel_size=(config.kernelFirstL[j],1),strides=(1,1),padding='same',name=name+layerStr+'_wave_0_%d'%j,kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(wave) for j in range(config.outputSize[-1])]+[dist],axis=CA)
            last = Conv2D(config.featureL[i],kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],data_format=config.data_format)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        last=config.poolL[i](pool_size=config.strideL[i],\
                strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            print(i)
            layerStr='_%d_%d'%(i,j)
            dConvL[j]  = UpSampling2D(size=config.strideL[j],interpolation="bilinear")(dConvL[j+1])
            dConvL[j] = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                outputsL.append(SeparableConv2D(config.outputSize[-1],kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',depth_multiplier=1)(dConvL[0]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    if config.data_format=='channels_first':
        outputs = Permute([2,3,1])(outputs)
    return inputs,outputs

class xyt:
    def __init__(self,x,y,t=''):
        self.x = x
        self.y = y
        self.t = t
        self.timeDisKwarg={'sigma':-1}
    def __call__(self,iL):
        if not isinstance(iL,np.ndarray):
            iL= np.array(iL).astype(np.int)
        if len(self.t)>0:
            tout = self.t[iL]
        else:
            tout = self.t
        self.iL = iL
        return self.x[iL],self.y[iL],tout
    def __len__(self):
        return self.x.shape[0]

class fcnConfig:
    def __init__(self,mode='surf',up=1,mul=1,**kwags):
        self.mode=mode
        if mode=='surf':
            self.inputSize     = [512*3,1,4]
            self.outputSize    = [512*3,1,50]
            self.featureL      = [min(2**(i+1)+20,80) for i in range(7)]
            self.featureL      = [30,40,60,60,80,60,40]
            self.featureL      = [15,20,20,25,25,40,60]
            self.featureL      = [32,32,64,64,64,128,128]#[8,16,32,64,128,128,256]
            self.featureL      = [32,32,32,64,64,64,128]
            self.featureL      = [32,32,32,64,64,64,128]
            self.featureL      = [24,24,32,48,48,64,128]
            self.featureL      = [32,32,48,48,64,64,128]
            self.featureL      = [32,48,48,64,64,96,128]
            self.featureL      = [32,32,32,32,48,64,96,128]
            self.featureL      = [16,32,48,64,128,256,512]#high
            self.featureL      = [24,32,48,64,128,256,512]
            self.featureL      = [64,128,256,512,256*3,256*3,1028,1028]
            self.featureL      = [32,48,86,64*3,128,256,128*3,512]
            self.featureL      = [16,32,48,64,128,64*3,256,512,1024]
            self.featureL      = [24,48,96,128,256,384,512,1024,2048]
            self.featureL      = [16,32,64,96,192,256,384,512,1024]
            self.featureL      = [24,48,64,96,128,192,256,384,512]
            self.featureL      = [24,48,64,96,128,192,256,256,384]
            self.featureL      = [16,32,48,64,96,128,198,256,256]
            self.featureL      = [16,32,48,64,96,128,198,256,256]
            self.featureL      = [12,24,32,48,64,96,128,198,256]
            self.featureL      = [8,16,24,32,48,64,96,128,198]
            self.featureL      = [8,12,16,24,32,48,64,96,128]
            self.featureL      = [6,8,12,16,24,36,48,64,96]
            self.featureL      = [8,12,16,24,36,48,64,96,128]
            self.featureL      = [6,8,12,16,24,32,48,64,96]
            self.strideL       = [(2,1),(4,1),(4,1),(4,1),(4,1),(4,1),(6,1),\
            (4,1),(2,1),(2,1),(2,1)]
            self.strideL       = [(2,1),(3,1),(2,1),(4,1),(2,1),(4,1),(2,1),\
            (4,1),(2,1),(2,1),(2,1)]
            self.strideL       = [(2,1),(3,1),(2,1),(2,1),(2,1),(4,1),(2,1),(2,1),\
                (2,1),(2,1),(2,1)]
            self.strideL       = [(2,1),(3,1),(2,1),(2,1),(2,1),(2,1),(2,1),(2,1),\
                (2,1),(2,1),(2,1)]
            self.kernelL       = [(6,1),(8,1),(8,1),(8,1),(8,1),(16,1),(6,1),\
            (8,1),(4,1),(4,1),(4,1)]
            self.initializerL  = ['truncated_normal' for i in range(10)]
            self.initializerL  = ['he_normal' for i in range(10)]
            self.bias_initializerL = ['random_normal' for i in range(10)]
            self.bias_initializerL = ['he_normal' for i in range(10)]
            self.dropOutL     =[]# [0,1,2]#[5,6,7]#[1,3,5,7]#[1,3,5,7]
            self.dropOutRateL = []#[0.2,0.2,0.2]#[0.2,0.2,0.2]
            self.activationL  = ['relu','relu','relu','relu','relu',\
            'relu','relu','relu','relu','relu','relu']
            self.activationL  = ['relu','relu']+['swish' for i in range(5)]+['relu','relu']
            #self.activationL  = ['relu','relu','relu','relu','relu',\
            #'relu','relu','relu','relu','relu','relu']
            self.poolL        = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.poolL        = [MaxPooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.lossFunc     = lossFuncSoft(w=10)#10
            self.inAndOutFunc = inAndOutFuncNewV6
            self.deepLevel = 1
        if mode=='surfUp':
            self.data_format = 'channels_last'
            self.mul           = mul
            self.inputSize     = [512*2,mul,4]
            self.outputSize    = [512*2*up,mul,50]
            self.inputSize     = [512*12,mul,4]
            self.outputSize    = [512*12*up,mul,50]
            self.inputSize     = [512*6,mul,4]
            self.outputSize    = [512*6*up,mul,50]
            self.featureL      = [75 ,150,300,400,600,800,1000,320]
            self.featureL      = [80 ,120,240,320,480,640,800,320]
            self.featureL      = [60 ,80 ,150,200,300,400,500,320]
            self.kernelFirstL  = ((14**np.arange(0,1,1/49)*10)*6).astype(np.int)
            self.strideL       = [(2,1),  (2,1), (3,1), (4,1), (4,1),  (4,1),(4,1),(1,mul)  ,(up,1)]
            self.kernelL       = [(int(160*2*1.5),1),(12,1),(12,1),(12,1),(12,1), (8,1),(8,1),(1,mul*2),(up*12,1)]
            self.isBNL       = [True]*20
            self.doubleConv    = [True]*20
            #self.kernelL       = [(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(4,1),(4,1)]
            self.initializerL  = ['truncated_normal' for i in range(20)]
            self.initializerL  = ['he_normal' for i in range(20)]
            self.bias_initializerL = ['random_normal' for i in range(20)]
            self.bias_initializerL = ['he_normal' for i in range(20)]
            self.dropOutL     =[]# [0,1,2]#[5,6,7]#[1,3,5,7]#[1,3,5,7]
            self.dropOutRateL = []#[0.2,0.2,0.2]#[0.2,0.2,0.2]
            #self.activationL  = ['relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu']
            #self.activationL  = ['relu','relu']+['swish' for i in range(5)]+['relu','relu']
            self.activationL  = ['relu']*10
            self.activationL  = ['selu']*10
            self.activationL  = ['relu']*10
            self.poolL        = [MaxPooling2D,MaxPooling2D,MaxPooling2D,\
            MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,\
            MaxPooling2D,MaxPooling2D,MaxPooling2D]
            #self.poolL        = [MaxPooling2D]*20
            self.poolL        = [MaxPooling2D]*20
            #self.poolL        = [AveragePooling2D]*20
            self.lossFunc     = lossFuncMSE(**kwags)#1lossFuncSoft(**kwags)#0
            self.inAndOutFunc = inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT#inAndOutFuncNewUp
            self.lossFuncNP     = lossFuncMSENP(**kwags)#lossFuncSoftNP(**kwags)#
            self.deepLevel = 1
        if mode=='surfMul':
            self.mul           = mul
            self.inputSize     = [20,1000,1]
            self.inputSizeI    = [20,1]
            self.outputSize    = [20,1000,50]
            self.featureL      = [75 ,150,300,450,600,800]
            self.featureL      = [75 ,150,300,450,600,800]
            self.featureL      = [75 ,150,300,600,900,800]
            self.featureL      = [75 ,150,300,600,1200,800]
            self.featureL      = [60 ,120,240,480,640,800]
            #self.featureL      = [75 ,100,125,150,200,250,720]
            #self.AGSizeL        = [100,75,50,50,50,50,320]
            #self.featureL      = [10,10,15,20,25,50,320]
            #self.featureL      = [50,50,75,75,100,100,125]
            #self.featureL      = [50,75,75,100,125,150,200]
            #self.featureL      = [75,75,75,75,75,75,75]
            #self.featureL      = [8,12,16,32,48,64,96,128,160]
            #self.featureL      = [10,15,20,30,40,60,80,100,120]
            #self.featureL      = [10,15,20,30,50,80,100,120,160]
            #self.featureL      = [12,16,24,32,64,128,256,512,512]
            self.strideL       = [(1,5), (1,5), (1,5), (1,4), (1,2), (1,mul)  ,(1,1)]
            self.kernelL       = [(1,20),(1,20),(1,20),(1,12), (1,8),(1,mul*2),(1,10)]
            #self.dilation_rate = [(1,1), (4,1), (16,1), (32,1),(128,1),(768,1),(768,1),(512,1),(up,1)]
            #self.kernelL       = [(8,1),(8,1),(8,1),(8,1), (8,1),(8,1),(8,1),(8,1),(8,1),(1,mul*2),(up*8,1)]
            #self.kernelL       = [(6,1),(6,1),(8,1),(8,1),(8,1),(8,1),(1,mul*2),(up*4,1)]
            #self.kernelL       = [(6,1),(9,1),(12,1),(12,1),(16,1),(8,1),(1,mul*2),(up*3,1)]
            self.isBNL       = [True]*20
            self.doubleConv    = [True]*20
            #self.kernelL       = [(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(4,1),(4,1)]
            self.initializerL  = ['truncated_normal' for i in range(20)]
            self.initializerL  = ['he_normal' for i in range(20)]
            self.bias_initializerL = ['random_normal' for i in range(20)]
            self.bias_initializerL = ['he_normal' for i in range(20)]
            self.dropOutL     =[]# [0,1,2]#[5,6,7]#[1,3,5,7]#[1,3,5,7]
            self.dropOutRateL = []#[0.2,0.2,0.2]#[0.2,0.2,0.2]
            #self.activationL  = ['relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','relu']
            #self.activationL  = ['relu','relu']+['swish' for i in range(5)]+['relu','relu']
            self.activationL  = ['relu']*10
            self.activationL  = ['selu']*10
            self.activationL  = ['relu']*10
            self.poolL        = [MaxPooling2D,MaxPooling2D,MaxPooling2D,\
            MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,\
            MaxPooling2D,MaxPooling2D,MaxPooling2D]
            #self.poolL        = [MaxPooling2D]*20
            #self.poolL        = [MaxPooling2D]*20
            self.poolL        = [AveragePooling2D]*20
            self.lossFunc     =lossFuncMSEMul(N=10)#1lossFuncSoft(**kwags)#0
            self.inAndOutFunc = inAndOutFuncMul_#inAndOutFuncNewUp
            self.lossFuncNP     = 'mse'#lossFuncSoftNP(**kwags)#
            self.deepLevel = 1
        elif mode=='surfdt':
            self.inputSize     = [512*3,1,4]
            self.outputSize    = [512*3,1,1]
            self.featureL      = [min(2**(i+1),20) for i in range(7)]
            self.featureL      = [30,40,60,60,80,60,40]
            self.featureL      = [15,20,20,25,25,40,60]
            self.featureL      = [32,32,64,64,64,128,128]#[8,16,32,64,128,128,256]
            self.featureL      = [32,32,32,64,64,64,128]
            self.featureL      = [32,32,32,64,64,64,128]
            self.featureL      = [24,24,32,48,48,64,128]
            self.featureL      = [32,32,48,48,64,64,128]
            self.featureL      = [32,48,48,64,64,96,128]
            self.featureL      = [32,32,32,32,48,64,96,128]
            self.featureL      = [16,32,48,64,128,256,512]#high
            self.featureL      = [12,24,48,64,128,256,512]
            self.featureL      = [64,128,256,512,256*3,256*3,1028,1028]
            self.strideL       = [(2,1),(4,1),(4,1),(4,1),(4,1),(4,1),(6,1),\
            (4,1),(2,1),(2,1),(2,1)]
            self.strideL       = [(2,1),(3,1),(2,1),(4,1),(2,1),(4,1),(2,1),\
            (4,1),(2,1),(2,1),(2,1)]
            self.strideL       = [(2,1),(3,1),(2,1),(2,1),(2,1),(4,1),(2,1),(2,1),\
                (2,1),(2,1),(2,1)]
            self.kernelL       = [(6,1),(8,1),(8,1),(8,1),(8,1),(16,1),(6,1),\
            (8,1),(4,1),(4,1),(4,1)]
            self.kernelL       = [(4,1),(6,1),(4,1),(8,1),(4,1),(4,1),(4,1),(4,1),\
                (4,1),(4,1),(4,1)]
            self.initializerL  = ['truncated_normal' for i in range(10)]
            self.initializerL  = ['he_normal' for i in range(10)]
            self.bias_initializerL = ['random_normal' for i in range(10)]
            self.bias_initializerL = ['he_normal' for i in range(10)]
            self.dropOutL     =[]# [0,1,2]#[5,6,7]#[1,3,5,7]#[1,3,5,7]
            self.dropOutRateL = []#[0.2,0.2,0.2]#[0.2,0.2,0.2]
            self.activationL  = ['relu','relu','relu','relu','relu',\
            'relu','relu','relu','relu','relu','relu']
            self.activationL  = ['relu','relu','relu']+['swish' for i in range(4)]+['relu']
            self.poolL        = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.poolL        = [MaxPooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.lossFunc     = lossFuncSoftSq(w=10)#10
            self.inAndOutFunc = inAndOutFuncDt
            self.deepLevel = 1
        elif mode == 'p' or mode=='s':
            self.inputSize     = [2000,1,3]
            self.outputSize    = [2000,1,1]
            #self.featureL      = [min(2**(i+1)+20,80) for i in range(7)]#high
            #self.featureL      = [8,16,32,64,128,256,512]
            #self.featureL      = [6,12,24,48,96,192,384]
            self.featureL      = [4,8,16,32,64,192,192*3]#old setting
            self.featureL      = [6,12,24,48,96,192,192*3]
            self.featureL      = [8,16,32,64,96,192,192*3]
            self.featureL      = [8,16,32,64,128,192,192*3]
            self.featureL      = [12,24,32,64,128,192,192*3]
            self.featureL      = [12,24,32,64,128,192,192*3]
            self.featureL      = [12,24,48,96,120,144,192]
            self.featureL      = [16,32,64,128,256,128*4,128*8]
            self.featureL      = [16,24,32,48,64,96,128]
            #self.featureL      = [6,12,24,48,96,192,288]
            self.strideL       = [(2,1),(2,1),(2,1),(2,1),(5,1),(5,1),(5,1)]
            #self.kernelL       = [(4,1),(4,1),(4,1),(4,1),(10,1),(10,1),(10,1),\
            #(8,1),(4,1),(4,1),(4,1)]
            self.kernelL       = [(4,1),(4,1),(4,1),(4,1),(10,1),(10,1),(5,1),\
            (8,1),(4,1),(4,1),(4,1)]
            #self.initializerL  = ['truncated_normal' for i in range(10)]
            self.initializerL  = ['he_normal' for i in range(10)]
            #self.bias_initializerL = ['random_normal' for i in range(10)]
            self.bias_initializerL = ['he_normal' for i in range(10)]
            self.dropOutL     =[]# [0,1,2]#[5,6,7]#[1,3,5,7]#[1,3,5,7]
            self.dropOutRateL = []#[0.2,0.2,0.2]#[0.2,0.2,0.2]
            self.activationL  = ['relu','relu','relu','relu','relu',\
            'relu','relu','relu','relu','relu','relu']
            #self.activationL  = ['relu','relu']+['swish' for i in range(4)]+['relu']
            #self.activationL  = ['relu','swish' ,'relu','swish','relu','swish' ,'relu']
            '''
            self.poolL        = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.poolL        = [MaxPooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            '''
            self.poolL        = [MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,]
            if mode=='p':
                self.lossFunc     = lossFuncNew#10
            elif mode =='s': 
                self.lossFunc     = lossFuncNewS
            self.inAndOutFunc =inAndOutFuncNewV6
            self.deepLevel = 1
        elif mode == 'p_Net' or mode=='s_Net':
            self.inputSize     = [2000,1,3]
            self.outputSize    = [2000,1,1]
            #self.featureL      = [min(2**(i+1)+20,80) for i in range(7)]#high
            #self.featureL      = [8,16,32,64,128,256,512]
            #self.featureL      = [6,12,24,48,96,192,384]
            self.featureL      = [4*6,8*5,16*4,32*3,64*2,192*1,192*3*1]#old setting
            self.featureL      = [8*4,16*3,16*5,32*4,64*3,64*5,192*3*1]#
            self.featureL      = [12*4,24*3,48*2,48*3,96*2,120*2,192]
            self.featureL      = [12*6,24*4,48*4,96*4,128*3,256*3,512*2]
            #self.featureL      = [16,32,48,96,192,96*3,96*4]
            #self.featureL      = [6,12,24,48,96,192,288]
            self.strideL       = [(2,1),(2,1),(2,1),(2,1),(5,1),(5,1),(5,1)]
            #self.kernelL       = [(4,1),(4,1),(4,1),(4,1),(10,1),(10,1),(10,1),\
            #(8,1),(4,1),(4,1),(4,1)]
            self.kernelL       = [(4,1),(4,1),(4,1),(4,1),(10,1),(10,1),(5,1),\
            (8,1),(4,1),(4,1),(4,1)]
            #self.initializerL  = ['truncated_normal' for i in range(10)]
            self.initializerL  = ['he_normal' for i in range(10)]
            #self.bias_initializerL = ['random_normal' for i in range(10)]
            self.bias_initializerL = ['he_normal' for i in range(10)]
            self.dropOutL     =[]# [0,1,2]#[5,6,7]#[1,3,5,7]#[1,3,5,7]
            self.dropOutRateL = []#[0.2,0.2,0.2]#[0.2,0.2,0.2]
            self.activationL  = ['relu','relu','relu','relu','relu',\
            'relu','relu','relu','relu','relu','relu']
            #self.activationL  = ['relu','relu']+['swish' for i in range(4)]+['relu']
            #self.activationL  = ['relu','swish' ,'relu','swish','relu','swish' ,'relu']
            self.poolL        = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.poolL        = [MaxPooling2D,AveragePooling2D,MaxPooling2D,\
            AveragePooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
            MaxPooling2D,AveragePooling2D,MaxPooling2D]
            self.poolL        = [MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,]
            if mode[0]=='p':
                self.lossFunc     = lossFuncNew#10
            elif mode[0] =='s': 
                self.lossFunc     = lossFuncNewS
            self.inAndOutFunc =inAndOutFuncNewNet
            self.deepLevel = 1
    def inAndOut(self,*argv,**kwarg):
        return self.inAndOutFunc(self,*argv,**kwarg)

batchMax=32#8*24/16
class modelUp(Model):
    def __init__(self,weightsFile='',metrics=rateNp,\
        channelList=[0],onlyLevel=-1000,up=1,mul=1,**kwags):
        #channelList=[0]
        config=fcnConfig('surfUp',up=up,mul=mul,**kwags)
        #defProcess()
        self.mul=mul
        config.inputSize[-1]=len(channelList)
        self.genM(config, onlyLevel)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        self.compile(loss=self.config.lossFunc, optimizer='Adam')
        print(self.summary())
        if len(weightsFile)>0:
            self.weight0 = self.get_weights()
            self.set_weights(self.weight0)
            self.load_weights(weightsFile,by_name= True)
        self.lossFunc=config.lossFuncNP
    def genM(self,config, onlyLevel=-1000):
        inputs, outputs = config.inAndOut(onlyLevel=onlyLevel)
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return self
    def predict(self,x,batch_size=-1,**kwargs):
        #print('inx')
        maxN = 1000
        NX = len(x)
        if batch_size==-1:
            batch_size = int(batchMax/self.mul)
        #print('inx done')s
        if self.config.mode=='surf' or self.config.mode=='surfUp':
            Y = []
            #print('inx',batch_size)
            for i in range(0,NX,maxN):
                X = self.inx(x[i:min(i+maxN,NX)])
                if len(X)>0:
                    Y.append(super().predict(X,batch_size=batch_size,**kwargs))
            Y=np.concatenate(Y,axis=0)
            return Y
        else:
            return super().predict(x,batch_size=batch_size,**kwargs).astype(np.float32)
    def fit(self,x,y,batchSize=None,**kwargs):
        x=self.inx(x)
        if np.isnan(x).sum()>0 or np.isinf(x).sum()>0:
            print('bad record')
            return None
        return super().fit(x ,y,batch_size=batchSize,**kwargs)
    def plot(self,filename='model.png'):
        plot_model(self, to_file=filename)
    def inx(self,x):
        x=x.copy()
        if self.config.mode=='surf' or self.config.mode=='surfUp' or self.config.mode=='surfUpMul':
            if x.shape[-1] > len(self.channelList):
                x = x[:,:,:,self.channelList]
            #timeN0 = np.float32(x.shape[1])
            #timeN  = (x[:,:,:,0]!=0).sum(axis=1,keepdims=True).astype(np.float32)
            #timeN *= 1+0.2*(np.random.rand(*timeN.shape).astype(np.float32)-0.5)
            x[:,:,:,0]/=(np.abs(x[:,:,:,0]).max(axis=(1,),keepdims=True))#*(timeN0/timeN)**0.5
        else:
            x/=x.std(axis=(1,2,3),keepdims=True)+np.finfo(x.dtype).eps
        return x
    #def __call__(self,x,*args,**kwargs):
    #    return super(Model, self).__call__(K.tensor(self.inx(x)))
    def train(self,x,y,**kwarg):
        if 't' in kwarg:
            t = kwarg['t']
        else:
            t = ''
        XYT = xyt(x,y,t)
        self.trainByXYT(XYT,**kwarg)
    def trainByXYT(self,XYT,N=2000,perN=200,batchSize=32,xTest='',\
        yTest='',k0 = 2e-3,t='',count0=3,resL=globalFL):
        resL.append(0)
        if k0>0:
            K.set_value(self.optimizer.lr, k0)
        indexL = range(len(XYT))
        sampleDone = np.zeros(len(XYT))
        #print(indexL)
        lossMin =100
        count   = count0
        w0 = self.get_weights()
        resStr=''
        trainTestLoss = []
        iL = random.sample(indexL,xTest.shape[0])
        xTrain, yTrain , t0LTrain = XYT(iL)
        #print(self.metrics)
        for i in range(N):
            gc.collect()
            iL = random.sample(indexL,perN)
            for ii in iL:
                sampleDone[ii]+=1
            x, y , t0L = XYT(iL)
            #print(XYT.iL)
            self.fit(x ,y,batchSize=batchSize)
            if i%10==0:
                if len(xTest)>0:
                    lossTrain = self.evaluate(self.inx(xTrain),yTrain)
                    lossTest    = self.evaluate(self.inx(xTest),yTest)
                    print('train loss',lossTrain,'test loss: ',lossTest,\
                        'sigma: ',XYT.timeDisKwarg['sigma'],\
                        'w: ',self.config.lossFunc.w, \
                        'no sampleRate:', 1 - np.sign(sampleDone).mean(),\
                        'sampleTimes',sampleDone.mean(),'last F:',resL[-1],'min Loss:',lossMin,'count:',count)
                    resStr+='\n %d train loss : %f valid loss :%f F: %f'%(i,lossTrain,lossTest,resL[-1])
                    #lossTest-=resL[-1]
                    trainTestLoss.append([i,lossTrain,lossTest])
                    if i%60==0 and i>10:
                        youtTrain = 0
                        youtTest  = 0
                        youtTrain = self.predict(xTrain)
                        youtTest  = self.predict(xTest)
                        for level in range(youtTrain.shape[-2]):
                            print('level',len(self.config.featureL)\
                                -youtTrain.shape[-2]+level+1)
                            resStr +='\nlevel: %d'%(len(self.config.featureL)\
                                -youtTrain.shape[-2]+level+1)
                            resStr+='\ntrain '+printRes_old(yTrain, youtTrain[:,:,level:level+1])
                            resStr+='\ntest '+printRes_old(yTest, youtTest[:,:,level:level+1],isUpdate=True)
                    if lossTest >= lossMin:
                        count -= 1
                    if lossTest > 3*lossMin and lossMin>0:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if lossTest < lossMin:
                        count = count0
                        lossMin = lossTest
                        w0 = self.get_weights()
                        print('find better')
                    if count <=0:
                        break
                    #print(self.metrics)
                    
            if i%15==0:
                print('learning rate: ',self.optimizer.lr)
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.95)
            if i>10 and i%50==0:
                perN += int(perN*0.05)
                perN = min(500, perN)
        self.set_weights(w0)
        return resStr,trainTestLoss
    
    def trainByXYTNew(self,XYT,N=2000,perN=200,batchSize=32,xTest='',\
        yTest='',k0 = 2e-3,t='',count0=20,resL=globalFL):
        resL.append(0)
        if k0>0:
            K.set_value(self.optimizer.lr, k0)
        indexL = range(len(XYT))
        sampleDone = np.zeros(len(XYT))
        #print(indexL)
        lossMin =100
        count   = count0
        w0 = self.get_weights()
        resStr=''
        trainTestLoss = []
        iL = random.sample(indexL,xTest.shape[0])
        xTrain, yTrain , t0LTrain = XYT(iL)
        #print(self.metrics)
        sampleTime=0
        for i in range(N):
            gc.collect()
            iL = random.sample(indexL,perN)
            for ii in iL:
                sampleDone[ii]+=1
            x, y , t0L = XYT(iL)
            print('loop:',sampleDone.mean())
            self.fit(x ,y,batchSize=batchSize)
            if int(sampleDone.mean())>sampleTime:
                sampleTime = int(sampleDone.mean())
                if len(xTest)>0:
                    lossTrain = self.evaluate(self.inx(xTrain),yTrain,batch_size=batchSize)
                    lossTest    = self.evaluate(self.inx(xTest),yTest,batch_size=batchSize)
                    print('train loss',lossTrain,'test loss: ',lossTest,\
                        'sigma: ',XYT.timeDisKwarg['sigma'],\
                        'w: ',self.config.lossFunc.w, \
                        'no sampleRate:', 1 - np.sign(sampleDone).mean(),\
                        'sampleTimes',sampleDone.mean(),'last F:',resL[-1],'min Loss:',lossMin,'count:',count)
                    resStr+='\n %d train loss : %f valid loss :%f F: %f sampleTime: %d'%(i,lossTrain,lossTest,resL[-1],sampleTime)
                    #lossTest-=resL[-1]
                    trainTestLoss.append([i,lossTrain,lossTest])
                    if True:#i%90==0 and i>10:
                        youtTrain = 0
                        youtTest  = 0
                        youtTrain = self.predict(xTrain,batch_size=batchSize)
                        youtTest  = self.predict(xTest,batch_size=batchSize)
                        for level in range(youtTrain.shape[-2]):
                            print('level',len(self.config.featureL)\
                                -youtTrain.shape[-2]+level+1)
                            resStr +='\nlevel: %d'%(len(self.config.featureL)\
                                -youtTrain.shape[-2]+level+1)
                            resStr+='\ntrain '+printRes_old(yTrain, youtTrain[:,:,level:level+1])
                            resStr+='\ntest '+printRes_old(yTest, youtTest[:,:,level:level+1],isUpdate=True)
                    if lossTest >= lossMin:
                        count -= 1
                    if lossTest > 3*lossMin and lossMin>0:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if lossTest < lossMin:
                        count = count0
                        lossMin = lossTest
                        w0 = self.get_weights()
                        print('find better')
                    if count <=0:
                        break
                    #print(self.metrics)
                    if True:#i%15==0:
                        K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.75)
                        print('learning rate: ',self.optimizer.lr)
                    if True:#i>10 and i%50==0:
                        batchSize += batchSize
                        batchSize = min(256, perN)
                        print('batchSize ',batchSize)
        self.set_weights(w0)
        return resStr,trainTestLoss
    def trainByXYTNewMul(self,XYT,N=2000,perN=200,batchSize=32,xTest='',\
        yTest='',k0 = 2e-3,t='',count0=20,resL=globalFL,mul=1,testN=-1):
        resL.append(0)
        if k0>0:
            K.set_value(self.optimizer.lr, k0)
        indexL = list(range(len(XYT)))
        indexLMul = indexL*100
        sampleDone = np.zeros(len(XYT))
        #print(indexL)
        lossMin =100
        count   = count0
        w0 = self.get_weights()
        resStr=''
        trainTestLoss = []
        iL = random.sample(indexL,int(testN/3))
        xTrain, yTrain , t0LTrain = XYT(iL,mul=mul)
        print('training',xTrain.shape,len(iL))
        sampleTime=-1
        r = len(XYT)/len(XYT.corrL)
        batchSize = int(batchMax/self.mul)
        perM=int(perN/mul)
        perM= int(perM/batchSize)*batchSize
        for i in range(N):
            gc.collect()
            iL = random.sample(indexLMul,perM)
            for ii in iL:
                sampleDone[ii]+=r*mul
            x, y , t0L = XYT(iL,mul=mul,N=mul)
            print('fromT:',np.array(t0L).mean())
            print('loop:',sampleDone.mean())
            self.fit(x ,y,batchSize=batchSize,)
            x=0
            y=0
            #print(self.layers[47].variables[-2][-10:],self.layers[47].variables[-1][-10:])
            if int(sampleDone.mean())>sampleTime:
                sampleTime = int(sampleDone.mean())
                if len(xTest)>0:
                    #xTrain=x
                    #yTrain=y
                    #xTrain = x
                    #yTrain = y
                    youtTrain = 0
                    youtTest  = 0
                    youtTrain = self.predict(xTrain,batch_size=batchSize)
                    #youtTrain = self.predict(x,batch_size=batchSize)
                    youtTest  = self.predict(xTest,batch_size=batchSize)
                    #print(youtTest.mean())
                    lossTrain = self.lossFunc(yTrain,youtTrain)
                    lossTest    = self.lossFunc(yTest,youtTest)
                    #lossTrain = self.evaluate(self.inx(xTrain),yTrain)
                    #lossTest    = self.evaluate(self.inx(xTest),yTest)
                    #print(xTrain.shape,yTrain.shape)
                    print('train loss',lossTrain,'test loss: ',lossTest,\
                        'no sampleRate:', 1 - np.sign(sampleDone).mean(),\
                        'sampleTimes',sampleDone.mean(),'last F:',resL[-1],'min Loss:',lossMin,'lr',K.get_value(self.optimizer.lr),'count:',count)
                    resStr+='\n %d train loss : %f valid loss :%f F: %f sampleTime: %d'%(i,lossTrain,lossTest,resL[-1],sampleTime)
                    #lossTest-=resL[-1]
                    trainTestLoss.append([i,lossTrain,lossTest])
                    resStr+='\ntrain '+printRes_old(yTrain,youtTrain)
                    resStr+='\ntest '+printRes_old(yTest, youtTest,isUpdate=True)
                    youtTrain = 0
                    youtTest  = 0
                    if lossTest >= lossMin:
                        count -= 1
                    if lossTest > 3*lossMin and lossMin>0:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if lossTest < lossMin:
                        count = count0
                        lossMin = lossTest
                        w0 = self.get_weights()
                        print('find better')
                    if count <=0:
                        break
                    #print(self.metrics)
                    if True:#i%15==0:
                        K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.9)
                        print('learning rate: ',self.optimizer.lr)
                    if True:#i>10 and i%50==0:
                        batchSize = int(batchMax/self.mul)
                        print('batchSize ',batchSize)
            gc.collect()
        self.set_weights(w0)
        return resStr,trainTestLoss
    def trainByXYTCross(self,self1,XYT0,XYT1,N=2000,perN=100,batchSize=None,\
        xTest='',yTest='',k0 = -1,t='',per1=0.5):
        #XYT0 syn
        #XYT1 real
        if k0>1:
            K.set_value(self.optimizer.lr, k0)
        indexL0 = range(len(XYT0))
        indexL1 = range(len(XYT1))
        #print(indexL)
        lossMin =100
        count0  = 10
        count   = count0
        w0 = self.get_weights()

        #print(self.metrics)
        for i in range(N):
            is0 =False
            if (i < 10) or (np.random.rand()<per1  and i <20):
                print('1')
                is0 =False
                XYT = XYT1
                iL = random.sample(indexL1,perN)
                SELF = self1
            else:
                print('0')
                is0 = True
                XYT = XYT0
                iL = random.sample(indexL0,perN)
                SELF = self
            x, y , t0L = XYT(iL)   
            #print(XYT.iL)
            SELF.fit(x ,y ,batchSize=batchSize)
            if  is0:
                self1.set_weights(self.get_weights())
            else:
                self.set_weights(self1.get_weights())
            if i%3==0 and (is0 or per1>=1):
                if len(xTest)>0:
                    loss    = self.evaluate(self.inx(xTest),yTest)
                    lossM = self.Metrics(yTest,self.predict(xTest))
                    if loss >= lossMin:
                        count -= 1
                    if loss > 3*lossMin:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if loss < lossMin:
                        count = count0
                        lossMin = loss
                        w0 = self.get_weights()
                    if count ==0:
                        break
                    #print(self.metrics)
                   
                    print('test loss: ',loss,' metrics: ',lossM,'sigma: ',\
                        XYT.timeDisKwarg['sigma'],'w: ',self.config.lossFunc.w)
            if i%5==0:
                print('learning rate: ',self.optimizer.lr)
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.9)
                K.set_value(self1.optimizer.lr, K.get_value(self1.optimizer.lr) * 0.9)
            if i==50:
                perN = 100
            if i>10 and i%5==0:
                perN += int(perN*0.02)
        self.set_weights(w0)
    def show_(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(19),fileStr='',\
        level=-1):
        y = self.predict(x)
        f = 1/T
        count = x.shape[1]
        x = x.transpose([0,2,1,3]).reshape([-1,x.shape[1],1,x.shape[-1]])
        y = y.transpose([0,2,1,3]).reshape([-1,y.shape[1],1,y.shape[-1]])
        time0L=time0L.reshape([-1])
        for i in range(len(x)):
            up = round(y.shape[1]/x.shape[1])
            #print('show',i)
            timeL = np.arange(count)*delta
            timeLOut = np.arange(count*up)*delta/up
            if len(time0L)>0:
                timeL+=time0L[i]
                timeLOut+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            xlim=[0,500]
            xlimNew=[0,500]
            #xlim=xlimNew
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            timeLOutL0=timeLOut[pos0.astype(np.int)]
            timeLOutL0[tmpy0.max(axis=0)<0.5]=np.nan
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0).astype(np.float)
            timeLOutL=timeLOut[pos.astype(np.int)]
            timeLOutL[tmpy.max(axis=0)<0.5]=np.nan
            #pos[tmpy.max(axis=0)<0.5]=np.nan
            plt.close()
            plt.figure(figsize=[12,8])
            plt.subplot(4,1,1)
            plt.title('%s%d'%(outputDir,i))
            legend = ['r s','i s',\
            'r h','i h']
            for j in range(x.shape[-1]):
                plt.plot(timeL,self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,-1,0]-j,'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            #plt.legend()
            plt.xlim(xlim)
            plt.subplot(4,1,2)
            #plt.clim(0,1)
            plt.pcolormesh(timeLOut,f,y0[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True)
            plt.plot(timeLOutL,f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,3)
            plt.pcolormesh(timeLOut,f,y[i,:,level,:].transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True)
            #plt.clim(0,1)
            plt.plot(timeLOutL0,f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,4)
            delta = timeL[1] -timeL[0]
            N = len(timeL)
            fL = np.arange(N)/N*1/delta
            for j in range(x.shape[-1]*0+1):
                spec=np.abs(np.fft.fft(self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0])).reshape([-1])
                plt.plot(fL,spec/(spec.max()+1e-16),'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            plt.xlabel('f/Hz')
            plt.ylabel('A')
            plt.xlim([fL[1],fL[-1]/2])
            plt.ylim([-3,3])
            #plt.gca().semilogx()
            plt.savefig('%s%s_%d_%d.eps'%(outputDir,fileStr,level,i),dpi=200)
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(19),fileStr='',\
        level=-1,number=4):
        y = self.predict(x)
        f = 1/T
        dirName = os.path.dirname(outputDir)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        x = x.transpose([0,2,1,3]).reshape([-1,x.shape[1],1,x.shape[-1]])
        y = y.transpose([0,2,1,3]).reshape([-1,y.shape[1],1,y.shape[-1]])
        y0 = y0.transpose([0,2,1,3]).reshape([-1,y0.shape[1],1,y0.shape[-1]])
        time0L=time0L.reshape([-1])
        count = x.shape[1]
        for i in range(len(x)):
            up = round(y.shape[1]/x.shape[1])
            #print('show',i)
            timeL = np.arange(count)*delta
            timeLOut = np.arange(count*up)*delta/up
            if len(time0L)>0:
                timeL+=time0L[i]
                timeLOut+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            xlim=[0,500]
            xlimNew=[0,500]
            #xlim=xlimNew
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            timeLOutL0=timeLOut[pos0.astype(np.int)]
            timeLOutL0[tmpy0.max(axis=0)<0.5]=np.nan
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0).astype(np.float)
            timeLOutL=timeLOut[pos.astype(np.int)]
            timeLOutL[tmpy.max(axis=0)<0.5]=np.nan
            #pos[tmpy.max(axis=0)<0.5]=np.nan
            plt.close()
            if number == 4:
                plt.figure(figsize=[6,4])
            else:
                plt.figure(figsize=[6,3])
            plt.subplot(number,1,1)
            plt.title('%s%d'%(outputDir,i))
            legend = ['r s','i s',\
            'r h','i h']
            for j in range(x.shape[-1]*0+2):
                plt.plot(timeL,self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,-1,0],('rkbg'[j]),\
                    label=legend[j],linewidth=0.5)
            #plt.legend()
            plt.xlim(xlim)
            ax = plt.gca()
            plt.xticks([])
            plt.subplot(number,1,2)
            #plt.clim(0,1)
            pc=plt.pcolormesh(timeLOut,f,y0[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True)
            #figureSet.setColorbar(pc,label='Probility',pos='right')
            if number==4:
                plt.plot(timeLOutL,f,'--k',linewidth=0.5)
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            if number==4:
                plt.xticks([])
            #plt.colorbar(label='Probility')
            if number==4:
                plt.subplot(number,1,3)
                pc=plt.pcolormesh(timeLOut,f,y[i,:,level,:].transpose(),cmap='bwr',vmin=0,vmax=1,rasterized=True)
                #plt.clim(0,1)
                plt.plot(timeLOutL0,f,'--k',linewidth=0.5)
                plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(number,1,number)
            plt.axis('off')
            #cax=figureSet.getCAX(pos='right')
            figureSet.setColorbar(pc,label='Probability',pos='Surf')
            #plt.colorbar(label='Probility')
            #plt.gca().semilogx()
            plt.savefig('%s%s_%d_%d.eps'%(outputDir,fileStr,level,i),dpi=200)
    def predictRaw(self,x):
        yShape = list(x.shape)
        yShape[-1] = self.config.outputSize[-1]
        y = np.zeros(yShape)
        d = self.config.outputSize[0]
        halfD = int(self.config.outputSize[0]/2)
        iL = list(range(0,x.shape[0]-d,halfD))
        iL.append(x.shape[0]-d)
        for i0 in iL:
            y[:,i0:(i0+d)] = x.predict(x[:,i0:(i0+d)])
        return y
    def set(self,modelOld):
        self.set_weights(modelOld.get_weights())
    def setTrain(self,name,trainable=True):
        lr0= K.get_value(self.optimizer.lr)
        for layer in self.layers:
            if layer.name.split('_')[0] in name:
                layer.trainable = trainable
                print('set',layer.name,trainable)
            else:
                layer.trainable = not trainable
                print('set',layer.name,not trainable)

        self.compile(loss=self.config.lossFunc, optimizer='Nadam')
        K.set_value(self.optimizer.lr,  lr0)
