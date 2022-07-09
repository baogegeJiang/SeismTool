from posixpath import dirname
from re import X

from torch import dtype, where
from tensorflow import keras
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,\
  Dropout, Dense,Softmax,Conv1D,Reshape,DenseFeatures,LayerNormalization,UpSampling2D,UpSampling1D,Multiply,SeparableConv2D,SeparableConv1D,DepthwiseConv2D,Permute,Flatten
from tensorflow.keras.layers import BatchNormalization
from.layer import MBatchNormalization
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.keras import initializers, regularizers, constraints, activations
#LayerNormalization = keras.layers.BatchNormalization
import numpy as np
#from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow import transpose
from matplotlib import pyplot as plt   
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
import random
import obspy
import time
from tensorflow.python.keras.layers.merge import dot

from tensorflow.python.ops.math_ops import tensordot
from .. import mathTool
from ..mathTool import mathFunc
from ..mathTool.mathFunc import findPos
import mathTool
import os
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from .node_cell import ODELSTMR
import gc
from ..plotTool import figureSet
from numba import jit
import numexpr as ne
from tensorflow.keras import mixed_precision
from tensorflow import signal as TS
import tensorflow as tf
from tensorflow.python.keras import initializers
def complex_initializer(base_initializer):
    f = base_initializer()

    def initializer(*args, dtype=tf.complex64, **kwargs):
        real = f(*args, **kwargs)
        imag = f(*args, **kwargs)
        return tf.complex(real, imag)

    return initializer
#complex_initializer(tf.random_normal_initializer)
class FFTConv2D(tf.keras.layers.Layer):
    def __init__(self,depth_multiplier=1,axis=1,initializer='glorot_uniform',**kwargs):
        super(FFTConv2D,self).__init__(**kwargs)
        self.depth_multiplier = depth_multiplier
        self.axis=axis
        self.initializer = initializers.get(initializer)
    def build(self,input_shape):
        w_lenth=int(input_shape[self.axis] / 2 + 1)
        shape=[1,1,1,1]
        CN = input_shape[-1]*self.depth_multiplier
        shape[self.axis]=CN
        shape[-1]=w_lenth
        shape[-1]=input_shape[self.axis]
        self.kernel = self.add_weight(
        shape=shape,
        initializer=self.initializer,
        name='depthwise_kernel')
        '''
        self.kernelR = self.add_weight(
        shape=shape,
        initializer=self.initializer,
        name='depthwise_kernelR')
        #self.kernelI = self.add_weight(
        shape=shape,
        initializer=self.initializer,
        name='depthwise_kernelI')
        '''
    def call(self,inputs,training=None):
        iL=[0,1,2,3]
        iL[self.axis]=3
        iL[-1]=self.axis
        inputs = tf.transpose(inputs,iL)
        #spec =  TS.rfft(inputs)
        #spec =  TS.rfft(inputs)
        spec =  TS.rfft(inputs)*TS.rfft(self.kernel)#spec*(self.kernelR+1j*self.kernelI)
        output = TS.irfft(spec)
        return tf.transpose(output,iL)
    def get_config(self):
        config = super(FFTConv2D, self).get_config()
        config['depth_multiplier'] = self.depth_multiplier
        config['initializer'] = initializers.serialize(self.initializer)
        config['axis'] = self.axis
        return config

class FFTConv2DRI(tf.keras.layers.Layer):
    def __init__(self,depth_multiplier=1,axis=1,initializer='glorot_uniform',**kwargs):
        super(FFTConv2DRI,self).__init__(**kwargs)
        self.depth_multiplier = depth_multiplier
        self.axis=1
        self.initializer =initializers.get(initializer)   
    def build(self,input_shape):
        w_lenth=int(input_shape[self.axis] / 2 + 1)
        shape=[1,1,1,1,1]
        #CN = input_shape[-1]*self.depth_multiplier
        shape[self.axis]=input_shape[-1]
        shape[self.axis+1]=self.depth_multiplier
        shape[-1]=w_lenth
        self.kernelR = self.add_weight(
        shape=shape,
        initializer=self.initializer,
        name='depthwise_kernelR')
        self.kernelI = self.add_weight(
        shape=shape,
        initializer=self.initializer,
        name='depthwise_kernelI')
    def call(self,inputs,training=None):
        iL=[0,1,2,3]
        iL[self.axis]=3
        iL[-1]=self.axis
        inputs = tf.transpose(inputs,iL)
        #spec =  TS.rfft(inputs)
        #spec =  TS.rfft(inputs)
        spec = TS.rfft(inputs)
        spec =  tf.reshape(spec,[-1,spec.shape[1],1,spec.shape[2],spec.shape[3]])*tf.complex(self.kernelR, self.kernelI) #spec*(self.kernelR+1j*self.kernelI)
        output = TS.irfft(spec)
        #print(output.shape)
        return tf.transpose(tf.reshape(output,[-1,output.shape[1]*output.shape[2],output.shape[3],output.shape[-1]]),iL)
    def get_config(self):
        config = super(FFTConv2DRI, self).get_config()
        config['depth_multiplier'] = self.depth_multiplier
        config['initializer'] = initializers.serialize(self.initializer)
        config['axis'] = self.axis
        return config

class FFTConv2DFS(tf.keras.layers.Layer):
    def __init__(self,depth_multiplier=1,axis=1,initializer='glorot_uniform',**kwargs):
        super(FFTConv2DFS,self).__init__(**kwargs)
        self.depth_multiplier = depth_multiplier
        self.axis=1
        self.initializer =initializers.get(initializer)   
    def build(self,input_shape):
        w_length=int(input_shape[self.axis] / 2 + 1)
        shape=[1,1,1,1,1]
        #CN = input_shape[-1]*self.depth_multiplier
        shape[self.axis]=1
        shape[self.axis+1]=1
        shape[-1]=w_length
        shapeP=[1,1,1,1,1]
        #CN = input_shape[-1]*self.depth_multiplier
        shapeP[self.axis]=input_shape[-1]
        shapeP[self.axis+1]=self.depth_multiplier
        shapeP[-1]=1
        mul=2000
        self.FR = self.add_weight(
        shape=shapeP,
        initializer= tf.keras.initializers.RandomUniform(0,0.1*mul),#RandomUniform(0, 0.5),
        constraint=tf.keras.constraints.NonNeg(),
        name='depthwise_FR')
        self.SR = self.add_weight(
        shape=shapeP,
        initializer=tf.keras.initializers.RandomUniform(0,0.1*mul),
        constraint=tf.keras.constraints.NonNeg(),
        name='depthwise_SR')
        self.freqL = self.add_weight(trainable=False,shape=shape,name='freqL') #
        #self.freqL. #
        K.set_value(self.freqL,(np.arange(w_length)/w_length*0.5).reshape(shape)*mul)
    def call(self,inputs,training=None):
        iL=[0,1,2,3]
        iL[self.axis]=3
        iL[-1]=self.axis
        inputs = tf.transpose(inputs,iL)
        #spec =  TS.rfft(inputs)
        #spec =  TS.rfft(inputs)
        spec = TS.rfft(inputs)
        spec =  tf.reshape(spec,[-1,spec.shape[1],1,spec.shape[2],spec.shape[3]])*tf.complex(K.exp(-(self.freqL-self.FR)**2/(self.SR**2+1e-7)),0. ) #spec*(self.kernelR+1j*self.kernelI)
        output = TS.irfft(spec)
        #print(output.shape)
        return tf.transpose(tf.reshape(output,[-1,output.shape[1]*output.shape[2],output.shape[3],output.shape[-1]]),iL)
    def get_config(self):
        config = super(FFTConv2DFS, self).get_config()
        config['depth_multiplier'] = self.depth_multiplier
        config['initializer'] = initializers.serialize(self.initializer)
        config['axis'] = self.axis
        return config

class FFTConv2DFS_(tf.keras.layers.Layer):
    def __init__(self,depth_multiplier=1,axis=1,initializer='glorot_uniform',**kwargs):
        super(FFTConv2DFS,self).__init__(**kwargs)
        self.depth_multiplier = depth_multiplier
        self.axis=1
        self.initializer =initializers.get(initializer)   
    def build(self,input_shape):
        w_length=int(input_shape[self.axis] / 2 + 1)
        shape=[1,1,1,1,1]
        #CN = input_shape[-1]*self.depth_multiplier
        shape[self.axis]=1
        shape[self.axis+1]=1
        shape[-1]=w_length
        shapeP=[1,1,1,1,1]
        #CN = input_shape[-1]*self.depth_multiplier
        shapeP[self.axis]=input_shape[-1]
        shapeP[self.axis+1]=self.depth_multiplier
        shapeP[-1]=1
        self.FR = self.add_weight(
        shape=shapeP,
        initializer= tf.keras.initializers.RandomUniform(0,0.5),#RandomUniform(0, 0.5),
        constraint=tf.keras.constraints.NonNeg(),
        name='depthwise_FR')
        self.SR = self.add_weight(
        shape=shapeP,
        initializer=tf.keras.initializers.RandomUniform(0,0.5),
        constraint=tf.keras.constraints.NonNeg(),
        name='depthwise_SR')
        self.FI = self.add_weight(
        shape=shapeP,
        initializer= tf.keras.initializers.RandomUniform(0,0.5),#RandomUniform(0, 0.5),
        constraint=tf.keras.constraints.NonNeg(),
        name='depthwise_FI')
        self.SI = self.add_weight(
        shape=shapeP,
        initializer=tf.keras.initializers.RandomUniform(0,0.5),
        constraint=tf.keras.constraints.NonNeg(),
        name='depthwise_SI')
        self.freqL = self.add_weight(trainable=False,shape=shape,name='freqL') #
        #self.freqL. #
        K.set_value(self.freqL,(np.arange(w_length)/w_length*0.5).reshape(shape))
    def call(self,inputs,training=None):
        iL=[0,1,2,3]
        iL[self.axis]=3
        iL[-1]=self.axis
        inputs = tf.transpose(inputs,iL)
        #spec =  TS.rfft(inputs)
        #spec =  TS.rfft(inputs)
        spec = TS.rfft(inputs)
        spec =  tf.reshape(spec,[-1,spec.shape[1],1,spec.shape[2],spec.shape[3]])*tf.complex(K.exp(-(self.freqL-self.FR)**2/(self.SR**2+1e-7)), K.exp(-(self.freqL-self.FI)**2/(self.SI**2+1e-7))) #spec*(self.kernelR+1j*self.kernelI)
        output = TS.irfft(spec)
        #print(output.shape)
        return tf.transpose(tf.reshape(output,[-1,output.shape[1]*output.shape[2],output.shape[3],output.shape[-1]]),iL)
    def get_config(self):
        config = super(FFTConv2DFS, self).get_config()
        config['depth_multiplier'] = self.depth_multiplier
        config['initializer'] = initializers.serialize(self.initializer)
        config['axis'] = self.axis
        return config




K.set_floatx('float32')
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
# this module is to construct deep learning network
    
def defProcess():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
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
def swish(inputs):
    return (K.sigmoid(inputs) * inputs)
class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'
get_custom_objects().update({'swish': Swish(swish)})
#传统的方式
class lossFuncSoft_:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1):
        self.w=w
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        y1 = 1-y0
        yout1 = 1-yout0
        yout0 = K.clip(yout0,1e-7,1)
        yout1 = K.clip(yout1,1e-7,1)
        return -K.mean(\
                         (\
                             self.w*y0*K.log(yout0)+y1*K.log(yout1)\
                                                                     )*\
                         K.max(y0,axis=1, keepdims=True),\
                         axis=-1\
                                                                         )
class lossFuncSoft__:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1):
        w0 = np.ones(50)
        w0[:20]=5/(1+4*np.arange(0,20)/19)
        w0[20:]=10/(1+9*np.arange(29,-1,-1)/29)
        w0/=w0.mean()
        w0 = w0*0+1
        channelW = K.variable(w0.reshape([1,1,1,-1]))
        self.channelW = channelW
        self.w=w
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        y1 = 1-y0
        yout1 = 1-yout0
        yout0 = K.clip(yout0,1e-7,1)
        yout1 = K.clip(yout1,1e-7,1)
        return -K.mean(\
                         (\
                             y0*K.log(yout0)+y1*K.log(yout1)\
                                                                     )*\
                         K.max(y0,axis=1, keepdims=True)*((self.w-1)*(K.sign(y0-1/20)+1)/2+1)*self.channelW,\
                         axis=-1\
                                                                         )
class lossFuncSoft__:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1,**kwags):
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        y1 = 1-y0
        yout1 = 1-yout0
        yout0 = K.clip(yout0,1e-7,1)
        yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = K.max(y0,axis=1, keepdims=True)
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return -K.mean(\
                         (\
                             y0*K.log(yout0)+y1*K.log(yout1)\
                                                                     )*\
                         (maxChannel*0.975+0.025)*maxSample,\
                                                                         )
isUnlabel=1#1
W0=0.05
W0FT=0.02
def getW(T,W0):
    return W0
    if T<20:
        return W0*0.3
    elif T<30:
        return W0*0.6
    elif T<60:
        return W0
    elif T<80:
        return W0*0.8
    elif T<100:
        return W0*0.6
    elif T<120:
        return W0*0.4
    else:
        return W0*0.2
class lossFuncSoft:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        self.w = K.ones([1,maxCount,1,len(TL)])
        w = np.ones([1,maxCount,1,len(TL)])*0.05
        for i in range(len(TL)):
            i0 = int(TL[i]/disMaxR/delta*(1+randA))
            i1 = min(int(TL[i]/disRandA/delta*(1-randA)),maxCount)
            w[0,:,0,i]=getW(TL[i],W0)
            #self.w[0,:,0,i]=10/TL[i]
            w[0,:i0,0,i]=0
            w[0,i1:,0,i]=0
        K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = (K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return -K.mean(\
                         (\
                             y0*K.log( K.clip(yout0,1e-7,1))+(1-y0)*K.log(K.clip(1-yout0,1e-7,1))\
                                                                     )*\
                         (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample)),\
                                                                         )

class lossFuncMSEMul:
    def __init__(self,N=10):
        self.__name__ = 'lossFuncSoft'
        self.N=10
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        return K.mean(\
                         (\
                             (y0-yout0[:,])**2
                                                                     )
                                                                         )

class lossFuncMSE:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
        self.focusChannel=-1
        self.w = K.ones([1,maxCount,1,len(TL)])
        w = np.ones([1,maxCount,1,len(TL)])*0.05
        for i in range(len(TL)):
            i0 = int(TL[i]/disMaxR/delta*(1+randA))
            i1 = min(int(TL[i]/disRandA/delta*(1-randA)),maxCount)
            w[0,:,0,i]=getW(TL[i],W0)
            #self.w[0,:,0,i]=10/TL[i]
            w[0,:i0,0,i]=0
            w[0,i1:,0,i]=0
        K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = (K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return K.mean(\
            (
                        (\
                            (y0-yout0)**2
                                                                    )*\
                        (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample))\
                                                                        )
                                                                        )
        if self.focusChannel == -1:
            return K.mean(\
                (
                            (\
                                (y0-yout0)**2
                                                                        )*\
                            (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample))\
                                                                            )
                                                                            )
        else:
            return K.mean(\
                (
                            (\
                                (y0-yout0)**2
                                                                        )*\
                            (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample))\
                                                                            )[:,:,:,self.focusChannel:self.focusChannel+1]
                                                                            )                                     

class lossFuncMSEFT:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
        #K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = (K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2
        maxSample   = K.max(maxChannel,axis=2, keepdims=True)
        return K.mean(\
            (
                        (\
                            (y0*(K.sign(y0+0.01)+1)/2-yout0)**2
                                                                    )*\
                        (maxChannel*1+isUnlabel*W0FT*(1-maxChannel)*maxSample*(K.sign(y0+0.01)+1)/2+(1-maxSample))\
                                                                        )                                                                     )
class lossFuncMSEFTNP:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
        #K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        return callMSEFT(y0,yout0)
class lossFuncSoftFT:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
        #K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = (K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2
        maxSample   = K.max(maxChannel,axis=2, keepdims=True)
        Y0 = y0*(K.sign(y0+0.01)+1)/2
        return K.mean(\
            (
                        (\
                            -Y0*K.log(1e-7+yout0)-(1-Y0)*K.log(1e-7+1-yout0)
                                                                    )*\
                        (maxChannel*1+isUnlabel*W0FT*(1-maxChannel)*maxSample*(K.sign(y0+0.01)+1)/2+(1-maxSample))\
                                                                        )                                                                     )
class lossFuncSoftFTNP:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
        #K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        return callSoftFT(y0,yout0)
class lossFuncMSE_:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
        self.w = K.ones([1,maxCount,1,len(TL)])
        w = np.ones([1,maxCount,1,len(TL)])*0.05
        for i in range(len(TL)):
            i0 = int(TL[i]/disMaxR/delta*(1+randA))
            i1 = min(int(TL[i]/disRandA/delta*(1-randA)),maxCount)
            w[0,:,0,i]=getW(TL[i],W0)
            #self.w[0,:,0,i]=10/TL[i]
            w[0,:i0,0,i]=0
            w[0,i1:,0,i]=0
        K.set_value(self.w,w)
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        CW=yout0[1]
        yout0     =yout0[0]
        maxChannel  = (K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return K.mean(\
                (
                            (\
                                (y0-yout0)**2
                                                                        )*\
                            (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample))\
                                                                            )*CW
                                                                            )        
class lossFuncMSEO:
    def __init__(self,**kwags):
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        return K.mean((y0-yout0)**2)
class lossFuncMSENPO:
    def __init__(self,**kwags):
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        return np.mean((y0-yout0)**2)
class lossFuncMSENP:
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=2048):
        K=np
        self.__name__ = 'lossFuncSoft'
        #disRandA=1/15
        #disMaxR=1
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
        K=np
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
class lossFuncSoftNP__:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1):
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        K=np
        y1 = 1-y0
        yout1 = 1-yout0
        yout0 = K.clip(yout0,1e-7,1)
        yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = K.max(y0,axis=1, keepdims=True)
        mean = maxChannel.mean()
        #print(mean)
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return -1/mean*K.mean(\
                         (\
                             y0*K.log(yout0)+y1*K.log(yout1)\
                                                                     )*\
                         (maxChannel*0.9+0.1),\
                                                                         )
class lossFuncSoftNP:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1,randA=0.05,disRandA=1/12,disMaxR=4,TL=[],delta=1,maxCount=1536):
        K=np
        self.__name__ = 'lossFuncSoft'
        w = np.ones([1,maxCount,1,len(TL)],dtype=np.float32)*0.05
        for i in range(len(TL)):
            i0 = int(TL[i]/disMaxR/delta*(1+randA))
            i1 = min(int(TL[i]/disRandA/delta*(1-randA)),maxCount)
            w[0,:,0,i]=getW(TL[i],W0)
            #self.w[0,:,0,i]=10/TL[i]
            w[0,:i0,0,i]=0
            w[0,i1:,0,i]=0
        self.w=w
    def __call__(self,y0,yout0):
        return call(y0,yout0,self.w)
        K=np
        #y1 = 1-y0
        #yout1 = 1-yout0
        #yout0 = K.clip(yout0,1e-7,1)
        #yout1 = K.clip(yout1,1e-7,1)
        maxChannel  = ((K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2)
        maxSample   = K.max(maxChannel,axis=3, keepdims=True)
        return -K.mean(\
                         (\
                             y0*K.log(K.clip(yout0,1e-7,1))+(1-y0)*K.log(K.clip(1-yout0,1e-7,1))\
                                                                     )*\
                         (maxChannel*1+isUnlabel*self.w*(1-maxChannel)*maxSample+(1-maxSample)),\
                                                                        )


def call(y0,yout0,w,isUnlabel=isUnlabel):
    K=np
    #y1 = 1-y0
    #yout1 = 1-yout0
    #yout0 = K.clip(yout0,1e-7,1)
    #yout1 = K.clip(yout1,1e-7,1)
    maxChannel  = ((K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2)
    maxSample   = K.max(maxChannel,axis=3, keepdims=True)
    isUnlabel=np.array(isUnlabel,dtype=np.float32)
    N = y0.size
    return -ne.evaluate('( y0*log( where( yout0>1e-7, yout0,1e-7 )) + (1-y0)*log( where( (1-yout0)>1e-7, 1-yout0,1e-7 ) ) )*( maxChannel + isUnlabel*w*(1-maxChannel)*maxSample+(1-maxSample) )  ').mean()
    return -K.mean(\
                        (\
                            y0*K.log(K.clip(yout0,1e-7,1))+(1-y0)*K.log(K.clip(1-yout0,1e-7,1))\
                                                                    )*\
                        (maxChannel*1+isUnlabel*w*(1-maxChannel)*maxSample+(1-maxSample)),\
                                                                        )                                                           
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

def callSoftFT(y0,yout0,isUnlabel=isUnlabel):
    K=np
    #y1 = 1-y0
    #yout1 = 1-yout0
    #yout0 = K.clip(yout0,1e-7,1)
    #yout1 = K.clip(yout1,1e-7,1)
    maxChannel  = ((K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2)
    maxSample   = K.max(maxChannel,axis=2, keepdims=True)
    isUnlabel=np.array(isUnlabel,dtype=np.float32)
    Y0 = np.where(y0>-0.01,y0,0)
    return ne.evaluate('(-Y0*log(1e-7+yout0)-(1-Y0)*log(1e-7+1-yout0))*(maxChannel + where(y0>-0.01,1,0)*isUnlabel*W0FT*(1-maxChannel)*maxSample+(1-maxSample) )').mean()

def callMSEFT(y0,yout0,isUnlabel=isUnlabel):
    K=np
    #y1 = 1-y0
    #yout1 = 1-yout0
    #yout0 = K.clip(yout0,1e-7,1)
    #yout1 = K.clip(yout1,1e-7,1)
    maxChannel  = ((K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2)
    maxSample   = K.max(maxChannel,axis=2, keepdims=True)
    isUnlabel=np.array(isUnlabel,dtype=np.float32)
    return ne.evaluate('(where(y0>-0.01,y0,0)-yout0)**2*(maxChannel + where(y0>-0.01,1,0)*isUnlabel*W0FT*(1-maxChannel)*maxSample+(1-maxSample) )').mean()


def call_(y0,yout0,w,isUnlabel=isUnlabel):
    K=np
    #y1 = 1-y0
    #yout1 = 1-yout0
    #yout0 = K.clip(yout0,1e-7,1)
    #yout1 = K.clip(yout1,1e-7,1)
    maxChannel  = ((K.sign(K.max(y0,axis=1, keepdims=True)-0.1)+1)/2)
    maxSample   = K.max(maxChannel,axis=3, keepdims=True)
    isUnlabel=np.array(isUnlabel,dtype=np.float32)
    return -K.mean(\
                        (\
                            y0*K.log(K.clip(yout0,1e-7,1))+(1-y0)*K.log(K.clip(1-yout0,1e-7,1))\
                                                                    )*\
                        (maxChannel*1+isUnlabel*w*(1-maxChannel)*maxSample+(1-maxSample)),\
                                                                        )   

class lossFuncSoftSq:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1):
        self.w=w
        self.__name__ = 'lossFuncSoft'
    def __call__(self,y0,yout0):
        y1 = 1-y0
        print(K.max(yout0),K.min(yout0))
        yout1 = 1-yout0
        return -K.mean((self.w*y0*K.log(yout0+1e-4)+y1*K.log(yout1+1e-4)),\
            axis=-1)
class lossFuncSoftBak:
    def __init__(self,w=1):
        self.w=w
    def __call__(self,y0,yout0):
        y1 = 1-y0
        yout1 = 1-yout0
        return -K.mean(self.w*y0*K.log(yout0+1e-8)+y1*K.log(yout1+1e-8),axis=-1)

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
'''
def rightRateNp(yinPos,youtPos,yinMax,youtMax,maxD=0.03,K=np, minP=0.5):
    threshold = yinPos*maxD
    d       = K.abs(yinPos - youtPos)
    #print(d)
    #print(d.mean(axis=(0,1)))
    count   = K.sum((yinMax>0.5)*(youtMax>minP))
    hitCount= K.sum((d<threshold)*(yinMax>0.5)*(youtMax>minP))
    return hitCount/count
'''
globalFL = [0]


def printRes_old_(yin, yout,resL=globalFL,isUpdate=False):
    #       0.01  0.8  0.36600 0.9996350
    strL   = 'maxD   minP hitRate rightRate F mean  std old'
    strfmt = '\n%5.3f&%3.1f&%7.5f&%7.5f&%7.5f&%7.3f&%7.3f'
    yinPos  = yin.argmax( axis=1)
    youtPos = yout.argmax(axis=1)
    yinMax = yin.max(axis=1)
    youtMax = yout.max(axis=1)
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
def printRes_old(yin, yout,resL=globalFL,isUpdate=False,N=10,**kwags):
    #       0.01  0.8  0.36600 0.9996350
    strL   = 'maxD   minP hitRate rightRate F mean  std old'
    strfmt = '\n%5.3f&%3.1f&%7.5f&%7.5f&%7.5f&%7.3f&%7.3f'
    yinPos,yinMax  = mathFunc.Max(yin,N=N)
    youtPos,youtMax = mathFunc.Max(yout,N=N)
    for maxD in [0.03,0.015,0.01]:
        for minP in [0.5,0.6,0.8]:
            hitRate,rightRate,F,mean,std = rateNp(\
                yinPos,youtPos,yinMax,youtMax,maxD=maxD,minP=minP,**kwags)
            tmpStr=strfmt%(maxD, minP, hitRate, rightRate,F,mean*100,std*100)
            strL += tmpStr
            if maxD == 0.015 and minP==0.5 and (not np.isnan(F)) and isUpdate:
                resL.append(F)
        strL+='\n'+('-'*len(tmpStr))
    print(strL)
    return strL

def printRes_sq(yin, yout):
    strL   = 'maxD   minP hitRate rightRate F old'
    strfmt = '\n%5.3f %3.1f %7.5f %7.5f %7.5f'
    yinPos  = yin.argmax( axis=1)
    youtPos = yout.argmax(axis=1)
    yinMax = yin.max(axis=1)
    youtMax = yout.max(axis=1)
    for maxD in [0.03,0.02,0.01,0.005]:
        for minP in [0.5,0.7,0.8,0.9]:
            hitRate,rightRate,F = rateNp(\
                yinPos,youtPos,yinMax,youtMax,maxD=maxD,minP=minP)
            strL += strfmt%(maxD, minP, hitRate, rightRate,F)
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
def inAndOutFuncNewNet(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)

        convL[i] =last

        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'1',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii],name='Dropout'+layerStr+'0')(last)
        else:
            last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'1')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)

        last = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)

    convL[depth] =last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        
        for j in range(i,i+1):

            layerStr='_%d_%d'%(i,j)

            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(convL[j+1])

            if j in config.dropOutL:
                jj   = config.dropOutL.index(j)
                dConvL[j] =  Dropout(config.dropOutRateL[jj],name='Dropout_'+layerStr+'0')(dConvL[j])
            else:
                dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'0')(dConvL[j])

            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j])
            dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'1')(dConvL[j])
            dConvL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                #outputsL.append(Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
                #padding='same',activation='sigmoid',name='dconv_out_%d'%i)(convL[0]))
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_%d'%i)(convL[0]))
        
    #outputs = Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
    #    padding='same',activation='sigmoid',name='dconv_out')(convL[0])
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs

def inAndOutFuncNewNetUp(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
    convL[depth] =last
    dConvL[depth] = convL[depth]
    outputsL =[]
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>1:
                    upL[j]= Conv2DTranspose(config.outputSize[-1]*3,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*6,kernel_size=config.kernelL[j],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUp(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            last = config.poolL[i](pool_size=config.strideL[i],\
                strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1',)(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs

def inAndOutFuncNewNetDenseUpNoPooling(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    momentum=0.999
    renorm_momentum=0.999
    renorm = True
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'1')(last,training=training)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'1',)(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,momentum=momentum,name='BN'+layerStr+'1')(last,training=training)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,name='BN_'+layerStr+'1')(dConvL[j],training=training)
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,name='BN_'+layerStr+'1'+'_Up')(upL[j],training=training)
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpNoPoolingSimple(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    #BatchNormalization=MBatchNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.999
    renorm_momentum=0.999
    renorm = True
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        last = config.poolL[i](pool_size=config.strideL[i],\
                strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1',)(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last,training=training)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimple(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    #BatchNormalization=MBatchNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.999
    renorm_momentum=0.999
    renorm = True
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1',)(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last,training=training)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleM(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        convL[i] =last
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1',)(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last,training=training)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*3,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*3,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparable1D(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    inputs = Reshape([config.inputSize[0],config.inputSize[2]])(inputs)
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv1D(config.featureL[i],kernel_size=config.kernelL[i][0],\
                strides=(1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv1D(config.featureL[i],kernel_size=config.kernelL[i][0],\
                strides=(1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        convL[i] =last
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= SeparableConv1D(config.featureL[j],kernel_size=config.kernelL[j][0],\
                strides=(1),padding='same',name=name+layerStr+'_conv_0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[j+1])
            dConvL[j]  = UpSampling1D(size=config.strideL[j][0],interpolation="lanczos3")(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    upL[j]= SeparableConv1D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1][0],strides=config.strideL[-1][0],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(SeparableConv1D(config.outputSize[-1],kernel_size=config.kernelL[-1][0],strides=config.strideL[-1][0],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',depth_multiplier=1)(upL[j]))
    outputs = Reshape(config.outputSize)(outputsL[-1])
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUp(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= DepthwiseConv2D(config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_conv_0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j],depth_multiplier=config.strideL[j][0])(dConvL[j+1])
            dConvL[j]  = Reshape((dConvL[j].shape[1],dConvL[j].shape[2],int(dConvL[j].shape[3]/config.strideL[j][0]),config.strideL[j][0]))(dConvL[j])
            dConvL[j]  = Permute((1,2,4,3))(dConvL[j])
            dConvL[j]  = Reshape((dConvL[j].shape[1]*config.strideL[j][0],dConvL[j].shape[2],dConvL[j].shape[4]))(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            dConvL[j] = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'1')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
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
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpReshape(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j] = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(dConvL[j+1])
            dConvL[j]= DepthwiseConv2D(config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_conv_0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j],depth_multiplier=config.strideL[j][0])(dConvL[j])
            dConvL[j]  = Reshape((dConvL[j].shape[1],dConvL[j].shape[2],int(dConvL[j].shape[3]/config.strideL[j][0]),config.strideL[j][0]))(dConvL[j])
            dConvL[j]  = Permute((1,2,4,3))(dConvL[j])
            dConvL[j]  = Reshape((dConvL[j].shape[1]*config.strideL[j][0],dConvL[j].shape[2],dConvL[j].shape[4]))(dConvL[j])
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
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV2(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
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
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
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
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
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
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            #last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
            wave = DepthwiseConv2D(kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'_wave_0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last[:,:,:,:1])
            last = K.concatenate([wave,last[:,:,:,1:]],axis=-1)
            last = Conv2D(config.featureL[i],kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
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
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
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
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            wave = last[:,:,:,:1]
            dist = last[:,:,:,1:]
            #last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
            last = K.concatenate([DepthwiseConv2D(kernel_size=(config.kernelFirstL[j],1),strides=(1,1),padding='same',name=name+layerStr+'_wave_0_%d'%j,kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(wave) for j in range(config.outputSize[-1])]+[dist],axis=-1)
            last = Conv2D(config.featureL[i],kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i])(last)
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
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT_norm_(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    last    = inputs
    CA = -1
    TA = 1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
        TA=2
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
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
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            if config.data_format=='channels_first':
                wave = last[:,:1,:,:]
                dist = last[:,1:,:,:]
            else:
                wave = last[:,:,:,:1]
                dist = last[:,:,:,1:]
            #last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
            wave = K.concatenate([DepthwiseConv2D(kernel_size=(config.kernelFirstL[j],1),strides=(1,1),padding='same',name=name+layerStr+'_wave_0_%d'%j,kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(wave) for j in range(config.outputSize[-1])],axis=CA)
            wave = wave/(K.std(wave,axis=TA,keepdims=True)+1e-8)
            last = K.concatenate([wave]+[dist],axis=CA)
            last = Conv2D(config.featureL[i],kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=CA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
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
                dConvL[j] = BatchNormalization(axis=CA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=CA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                outputsL.append(SeparableConv2D(config.outputSize[-1],kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',depth_multiplier=1)(dConvL[0]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=TA+1,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=CA)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    if config.data_format=='channels_first':
        outputs = Permute([2,3,1])(outputs)
    return inputs,outputs

def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT_norm(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    last    = inputs
    CA = -1
    TA=1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
        TA =2
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
            wave = K.concatenate([DepthwiseConv2D(kernel_size=(config.kernelFirstL[j],1),strides=(1,1),padding='same',name=name+layerStr+'_wave_0_%d'%j,kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(wave) for j in range(config.outputSize[-1])],axis=CA)
            #wave = wave/(K.max(K.abs(wave),axis=TA,keepdims=True)+1e-8)
            wave = wave/(K.std((wave),axis=TA,keepdims=True)+1e-8)
            last = K.concatenate([wave]+[dist],axis=CA)
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
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3LK_norm(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    inputs
    last    = inputs
    CA = -1
    TA=1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
        TA =2
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
            wave = DepthwiseConv2D(kernel_size=(config.kernelFirstL[-1],1),strides=(1,1),padding='same',name=name+layerStr+'_wave_0_%d'%config.outputSize[-1],kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=config.outputSize[-1])(wave)
            #dist = dist+0
            dist = DepthwiseConv2D(kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'_dist_0_%d'%config.outputSize[-1],kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(dist)
            #wave = wave/(K.max(K.abs(wave),axis=TA,keepdims=True)+1e-8)
            wave = wave/(K.std((wave),axis=TA,keepdims=True)+1e-8)
            last = K.concatenate([wave]+[dist],axis=CA)
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

def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3FFT_norm(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    inputs
    last    = inputs
    CA = -1
    TA=1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
        TA =2
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
            if i >0:
                last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2,data_format=config.data_format)(last)
            else:
                last = FFTConv2DRI(name=name+layerStr+'_fft_0_%d'%i,depth_multiplier=2)(last)
                last = Dense(config.featureL[i])(last)
        else:
            if config.data_format=='channels_first':
                wave = last[:,:1,:,:]
                dist = last[:,1:,:,:]
            else:
                wave = last[:,:,:,:1]
                dist = last[:,:,:,1:]
            #last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
            #wave = DepthwiseConv2D(kernel_size=(config.kernelFirstL[-1],1),strides=(1,1),padding='same',name=name+layerStr+'_wave_0_%d'%config.outputSize[-1],kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=config.outputSize[-1])(wave)
            wave = FFTConv2DFS(name=name+layerStr+'_wave_0_%d'%config.outputSize[-1],depth_multiplier=int(config.outputSize[-1]*2))(wave)
            #dist = dist+0
            #wave = wave/(K.max(K.abs(wave),axis=TA,keepdims=True)+1e-8)
            wave = wave/(K.std((wave),axis=TA,keepdims=True)+1e-8)
            last = K.concatenate([wave]+[dist],axis=CA)
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

def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT_norm_FC(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    last    = inputs
    CA = -1
    TA=1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
        TA =2
    BNA =CA
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    upL  = [None for i in range(depth+1)]
    momentum=0.95
    renorm_momentum=0.95
    renorm = False
    last = inputs
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i],depth_multiplier=2,data_format=config.data_format)(last)
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
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT_FT(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    
    K.set_image_data_format(config.data_format)
    inputs  = Input(config.inputSize,name='inputs')
    last    = inputs
    CA = -1
    TA=1
    if config.data_format=='channels_first':
        last=Permute([3,1,2])(last)
        CA=1
        TA =2
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
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i],data_format=config.data_format)(last)
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
            dConvL[j] = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                last= Conv2D(2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
    outputs = Softmax(axis=3)(last)
    return inputs,outputs[:,:,:,:1]
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV2Group(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    groups=5
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = DepthwiseConv2D(kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
            last = Conv2D(config.featureL[i],kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'_group_0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],groups=groups)(last)
        else:
            #last = DepthwiseConv2D(kernel_size=config.kernelL[i],strides=(1,1),padding='same',name=name+layerStr+'0',kernel_initializer=config.initializerL[i],bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]  = UpSampling2D(size=config.strideL[j],interpolation="bilinear")(dConvL[j+1])
            dConvL[j] = DepthwiseConv2D(kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[j])
            dConvL[j] = Conv2D(config.featureL[j],kernel_size=(1,1),strides=(1,1),padding='same',name=name+layerStr+'_group_0',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],groups=groups)(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                last =DepthwiseConv2D(kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name='con_out_0_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],depth_multiplier=2)(dConvL[0])
                outputsL.append(Conv2D(config.outputSize[-1],kernel_size=(1,1),strides=config.strideL[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',groups=groups)(last))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewD(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                dilation_rate=config.dilation_rate[i],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                dilation_rate=config.dilation_rate[i],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j] = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                dilation_rate=config.dilation_rate[j],padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(dConvL[j+1])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                outputsL.append(SeparableConv2D(config.outputSize[-1],kernel_size=config.kernelL[-1],dilation_rate=config.dilation_rate[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',depth_multiplier=1)(dConvL[0]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewD(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                dilation_rate=config.dilation_rate[i],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                dilation_rate=config.dilation_rate[i],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j] = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                dilation_rate=config.dilation_rate[j],padding='same',name=name+layerStr+'_sc_0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=1)(dConvL[j+1])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                outputsL.append(SeparableConv2D(config.outputSize[-1],kernel_size=config.kernelL[-1],dilation_rate=config.dilation_rate[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',depth_multiplier=1)(dConvL[0]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparable(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')

    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        convL[i] =last
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=(1,1),padding='same',name=name+layerStr+'_conv_0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[j+1])
                dConvL[j]  = UpSampling2D(size=config.strideL[j],interpolation="bilinear")(dConvL[j])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= SeparableConv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],depth_multiplier=1)(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(SeparableConv2D(config.outputSize[-1],kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',depth_multiplier=1)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMSeparableGroups(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2,groups=5)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50,groups=10)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        convL[i] =last
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=(1,1),padding='same',name=name+layerStr+'_conv_0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j],depth_multiplier=1,groups=10)(dConvL[j+1])
                dConvL[j]  = UpSampling2D(size=config.strideL[j],interpolation="bilinear")(dConvL[j])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = SeparableConv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j],depth_multiplier=1,groups=10)(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*3,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= SeparableConv2D(config.outputSize[-1]*3,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],depth_multiplier=1,groups=10)(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpStrideSimpleMGroups(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    momentum=0.995
    renorm_momentum=0.995
    renorm = False
    groups=10
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],groups=groups)(last)
        else:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],groups=1)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        convL[i] =last
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=(1,1),padding='same',name=name+layerStr+'_conv_0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j],groups=groups)(dConvL[j+1])
                dConvL[j]  = UpSampling2D(size=config.strideL[j],interpolation="bilinear")(dConvL[j])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j],groups=groups)(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0')(dConvL[j],training=training)
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*4,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*4,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],groups=groups)(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN_'+layerStr+'0'+'_Up')(upL[j],training=training)
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Conv2D(config.outputSize[-1],kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name='con_out_1_%d'%i,kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j],activation='sigmoid',groups=groups)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpNoPoolingAxisNorm(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        last,norm=tf.linalg.normalize(last,ord=2,axis=0,name='norm'+layerStr+'0')
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1',)(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpNoPoolingAndRelu(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1',)(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpNoPoolingGather(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            pass
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    i=depth
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if  j==0:
                upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    return inputs,outputsL[0]
def inAndOutFuncNewNetDenseUpAttention(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            last = config.poolL[i](pool_size=config.strideL[i],\
                strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1',)(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConv = dConvL[j]
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            AG = Dense(config.featureL[j], activation='relu',name='dense_AG_relu_%d'%j)(dConvL[j])
            AG = Dense(config.featureL[j], activation='sigmoid',name='dense_AG_sigmoid_%d'%j)(AG)
            dConv = Multiply()([AG,dConv])
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],dConv],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpPP(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            last = config.poolL[i](pool_size=config.strideL[i],\
                strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1',)(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    convL[i] = last
    #dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                print(j)
                dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                    strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                    kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(convL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(convL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                #dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(convL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(convL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpSmapling(config, onlyLevel=-10000):
    #BatchNormalization =LayerNormalization
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        if i <depth:
            last = config.poolL[i](pool_size=config.strideL[i],\
                strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)
        else:
            last = DenseNew2(1,name='Dense'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1',)(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'UpSampling'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            if i < depth:
                dConvL[j]= UpSampling2D(config.strideL[j],name=name+layerStr+'0')(dConvL[j+1])
            else:
                dConvL[j]= DenseNew2(config.strideL[i][0],name=name+layerStr+'0')(dConvL[j+1])
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'01',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*1,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpWithoutPooling(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*6,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpWithoutPoolingAndBN(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth-1
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-2,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*6,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewNetDenseUpWithoutPoolingGather(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    upL  = [None for i in range(depth+1)]
    for i in range(depth):
        name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last
        if config.doubleConv[i]:
            last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=config.strideL[i],padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i])(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    '''
    name = 'Dense'
    i=depth-1
    layerStr='_%d_'%i
    last = Reshape([1,1,config.featureL[i-1]*config.mul],name='Reshape'+layerStr+'0')(last)
    last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    
    name = 'DDense'
    i=depth-1
    layerStr='_%d_%d'%(i,i)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i],name='Dense'+layerStr+'0')(last)
    else:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'0')(last)
    if config.isBNL[i]:
        last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
    last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
    if config.doubleConv[i]:
        last = Dense(config.featureL[i-1]*config.mul,name='Dense'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
    last = Reshape([1,config.mul,config.featureL[i-1]],name='Reshape'+layerStr+'0')(last)
    '''
    if config.mul>1:
        name = 'Dense'
        i=depth-1
        layerStr='_%d_'%i
        last = DenseNew(config.mul,name='DenseNew'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        convL[i] =last

        last = DenseNew(1,name='DenseNew'+layerStr+'1')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        
        
        name = 'DDense'
        i=depth-1
        layerStr='_%d_%d'%(i,i)

        last = Dense(config.featureL[i-1],name='Dense'+layerStr+'0')(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,name='BN'+layerStr+'0')(last)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        if config.mul>1:
            last = concatenate([last]*config.mul,axis=2)
        last = concatenate([last,convL[i]],axis=-1)
        if config.doubleConv[i]:
            last = Dense(config.featureL[i-1],name='Dense'+layerStr+'1')(last)
            if config.isBNL[i]:
                last = BatchNormalization(axis=BNA,name='BN'+layerStr+'1')(last)
            last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)
        last = concatenate([last,convL[i]],axis=-1)
    i=depth
    dConvL[i] = last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        name = 'DCONV'
        for j in range(i,i+1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j+1])
            
            if config.isBNL[j]:
                dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            if config.doubleConv[i]:
                dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(dConvL[j])
                if config.isBNL[j]:
                    dConvL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
                dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>0:
                    if config.strideL[-1][0]>1:
                        upL[j]= Conv2DTranspose(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    else:
                        upL[j]= Conv2D(config.outputSize[-1]*2,kernel_size=config.kernelL[-1],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(dConvL[0])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.outputSize[-1]*6,kernel_size=config.kernelL[-1],strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',kernel_initializer=config.initializerL[j],bias_initializer=config.bias_initializerL[j])(upL[j])
                    if config.isBNL[-1]:
                        upL[j] = BatchNormalization(axis=BNA,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=dConvL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_1_%d'%i)(upL[j]))
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewV6(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)

        convL[i] =last

        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'1',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii],name='Dropout'+layerStr+'0')(last)
        else:
            last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'1')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)

        last = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)

    convL[depth] =last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        
        for j in range(i+1):

            layerStr='_%d_%d'%(i,j)

            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(convL[j+1])

            if j in config.dropOutL:
                jj   = config.dropOutL.index(j)
                dConvL[j] =  Dropout(config.dropOutRateL[jj],name='Dropout_'+layerStr+'0')(dConvL[j])
            else:
                dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'0')(dConvL[j])

            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j])
            dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'1')(dConvL[j])
            dConvL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                #outputsL.append(Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
                #padding='same',activation='sigmoid',name='dconv_out_%d'%i)(convL[0]))
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_%d'%i)(convL[0]))
        
    #outputs = Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
    #    padding='same',activation='sigmoid',name='dconv_out')(convL[0])
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncNewUp(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    upL  = [None for i in range(depth+1)]
    last    = inputs
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)

        convL[i] =last

        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'1',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii],name='Dropout'+layerStr+'0')(last)
        else:
            last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'1')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)

        last = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)

    convL[depth] =last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        
        for j in range(i+1):

            layerStr='_%d_%d'%(i,j)

            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(convL[j+1])

            if j in config.dropOutL:
                jj   = config.dropOutL.index(j)
                dConvL[j] =  Dropout(config.dropOutRateL[jj],name='Dropout_'+layerStr+'0')(dConvL[j])
            else:
                dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'0')(dConvL[j])

            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j])
            dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'1')(dConvL[j])
            dConvL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                if config.strideL[-1][0]>1:
                    upL[j]= Conv2DTranspose(config.featureL[j]*(len(config.featureL)-i+1),kernel_size=config.kernelL[j],strides=config.strideL[-1],padding='same',name=name+layerStr+'0'+'_Up',kernel_initializer=config.initializerL[j],\
                    bias_initializer=config.bias_initializerL[j])(convL[0])
                    upL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0'+'_Up')(upL[j])
                    upL[j]  = Conv2D(config.featureL[j]*(len(config.featureL)-i+1),kernel_size=config.kernelL[j],\
                        strides=(1,1),padding='same',name=name+layerStr+'1'+'_Up',\
                        kernel_initializer=config.initializerL[j],\
                        bias_initializer=config.bias_initializerL[j])(upL[j])
                    upL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'1'+'_Up')(upL[j])
                    upL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1'+'_Up')(upL[j])
                else: 
                    upL[j]=convL[0]
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_%d'%i)(upL[j]))
        
    #outputs = Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
    #    padding='same',activation='sigmoid',name='dconv_out')(convL[0])
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs
def inAndOutFuncODELSTMRSimple(config):
    pixel_input = tf.keras.Input(shape=(config.padSize, config.channelSize), name="pixel")
    time_input = tf.keras.Input(shape=(config.padSize,config.outSize ), name="time")
    inputs = [pixel_input,time_input]
    rnn = tf.keras.layers.RNN(ODELSTMR(units=config.size),time_major=False,return_sequences=True,go_backwards=True,)
    dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation='sigmoid'))
    output_states = rnn((pixel_input, time_input))
    y = dense_layer(output_states)
    return inputs,[y]
def inAndOutFuncMul(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    momentum=0.95
    renorm_momentum=0.95
    renorm = False
    inputs  = Input(config.inputSize,name='waveform')
    rInterval  = Input(config.inputSizeI,name='interval')
    #rInterval = Reshape(config.inputSizeI+[1])(rInterval)
    mask    = Input(config.inputSizeI[0],name='mask')
    depth   =  len(config.featureL)
    last = inputs
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        #last = AveragePooling2D(pool_size=config.strideL[i],strides=config.strideL[i],padding='same')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
    last = Reshape((last.shape[1],last.shape[3]))(last)
    y=K.mean(last,axis=1,keepdims=True)
    y2=K.mean(last*last,axis=1,keepdims=True)
    x = K.mean(rInterval,axis=1,keepdims=True)
    x2 = K.mean(rInterval*rInterval,axis=1,keepdims=True)
    xy = K.mean(rInterval*last,axis=1,keepdims=True)
    k =(xy-x*y)/(x2-x*x)
    r = (xy)/K.sqrt(x2-x*x)/K.clip(K.sqrt(y2-y*y),1e9,1e-3)
    i=-1
    last = K.concatenate([k,r,y],axis=-1)
    last = Dense(config.outputSize[-1]*6)(last)
    last=Activation(config.activationL[i],name='out'+layerStr+'0')(last)
    last = Dense(config.outputSize[-1]*3)(last)
    last=Activation(config.activationL[i],name='out'+layerStr+'1')(last)
    if False:
        last = Dense(config.outputSize[-1]*4)(last)
        last=Activation(config.activationL[i],name='out'+layerStr+'2')(last)
        last = Dense(config.outputSize[-1]*3)(last)
        last=Activation(config.activationL[i],name='out'+layerStr+'3')(last)
        last = Dense(config.outputSize[-1]*2)(last)
        last=Activation(config.activationL[i],name='out'+layerStr+'4')(last)
    output = Dense(config.outputSize[-1])(last)
    return [inputs,rInterval],output
def inAndOutFuncMulTMean(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    momentum=0.95
    renorm_momentum=0.95
    renorm = False
    inputs  = Input(config.inputSize,name='waveform')
    rInterval  = Input(config.inputSizeI,name='interval')
    #rInterval = Reshape(config.inputSizeI+[1])(rInterval)
    mask    = Input(config.inputSizeI[0],name='mask')
    depth   =  len(config.featureL)
    last = inputs
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        #last = AveragePooling2D(pool_size=config.strideL[i],strides=config.strideL[i],padding='same')(last)
        stride= config.strideL[i]
        if stride[0]>1:
            last = last[:,::stride[0]]
        if stride[1]>1:
            last = last[:,:,::stride[1]]
    last = Reshape((last.shape[1],last.shape[3]))(last)
    w = last[:,:,1::2]
    last = last[:,:,::2]
    y=K.mean(last*w,axis=1,keepdims=True)
    y2=K.mean(last*last*w,axis=1,keepdims=True)
    x = K.mean(rInterval*w,axis=1,keepdims=True)
    x2 = K.mean(rInterval*rInterval*w,axis=1,keepdims=True)
    xy = K.mean(rInterval*last*w,axis=1,keepdims=True)
    k =(xy-x*y)/K.clip(x2-x*x,1e9,1e-5)
    r = (xy)/K.sqrt(x2-x*x)/K.clip(K.sqrt(y2-y*y),1e9,1e-5)
    i=-1
    last = K.concatenate([k,r,y],axis=-1)
    last = Dense(config.outputSize[-1]*6)(last)
    last=Activation(config.activationL[i],name='out'+layerStr+'0')(last)
    last = Dense(config.outputSize[-1]*3)(last)
    last=Activation(config.activationL[i],name='out'+layerStr+'1')(last)
    if True:
        last = Dense(config.outputSize[-1]*4)(last)
        last=Activation(config.activationL[i],name='out'+layerStr+'2')(last)
        last = Dense(config.outputSize[-1]*3)(last)
        last=Activation(config.activationL[i],name='out'+layerStr+'3')(last)
        last = Dense(config.outputSize[-1]*2)(last)
        last=Activation(config.activationL[i],name='out'+layerStr+'4')(last)
    output = Dense(config.outputSize[-1])(last)
    return [inputs,rInterval],output
def inAndOutFuncMul_(config, onlyLevel=-10000,training=None):
    #BatchNormalization =LayerNormalization
    BatchNormalization=MBatchNormalization
    #Conv2D = SeparableConv2D
    BNA = -1
    momentum=0.95
    renorm_momentum=0.95
    renorm = False
    inputs  = Input(config.inputSize,name='waveform')
    rInterval  = Input(config.inputSizeI,name='interval')
    mask    = Input(config.inputSizeI[0],name='mask')
    depth   =  len(config.featureL)
    last = inputs
    for i in range(depth-1):
        name = 'CONV'
        layerStr='_%d_'%i
        if i >0:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=2)(last)
        else:
            last = SeparableConv2D(config.featureL[i],kernel_size=config.kernelL[i],\
                strides=(1,1),padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[i],\
                bias_initializer=config.bias_initializerL[i],depth_multiplier=50)(last)
        if config.isBNL[i]:
            last = BatchNormalization(axis=BNA,momentum=momentum,renorm=renorm,renorm_momentum=renorm_momentum,name='BN'+layerStr+'0')(last,training=training)
        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)
        last = AveragePooling2D(pool_size=config.strideL[i],strides=config.strideL[i],padding='same')(last)
    last = Reshape((last.shape[1],last.shape[3]))(last)
    rnn0 = tf.keras.layers.RNN(ODELSTMR(units=200),time_major=False,return_sequences=True,go_backwards=True)
    rnn1 = tf.keras.layers.RNN(ODELSTMR(units=200),time_major=False,return_sequences=True,go_backwards=True)
    #print(last,rInterval)
    last = rnn0((last, rInterval))
    last = rnn1((last, rInterval))
    output = Dense(config.outputSize[-1])(last)
    return [inputs,rInterval],output
def inAndOutFuncODELSTMRConv(config):
    pixel_input = tf.keras.Input(shape=(config.padSize, config.channelSize), name="pixel")
    time_input = tf.keras.Input(shape=(config.padSize,config.outSize ), name="time")
    inputs = [pixel_input,time_input]
    conv1 = Conv1D(10,kernel_size=4,strides=1,padding='same',name='conv1',activity_regularizer='relu')(pixel_input)
    conv2 = Conv1D(10,kernel_size=8,strides=1,padding='same',name='conv2',activity_regularizer='relu')(conv1)
    rnn = tf.keras.layers.RNN(ODELSTMR(units=config.size),time_major=False,return_sequences=True,go_backwards=True,)
    dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10,activation='sigmoid'))
    output_states = rnn((conv2, time_input))
    conv3 = Conv1D(20,kernel_size=1,strides=1,padding='same',name='conv3',activity_regularizer='relu')(output_states)
    conv4 = Conv1D(10,kernel_size=1,strides=1,padding='same',name='conv4',activity_regularizer='relu')(conv3)
    y = dense_layer(conv4)
    return inputs,[y]
def inAndOutFuncODELSTMR(config):
    pixel_input = tf.keras.Input(shape=(config.padSize, config.channelSize), name="pixel")
    time_input = tf.keras.Input(shape=(config.padSize,config.outSize ), name="time")
    inputs = [pixel_input,time_input]
    rnn = tf.keras.layers.RNN(ODELSTMR(units=config.size),time_major=False,return_sequences=True,go_backwards=True,)
    dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10,activation='relu'))
    output_states = rnn((pixel_input, time_input))
    y = dense_layer(output_states)
    rnn = tf.keras.layers.RNN(ODELSTMR(units=config.size),time_major=False,return_sequences=True,go_backwards=True,)
    dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10,activation='relu'))
    output_states = rnn((y, time_input))
    y = dense_layer(output_states)
    rnn = tf.keras.layers.RNN(ODELSTMR(units=config.size),time_major=False,return_sequences=True,go_backwards=True,)
    dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,activation='sigmoid'))
    output_states = rnn((y, time_input))
    y = dense_layer(output_states)
    return inputs,[y]
def inAndOutFuncDt(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input([*config.inputSize[:-2],config.inputSize[-1]*2],name='inputs')
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last = Reshape([*config.inputSize[:-1],config.inputSize[-1]*2])(inputs)
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)

        convL[i] =last

        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'1',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii],name='Dropout'+layerStr+'0')(last)
        else:
            last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'1')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)

        last = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)

    convL[depth] =last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        
        for j in range(i+1):

            layerStr='_%d_%d'%(i,j)

            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(convL[j+1])

            if j in config.dropOutL:
                jj   = config.dropOutL.index(j)
                dConvL[j] =  Dropout(config.dropOutRateL[jj],name='Dropout_'+layerStr+'0')(dConvL[j])
            else:
                dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'0')(dConvL[j])

            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j])
            dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'1')(dConvL[j])
            dConvL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                #outputsL.append(Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
                #padding='same',activation='sigmoid',name='dconv_out_%d'%i)(convL[0]))
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_%d'%i)(convL[0]))
        
    #outputs = Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
    #    padding='same',activation='sigmoid',name='dconv_out')(convL[0])
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
        if config.mode == 'p' or config.mode == 's'or config.mode == 'ps':
            if config.outputSize[-1]>1:
                outputs = Softmax(axis=3)(outputs) 
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    outputs = Reshape([config.outputSize[0],-1])(outputs)
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

class DenseNew(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.initializer = "glorot_uniform"
        super(DenseNew, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_kernel = self.add_weight(\
            shape=(1,1,self.units,input_shape[-2],input_shape[-1]),\
            initializer=self.initializer,\
            name="input_kernel",\
        )
        self.bias = self.add_weight(
            shape=(1,1,self.units,input_shape[-1]),
            initializer=self.initializer,
            name="bias",
        )
        self.built = True
    def call(self, inputs):
        inputs = inputs
        rInput=K.reshape(inputs,(-1,inputs.shape[1],1,inputs.shape[2],inputs.shape[3]))
        return K.sum(rInput*self.input_kernel,axis=3)+self.bias
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'initializer': self.initializer,
        })
        return config

class DenseNew2(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.initializer = "glorot_uniform"
        super(DenseNew2, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_kernel = self.add_weight(\
            shape=(1,self.units,input_shape[-3],1,input_shape[-1]),\
            initializer=self.initializer,\
            name="input_kernel",\
        )
        self.bias = self.add_weight(
            shape=(1,self.units,1,input_shape[-1]),
            initializer=self.initializer,
            name="bias",
        )
        self.built = True
    def call(self, inputs):
        inputs = inputs
        rInput=K.reshape(inputs,(-1,1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))
        return K.sum(rInput*self.input_kernel,axis=2)+self.bias
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'initializer': self.initializer,
        })
        return config
tTrain = (10**np.arange(0,1.000001,1/29))*16

def trainAndTest(model,corrLTrain,corrLValid,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,3,2,1.5],count0=3,perN=200,w0=4,k0=-1):
    '''
    依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    #2#4#8#8*3#8#5#10##model.config.lossFunc.w
    #w0=1.25#1.5
    #perN=100
    tmpDir =  os.path.dirname(outputDir)
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    #model.plot(outputDir+'model.png')
    testCount = len(corrLTest)
    showCount = int(len(corrLTest)*1)
    showD     = int(showCount/40)
    resStr = 'testCount %d showCount %d \n'%(testCount,showCount)
    resStr +='train set setting: %s\n'%corrLTrain
    resStr +='test  set setting: %s\n'%corrLTest
    resStr +='perN: %d count0: %d w0: %.5f\n'%(perN, count0, w0)
    resStr +='sigmaL: %s\n'%sigmaL
    print(resStr)
    trainTestLossL =[]
    for sigma in sigmaL:
        #model.config.lossFunc.w = w0
        corrLTrain.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLValid.timeDisKwarg['sigma']=sigma
        corrLValid.iL=np.array([])
        corrLTrain.iL=np.array([])
        corrLTest.iL=np.array([])
        model.compile(loss=model.config.lossFunc, optimizer='Nadam')
        xTest, yTest, tTest =corrLValid(np.arange(len(corrLValid)))
        resStrTmp, trainTestLoss=model.trainByXYTNew(corrLTrain,xTest=xTest,yTest=yTest, count0=count0, perN=perN,N=20000,k0=k0)
        resStr += resStrTmp
        trainTestLossL.append(trainTestLoss)
   # xTest, yTest, tTest =corrLValid(np.arange(len(corrLValid)))
    yout=model.predict(xTest)  
    for threshold in [0.5,0.7,0.8]:
        corrLValid.plotPickErro(yout,tTrain,fileName=outputDir+'erro_valid.jpg',threshold=threshold)
        #resStr+= printRes(yTest, yout[:,:,level:level+1])+'\n'
    resStr += '\n valid part\n'
    for level in range(yout.shape[-2]):
        print('level: %d'%(len(model.config.featureL)\
            -yout.shape[-2]+level+1))
        resStr +='\nlevel: %d'%(len(model.config.featureL)\
            -yout.shape[-2]+level+1)
        resStr+= printRes_old(yTest, yout[:,:,level:level+1])+'\n'
    xTest=None
    yTest=None
    tTest=None
    corrLValid.x=None
    corrLValid.y=None
    xTest, yTest, tTest =corrLTest(np.arange(showCount))
    yout=model.predict(xTest)
    for threshold in [0.5,0.7,0.8]:
        corrLTest.plotPickErro(yout,tTrain,fileName=outputDir+'erro_test.jpg',\
                threshold=threshold)
    resStr += '\n test part\n'
    for level in range(yout.shape[-2]):
        print('level: %d'%(len(model.config.featureL)\
            -yout.shape[-2]+level+1))
        resStr +='\nlevel: %d'%(len(model.config.featureL)\
            -yout.shape[-2]+level+1)
        resStr+= printRes_old(yTest, yout[:,:,level:level+1])+'\n'
    head = outputDir+'resStr_'+\
        obspy.UTCDateTime(time.time()).strftime('%y%m%d-%H%M%S')
    with open(head+'.log','w') as f:
        ###save model
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(resStr)
    for i in range(len(sigmaL)):
        sigma = sigmaL[i]
        trainTestLoss = trainTestLossL[i]
        np.savetxt('%s_sigma%.3f_loss'%(head,np.mean(sigma)),np.array(trainTestLoss))
    model.save(head+'_model.h5')
    iL=np.arange(0,showCount,showD)
    tmpOutputDir ='%s_predict/'%(head)
    if not os.path.exists(tmpOutputDir):
        os.makedirs(tmpOutputDir)
    for level in range(-1,-model.config.deepLevel-1,-1):
        model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
        T=tTrain,outputDir=tmpOutputDir,level=level)
def trainAndTestMul(model,corrDTrain,corrDValid,corrDTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,3,2,1.5],count0=3,perN=200,w0=4,k0=-1,mul=1,FT=False,FC=False,isOrigin=True):
    '''
    依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    #2#4#8#8*3#8#5#10##model.config.lossFunc.w
    #w0=1.25#1.5
    #perN=100
    tmpDir =  os.path.dirname(outputDir)
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    #model.plot(outputDir+'model.png')
    testCount = len(corrDTest)
    showCount = int(len(corrDTest)*1)
    showD     = int(showCount/40)
    resStr = 'testCount %d showCount %d \n'%(testCount,showCount)
    resStr +='train set setting: %s\n'%corrDTrain
    resStr +='test  set setting: %s\n'%corrDTest
    resStr +='perN: %d count0: %d w0: %.5f\n'%(perN, count0, w0)
    resStr +='sigmaL: %s\n'%sigmaL
    print(resStr)
    trainTestLossL =[]
    for sigma in sigmaL:
        #model.config.lossFunc.w = w0
        corrDTrain.corrL.timeDisKwarg['sigma']=sigma
        corrDTest.corrL.timeDisKwarg['sigma']=sigma
        corrDValid.corrL.timeDisKwarg['sigma']=sigma
        corrDValid.corrL.iL=np.array([])
        corrDTrain.corrL.iL=np.array([])
        corrDTest.corrL.iL=np.array([])
        model.compile(loss=model.config.lossFunc, optimizer='Nadam')
        xTest, yTest, tTest =corrDValid(mul=mul,FT=FT,FC=FC,isOrigin=isOrigin)
        resStrTmp, trainTestLoss=model.trainByXYTNewMul(corrDTrain,xTest=xTest,yTest=yTest, count0=count0, perN=perN,N=20000,k0=k0,mul=mul,testN=len(corrDValid),FC=FC,isOrigin=isOrigin)
        resStr += resStrTmp
        trainTestLossL.append(trainTestLoss)
   # xTest, yTest, tTest =corrLValid(np.arange(len(corrLValid)))
    yout=model.predict(xTest)
    resStr += '\n valid part\n'
    if FT:
        resStr+= printRes_old(yTest, yout,fromI=200)+'\n'
    else:
        resStr+= printRes_old(yTest, yout)+'\n'
    xTest=None
    yTest=None
    tTest=None
    corrDTrain.corrL.clear()
    corrDValid.corrL.clear()
    #corrDValid.corrL.clear()
    xTest, yTest, tTest =corrDTest(mul=mul,FT=FT,FC=FC,isOrigin=isOrigin)
    yout=model.predict(xTest)
    if FT:
        resStr+= printRes_old(yTest, yout,fromI=200)+'\n'
    else:
        resStr+= printRes_old(yTest, yout)+'\n'
    head = outputDir+'resStr_'+\
        obspy.UTCDateTime(time.time()).strftime('%y%m%d-%H%M%S')
    with open(head+'.log','w') as f:
        ###save model
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(resStr)
    for i in range(len(sigmaL)):
        sigma = sigmaL[i]
        trainTestLoss = trainTestLossL[i]
        np.savetxt('%s_sigma%.3f_loss'%(head,np.mean(sigma)),np.array(trainTestLoss))
    model.save(head+'_model.h5')
    iL=random.sample(np.arange(0,showCount,showD).tolist(),10)
    tmpOutputDir ='%s_predict/'%(head)
    if not os.path.exists(tmpOutputDir):
        os.makedirs(tmpOutputDir)
    #for level in range(-1,-model.config.deepLevel-1,-1):
    #    model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    #    T=tTrain,outputDir=tmpOutputDir,level=level)
    corrDTest.corrL.clear()
    return head+'_model.h5'

def trainAndTestSq(model,corrLTrain,corrLValid,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,3,2,1.5],count0=3,perN=2,w0=4):
    '''
    依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    #2#4#8#8*3#8#5#10##model.config.lossFunc.w
    tmpDir =  os.path.dirname(outputDir)
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    #model.plot(outputDir+'model.png')
    testCount = len(corrLTest)
    showCount = int(len(corrLTest)*1)
    showD     = int(showCount/40)
    resStr = 'testCount %d showCount %d \n'%(testCount,showCount)
    resStr +='train set setting: %s\n'%corrLTrain
    resStr +='test  set setting: %s\n'%corrLTest
    resStr +='perN: %d count0: %d w0: %.5f\n'%(perN, count0, w0)
    resStr +='sigmaL: %s\n'%sigmaL
    print(resStr)
    trainTestLossL =[]
    for sigma in sigmaL:
        #model.config.lossFunc.w = w0*(1.5/sigma)**0.5
        corrLTrain.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLValid.timeDisKwarg['sigma']=sigma
        corrLValid.iL=np.array([])
        corrLTrain.iL=np.array([])
        corrLTest.iL=np.array([])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.000001),
            loss=model.config.lossFunc,
        )
        xTest, yTest,nTest, tTest =corrLValid.newCall(np.arange(len(corrLValid)))
        resStrTmp, trainTestLoss=model.trainByXYT(corrLTrain,xTest=xTest,yTest=yTest,\
            nTest=nTest,count0=count0, perN=perN)
        resStr += resStrTmp
        trainTestLossL.append(trainTestLoss)
    corrLValid.timeDisArgv[-1]=(16**np.arange(0,1.000001,1/49))*10
    xTest, yTest,nTest, tTest =corrLValid.newCall(np.arange(len(corrLValid)))
    yout=model.predict((xTest,nTest))  
    for threshold in [0.5,0.7,0.8]:
        corrLValid.plotPickErro(yout,tTrain,fileName=outputDir+'erro_valid.jpg',\
            threshold=threshold)
    xTest, yTest,nTest, tTest =corrLTest.newCall(np.arange(showCount))
    yout=model.predict((xTest,nTest))
    resStr += '\n test part\n'
    resStr+= printRes_sq(yTest, yout)+'\n'
        #resStr+= printRes(yTest, yout[:,:,level:level+1])+'\n'
    head = outputDir+'resStr_'+\
        obspy.UTCDateTime(time.time()).strftime('%y%m%d-%H%M%S')
    with open(head+'.log','w') as f:
        ###save model
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(resStr)
    for i in range(len(sigmaL)):
        sigma = sigmaL[i]
        trainTestLoss = trainTestLossL[i]
        np.savetxt('%s_sigma%.3f_loss'%(head,sigma),np.array(trainTestLoss))
    for threshold in [0.5,0.7,0.8]:
        corrLTest.plotPickErro(yout,tTrain,fileName=outputDir+'erro_test.jpg',\
                threshold=threshold)
    model.save(head+'_model',save_format='h5')
    iL=np.arange(0,showCount,showD)
    for level in range(-1,-model.config.deepLevel-1,-1):
        model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
        T=tTrain,outputDir=outputDir,level=level)


def trainAndTestCross(model0,model1,corrLTrain0,corrLTrain1,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,2],modeL=['conv','conv']):
    '''
    依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    #different data train different part
    w0 = 2#5#10##model.config.lossFunc.w
    for i in range(len(sigmaL)):
        sigma = sigmaL[i]
        mode = modeL[i]
        #model0.config.lossFunc.w = w0*(4/sigma)**0.5
        #model1.config.lossFunc.w = w0*(4/sigma)**0.5
        corrLTrain0.timeDisKwarg['sigma']=sigma
        corrLTrain1.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLTest.iL=np.array([])
        if mode =='conv':
            model0.setTrain(['conv','CONV'],True)
            model1.setTrain([],False)
            per1=0.5
        if mode =='anti_conv':
            model0.setTrain([],False)
            model1.setTrain(['conv','CONV'],True)
            per1=0.5
        if mode =='dconv':
            model0.setTrain([],False)
            model1.setTrain(['dconv'],True)
            per1=2
        if mode =='None':
            model0.setTrain([],False)
            model1.setTrain([],False)
            per1=0.5
        if mode =='conv_dconv':
            model0.setTrain(['conv','CONV'],True)
            model1.setTrain(['dconv','DCONV'],True)
            per1 = 0.5
        if mode =='0':
            model0.setTrain([],False)
            model1.setTrain([],True)
            per1 = 0.5
        xTest, yTest, tTest =corrLTest.newCall(np.arange(2000,4000))
        model0.trainByXYTCross(model1,corrLTrain0,corrLTrain1,xTest=xTest,yTest=yTest,per1=per1)
    xTest, yTest, tTest =corrLTest.newCall(np.arange(2000))
    corrLTest.plotPickErro(model0.predict(xTest),tTrain,\
    fileName=outputDir+'erro.jpg')
    iL=np.arange(0,1000,50)
    model0.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir)
    xTest, yTest, tTest =corrLTrain0(np.arange(10000))
    corrLTrain0.plotPickErro(model0.predict(xTest),tTrain,\
    fileName=outputDir+'erro0.jpg')
    iL=np.arange(0,1000,50)
    model0.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir+'_0_')
    xTest, yTest, tTest =corrLTrain1(np.arange(10000))
    corrLTrain1.plotPickErro(model0.predict(xTest),tTrain,\
    fileName=outputDir+'erro1.jpg')
    iL=np.arange(0,1000,50)
    model0.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir+'_1_')

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
            self.mul           = mul
            self.data_format = 'channels_last'
            self.inputSize     = [512*2,mul,4]
            self.outputSize    = [512*2*up,mul,50]
            self.inputSize     = [512*12,mul,4]
            self.outputSize    = [512*12*up,mul,50]
            self.inputSize     = [512*6,mul,4]
            self.outputSize    = [512*6*up,mul,50]
            self.featureL      = [75 ,125,175,250,350,500,600,320]
            self.featureL      = [60 ,120,160,240,300,400,500,320]
            self.featureL      = [60 ,100,150,200,250,350,400,320]
            self.featureL      = [60 ,120,160,240,300,400,500,320]
            self.featureL      = [60 ,100,150,200,250,350,400,320]
            self.featureL      = [60 ,80,120,180,240,320,360,320]
            self.featureL      = [60 ,90,150,210,270,360,420,320]
            self.featureL      = [60 ,80,160,200,240,320,400,320]
            self.featureL      = [80 ,100,120,160,200,240,320,320]
            #self.featureL      = [60 ,120,160,240,300,400,500,320]
            self.kernelFirstL  = ((14**np.arange(0,1,1/49)*10)*6).astype(np.int)
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
            self.strideL       = [(2,1),  (2,1), (3,1), (4,1), (4,1),  (4,1),(4,1),(1,mul)  ,(up,1)]
            self.kernelL       = [(int(160*2*1.5),1),(12,1),(12,1),(12,1),(12,1), (8,1),(8,1),(1,mul*2),(up*12,1)]
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
            self.poolL        = [MaxPooling2D]*20
            #self.poolL        = [AveragePooling2D]*20
            self.lossFunc     = lossFuncMSE(**kwags)#1lossFuncSoft(**kwags)#0
            self.inAndOutFunc = inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3FFT_norm#inAndOutFuncNewUp
            self.lossFuncNP     = lossFuncMSENP(**kwags)#lossFuncSoftNP(**kwags)#
            self.deepLevel = 1
        if mode=='surfFC':
            self.mul           = mul
            self.data_format = 'channels_last'
            self.inputSize     = [512*6,mul,52]
            self.outputSize    = [512*6*up,mul,50]
            self.featureL      = [75 ,125,175,250,350,500,600,320]
            self.featureL      = [60 ,120,160,240,300,400,500,320]
            self.featureL      = [60 ,100,150,200,250,350,400,320]
            self.featureL      = [60 ,120,160,240,300,400,500,320]
            self.featureL      = [60 ,100,150,200,250,350,400,320]
            self.featureL      = [60 ,80,120,180,240,320,360,320]
            self.featureL      = [60 ,90,150,210,270,360,420,320]
            self.featureL      = [60 ,80,160,200,240,320,400,320]
            #self.featureL      = [60 ,120,160,240,300,400,500,320]
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
            self.strideL       = [(2,1),  (2,1), (3,1), (4,1), (4,1),  (4,1),(4,1),(1,mul)  ,(up,1)]
            self.kernelL       = [(4,1),  (6,1),(6,1),(8,1),(8,1), (8,1),(8,1),(1,mul*2),(up*4,1)]
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
            self.poolL        = [MaxPooling2D]*20
            #self.poolL        = [AveragePooling2D]*20
            self.lossFunc     = lossFuncMSE(**kwags)#1lossFuncSoft(**kwags)#0
            self.inAndOutFunc = inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT_norm_FC#inAndOutFuncNewUp
            self.lossFuncNP     = lossFuncMSENP(**kwags)#lossFuncSoftNP(**kwags)#
            self.deepLevel = 1
        if mode=='surfFT':
            self.mul           = mul
            self.data_format = 'channels_last'
            self.inputSize     = [800,50,1]
            self.outputSize    = [800,50,1]
            self.featureL      = [8,16,32,128,256,512]
            self.featureL      = [4,8,16,32,128,256]
            self.featureL      = [8,16,32,128,256,512]
            self.strideL       = [(4,1), (2,2), (1, 5), (4, 5), (5,1),  (5,1),(4,1),(1,mul)  ,(up,1)]
            self.kernelL       = [(8,4), (4,2), (4,10),(8,10), (10,1), (10,1),(4,1),(1,mul*2),(4,4)]
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
            self.poolL        = [MaxPooling2D]*20
            #self.poolL        = [AveragePooling2D]*20
            self.lossFunc     = lossFuncSoftFT(**kwags)#1lossFuncSoft(**kwags)#0
            self.inAndOutFunc = inAndOutFuncNewNetDenseUpStrideSimpleMSeparableUpNewV3MFT_FT#inAndOutFuncNewUp
            self.lossFuncNP     = lossFuncSoftFTNP(**kwags)#lossFuncSoftNP(**kwags)#
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

class fcnConfigGather:
    def __init__(self,**kwags):
        self.inputSize     = [3200,200,1]
        self.outputSize    = [3200,200,10]
        self.featureL      = [10,15,20,25,30,40]
        self.strideL       = [(2,2),(4,2),(4,2),(4,5),(5,5),(5,1)]
        self.kernelL       = [(4,4),(8,4),(8,4),(8,10),(10,10),(10,1)]
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
        self.poolL        = [MaxPooling2D,MaxPooling2D,MaxPooling2D,\
        MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,\
        MaxPooling2D,MaxPooling2D,MaxPooling2D]
        self.poolL        = [MaxPooling2D]*20
        #self.poolL        = [AveragePooling2D]*20
        self.lossFunc     = lossFuncMSEO()
        self.inAndOutFunc = inAndOutFuncNewNetDenseUpNoPoolingGather#inAndOutFuncNewUp
        self.lossFuncNP     = lossFuncMSENPO
        self.deepLevel = 1
    def inAndOut(self,*argv,**kwarg):
        return self.inAndOutFunc(self,*argv,**kwarg)

class odelstmrConfig:
    def __init__(self,mode='surf',size=128):
        self.inAndOutFunc = inAndOutFuncODELSTMR
        self.mode=mode
        self.lossFunc     = lossFuncSoftSq(w=10)
        self.padSize, self.channelSize,self.outSize=[4096*3,3,1]
        self.size=size
    def inAndOut(self,*argv,**kwarg):
        return self.inAndOutFunc(self,*argv,**kwarg)
w1=np.ones(1500)*0.5
w0=np.ones(250)*(-0.75)
w2=np.ones(250)*(-0.25)
w=np.append(w0,w1)
w=np.append(w,w2)
#wY=K.variable(w.reshape((1,2000,1,1)))

w11=np.ones(1800)*0
w01=np.ones(100)*(-0.8)*0
w21=np.ones(100)*(-0.3)*0
w1=np.append(w01,w11)
w1=np.append(w1,w21)
W1=w1.reshape((1,2000,1,1))
#wY1=K.variable(W1)
#wY1Short=K.variable(W1[:,200:1800])
#wY1Shorter=K.variable(W1[:,400:1600])
#wY1500=K.variable(W1[:,250:1750])
W2=np.zeros((1,2000,1,3))
W2[0,:,:,0]=W1[0,:,:,0]*0+(1-0.13)
W2[0,:,:,1]=W1[0,:,:,0]*0+(1-0.13)
W2[0,:,:,2]=W1[0,:,:,0]*0+0.13
#wY2=K.variable(W2)

def lossFuncNew(y,yout):

    #yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    #y=(K.sign(y+0.1)+1)*y/2
    y0=0.04
    return -K.mean((y*K.log(K.clip(yout,1e-6,1))/y0+(1-y)*K.log(K.clip(1-yout,1e-6,1))/(1-y0))*(1+K.sign(y)*wY1),axis=[0,1,2,3])

def lossFuncNewS(y,yout):
    #y=y
    #yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    #y=(K.sign(y+0.1)+1)*y/2
    y0=0.04
    return -K.mean((y*K.log(K.clip(yout,1e-6,1))/y0+(1-y)*K.log(K.clip(1-yout,1e-6,1))/(1-y0))*(1+K.sign(y)*wY1),axis=[0,1,2,3])

def genModel0(modelType='norm',phase='p'):
    return model(config=fcnConfig(mode=phase),channelList=[0,1,2]),2000,1
'''

for i in range(10):
    plt.plot(inputData[i,:,0,0]/5,'k',linewidth=0.3)
    plt.plot(probP[i,:,0,0].transpose(),'b',linewidth=0.3)
    plt.plot(probS[i,:,0,0].transpose(),'r',linewidth=0.3)
    plt.show()
'''
class model(Model):
    def __init__(self,weightsFile='',metrics=rateNp,\
        channelList=[0],onlyLevel=-1000):
        #defProcess()
        config=fcnConfig()
        #channelList=[0]
        config.inputSize[-1]=len(channelList)
        self.genM(config, onlyLevel)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        self.compile(loss=self.config.lossFunc, optimizer='Nadam')
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
        print(self.summary())

    def genM(self,config, onlyLevel=-1000):
        inputs, outputs = config.inAndOut(onlyLevel=onlyLevel)
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x,**kwargs):
        #print('inx')
        x = self.inx(x)
        #print('inx done')
        if self.config.mode=='surf':
            return super().predict(x,**kwargs).astype(np.float16)
        else:
            return super().predict(x,**kwargs)
    def fit(self,x,y,batchSize=None,**kwargs):
        x=self.inx(x)
        if np.isnan(x).sum()>0 or np.isinf(x).sum()>0:
            print('bad record')
            return None
        return super().fit(x ,y,batch_size=batchSize,**kwargs)
    def plot(self,filename='model.png'):
        plot_model(self, to_file=filename)
    def inx(self,x):
        if self.config.mode=='surf':
            if x.shape[-1] > len(self.channelList):
                x = x[:,:,:,self.channelList]
            timeN0 = np.float32(x.shape[1])
            timeN  = (x!=0).sum(axis=1,keepdims=True).astype(np.float32)
            timeN *= 1+0.2*(np.random.rand(*timeN.shape).astype(np.float32)-0.5)
            x/=(x.std(axis=(1,2),keepdims=True))*(timeN0/timeN)**0.5
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
    def trainByXYT(self,XYT,N=2000,perN=200,batchSize=None,xTest='',\
        yTest='',k0 = 1e-3,t='',count0=3,resL=globalFL):
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
                        'sampleTimes',sampleDone.mean(),'last F:',resL[-1],'min Loss:',lossMin)
                    resStr+='\n %d train loss : %f valid loss :%f F: %f'%(i,lossTrain,lossTest,resL[-1])
                    lossTest-=resL[-1]
                    trainTestLoss.append([i,lossTrain,lossTest])
                    if i%30==0 and i>10:
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
                    if count ==0:
                        break
                    #print(self.metrics)
                    
            if i%100==0:
                print('learning rate: ',self.optimizer.lr)
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.95)
            if i>10 and i%50==0:
                perN += int(perN*0.05)
                perN = min(400, perN)
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
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(19),fileStr='',\
        level=-1):
        y = self.predict(x)
        f = 1/T
        count = x.shape[1]
        for i in range(len(x)):
            #print('show',i)
            timeL = np.arange(count)*delta
            if len(time0L)>0:
                timeL+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            xlimNew=[0,500]
            #xlim=xlimNew
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0)
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
            plt.pcolor(timeL,f,y0[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            plt.plot(timeL[pos.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,3)
            plt.pcolor(timeL,f,y[i,:,level,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            #plt.clim(0,1)
            plt.plot(timeL[pos0.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,4)
            delta = timeL[1] -timeL[0]
            N = len(timeL)
            fL = np.arange(N)/N*1/delta
            for j in range(x.shape[-1]):
                spec=np.abs(np.fft.fft(self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0])).reshape([-1])
                plt.plot(fL,spec/(spec.max()+1e-16),'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            plt.xlabel('f/Hz')
            plt.ylabel('A')
            plt.xlim([fL[1],fL[-1]/2])
            #plt.gca().semilogx()
            plt.savefig('%s%s_%d_%d.jpg'%(outputDir,fileStr,level,i),dpi=200)
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

batchMax=32#8*24/16
#batchMax=8*24/32

batchMaxFT=8
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
        #self.lf = self.config.lossFunc
        self.compile(loss=self.config.lossFunc, optimizer='Adam')
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
        print(self.summary())
        self.lossFunc=config.lossFuncNP
    def genM(self,config, onlyLevel=-1000):
        inputs, outputs = config.inAndOut(onlyLevel=onlyLevel)
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x,batch_size=-1,**kwargs):
        #print('inx')
        maxN = 512
        NX = len(x)
        if batch_size==-1:
            batch_size = int(batchMax/self.mul)
        #print('inx done')
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
            x[:,:,:,0]/=(np.abs(x[:,:,:,0]).max(axis=(1,),keepdims=True))*(np.random.rand(len(x),1,1)*0.2+0.9)#*(timeN0/timeN)**0.5
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
        yTest='',k0 = 2e-3,t='',count0=20,resL=globalFL,mul=1,testN=-1,isSeparate=False,**kwargs):
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
            if isSeparate:
                for j in range(list(y.shape)[-1]):
                    self.config.lossFunc.focusChannel = j#int(i%list(y.shape)[-1])
                    for layer in self.layers:
                        if 'CONV_0__wave_0' in layer.name:
                            channelStr = 'CONV_0__wave_0_%d'%j
                            if layer.name==channelStr:
                                layer.trainable=True
                                print('train',channelStr)
                            else:
                                layer.trainable=False
                    lr = K.get_value(self.optimizer.lr)
                    #self.compile(loss=self.config.lossFunc, optimizer='Adam')
                    for layer in self.layers:
                        if 'CONV_0__wave_0' in layer.name:
                            channelStr = 'CONV_0__wave_0_%d'%j
                            if layer.name==channelStr:
                                #layer.trainable=True
                                print('train',channelStr,layer.trainable)
                            else:
                                #layer.trainable=False
                                pass
                    K.set_value(self.optimizer.lr,lr)
                    CW = np.zeros([y.shape[0],1,1,y.shape[-1]])
                    CW[:,:,:,j]=1
                    CW /= CW.mean() 
                    self.fit(x ,(y,CW),batchSize=batchSize,)
            else:
                #CW = np.zeros([y.shape[0],1,1,y.shape[-1]])
                #CW[:,:,:,:]=1
                #CW /= CW.mean() 
                self.fit(x,y,batchSize=batchSize,)
            x=0
            y=0
            #print(self.layers[47].variables[-2][-10:],self.layers[47].variables[-1][-10:])
            if int(sampleDone.mean())>sampleTime or isSeparate:
                if isSeparate:
                    #self.config.lossFunc.focusChannel =-1
                    lr = K.get_value(self.optimizer.lr)
                    #self.compile(loss=self.config.lossFunc, optimizer='Adam')
                    K.set_value(self.optimizer.lr,lr )
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

class modelFT(Model):
    def __init__(self,weightsFile='',metrics=rateNp,\
        channelList=[0],onlyLevel=-1000,up=1,mul=1,**kwags):
        #channelList=[0]
        config=fcnConfig('surfFT',up=up,mul=mul,**kwags)
        #defProcess()
        self.mul=mul
        #config.inputSize[-1]=len(channelList)
        self.genM(config, onlyLevel)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        #self.lf = self.config.lossFunc
        self.compile(loss=self.config.lossFunc, optimizer='Adam')
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
        print(self.summary())
        self.lossFunc=config.lossFuncNP
    def genM(self,config, onlyLevel=-1000):
        inputs, outputs = config.inAndOut(onlyLevel=onlyLevel)
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x,batch_size=-1,**kwargs):
        #print('inx')
        maxN = 512
        NX = len(x)
        if batch_size==-1:
            batch_size = int(batchMax/self.mul)
        #print('inx done')
        if self.config.mode=='surf' or self.config.mode=='surfUp' or self.config.mode=='surfFT':
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
        x[:,:,:,0]/=(np.abs(x[:,:,:,0]).max(axis=(1,),keepdims=True))*(np.random.rand(len(x),1,1)*0.2+0.9)#*(timeN0/timeN)**0.5
        return x
    #def __call__(self,x,*args,**kwargs):
    #    return super(Model, self).__call__(K.tensor(self.inx(x)))
    def trainByXYTNewMul(self,XYT,N=2000,perN=200,batchSize=32,xTest='',\
        yTest='',k0 = 2e-3,t='',count0=20,resL=globalFL,mul=1,testN=-1,isSeparate=False,**kwargs):
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
        xTrain, yTrain , t0LTrain = XYT(iL,mul=mul,FT=True)
        print('training',xTrain.shape,len(iL))
        sampleTime=-1
        r = len(XYT)/len(XYT.corrL)
        batchSize = int(batchMaxFT/self.mul)
        perM=int(perN/mul)
        perM= int(perM/batchSize)*batchSize
        for i in range(N):
            gc.collect()
            iL = random.sample(indexLMul,perM)
            for ii in iL:
                sampleDone[ii]+=r*mul
            x, y , t0L = XYT(iL,mul=mul,N=mul,FT=True)
            print('fromT:',np.array(t0L).mean())
            print('loop:',sampleDone.mean())
            if isSeparate:
                for j in range(list(y.shape)[-1]):
                    self.config.lossFunc.focusChannel = j#int(i%list(y.shape)[-1])
                    for layer in self.layers:
                        if 'CONV_0__wave_0' in layer.name:
                            channelStr = 'CONV_0__wave_0_%d'%j
                            if layer.name==channelStr:
                                layer.trainable=True
                                print('train',channelStr)
                            else:
                                layer.trainable=False
                    lr = K.get_value(self.optimizer.lr)
                    #self.compile(loss=self.config.lossFunc, optimizer='Adam')
                    for layer in self.layers:
                        if 'CONV_0__wave_0' in layer.name:
                            channelStr = 'CONV_0__wave_0_%d'%j
                            if layer.name==channelStr:
                                #layer.trainable=True
                                print('train',channelStr,layer.trainable)
                            else:
                                #layer.trainable=False
                                pass
                    K.set_value(self.optimizer.lr,lr)
                    CW = np.zeros([y.shape[0],1,1,y.shape[-1]])
                    CW[:,:,:,j]=1
                    CW /= CW.mean() 
                    self.fit(x ,(y,CW),batchSize=batchSize,)
            else:
                #CW = np.zeros([y.shape[0],1,1,y.shape[-1]])
                #CW[:,:,:,:]=1
                #CW /= CW.mean() 
                self.fit(x,y,batchSize=batchSize,)
            x=0
            y=0
            #print(self.layers[47].variables[-2][-10:],self.layers[47].variables[-1][-10:])
            if int(sampleDone.mean())>sampleTime or isSeparate:
                if isSeparate:
                    #self.config.lossFunc.focusChannel =-1
                    lr = K.get_value(self.optimizer.lr)
                    #self.compile(loss=self.config.lossFunc, optimizer='Adam')
                    K.set_value(self.optimizer.lr,lr )
                sampleTime = int(sampleDone.mean())
                if len(xTest)>0:
                    youtTrain = 0
                    youtTest  = 0
                    print('cal Predict')
                    youtTrain = self.predict(xTrain,batch_size=batchSize)
                    print('cal Test')
                    print(xTest.shape)
                    youtTest  = self.predict(xTest,batch_size=batchSize)
                    print('calDone')
                    lossTrain = self.lossFunc(yTrain,youtTrain)
                    lossTest    = self.lossFunc(yTest,youtTest)
                    print('train loss',lossTrain,'test loss: ',lossTest,\
                        'no sampleRate:', 1 - np.sign(sampleDone).mean(),\
                        'sampleTimes',sampleDone.mean(),'last F:',resL[-1],'min Loss:',lossMin,'lr',K.get_value(self.optimizer.lr),'count:',count)
                    resStr+='\n %d train loss : %f valid loss :%f F: %f sampleTime: %d'%(i,lossTrain,lossTest,resL[-1],sampleTime)
                    #lossTest-=resL[-1]
                    trainTestLoss.append([i,lossTrain,lossTest])
                    resStr+='\ntrain '+printRes_old(yTrain,youtTrain,fromI=200)
                    resStr+='\ntest '+printRes_old(yTest, youtTest,isUpdate=True,fromI=200)
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
                        batchSize = int(batchMaxFT/self.mul)
                        print('batchSize ',batchSize)
            gc.collect()
        self.set_weights(w0)
        return resStr,trainTestLoss
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

class modelFC(Model):
    def __init__(self,weightsFile='',metrics=rateNp,\
        channelList=[0],onlyLevel=-1000,up=1,mul=1,**kwags):
        #channelList=[0]
        config=fcnConfig('surfFC',up=up,mul=mul,**kwags)
        #defProcess()
        self.mul=mul
        #config.inputSize[-1]=len(channelList)
        self.genM(config, onlyLevel)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        #self.lf = self.config.lossFunc
        self.compile(loss=self.config.lossFunc, optimizer='Adam')
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
        print(self.summary())
        self.lossFunc=config.lossFuncNP
    def genM(self,config, onlyLevel=-1000):
        inputs, outputs = config.inAndOut(onlyLevel=onlyLevel)
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x,batch_size=-1,**kwargs):
        #print('inx')
        maxN = 512
        NX = len(x)
        if batch_size==-1:
            batch_size = int(batchMax/self.mul)
        #print('inx done')
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
        #timeN0 = np.float32(x.shape[1])
        #timeN  = (x[:,:,:,0]!=0).sum(axis=1,keepdims=True).astype(np.float32)
        #timeN *= 1+0.2*(np.random.rand(*timeN.shape).astype(np.float32)-0.5)
        x[:,:,:,:]/=(np.abs(x).max(axis=(1,),keepdims=True))*(np.random.rand(len(x),1,1,x.shape[-1])*0.2+0.9)#*(timeN0/timeN)**0.5
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
        yTest='',k0 = 2e-3,t='',count0=20,resL=globalFL,mul=1,testN=-1,isSeparate=False,FC=True,isOrigin=True):
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
        xTrain, yTrain , t0LTrain = XYT(iL,mul=mul,FC=FC,isOrigin=isOrigin)
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
            x, y , t0L = XYT(iL,mul=mul,N=mul,FC=FC,isOrigin=isOrigin)
            print(x.shape,y.shape)
            print('fromT:',np.array(t0L).mean())
            print('loop:',sampleDone.mean())
            if isSeparate:
                for j in range(list(y.shape)[-1]):
                    self.config.lossFunc.focusChannel = j#int(i%list(y.shape)[-1])
                    for layer in self.layers:
                        if 'CONV_0__wave_0' in layer.name:
                            channelStr = 'CONV_0__wave_0_%d'%j
                            if layer.name==channelStr:
                                layer.trainable=True
                                print('train',channelStr)
                            else:
                                layer.trainable=False
                    lr = K.get_value(self.optimizer.lr)
                    #self.compile(loss=self.config.lossFunc, optimizer='Adam')
                    for layer in self.layers:
                        if 'CONV_0__wave_0' in layer.name:
                            channelStr = 'CONV_0__wave_0_%d'%j
                            if layer.name==channelStr:
                                #layer.trainable=True
                                print('train',channelStr,layer.trainable)
                            else:
                                #layer.trainable=False
                                pass
                    K.set_value(self.optimizer.lr,lr)
                    CW = np.zeros([y.shape[0],1,1,y.shape[-1]])
                    CW[:,:,:,j]=1
                    CW /= CW.mean() 
                    self.fit(x ,(y,CW),batchSize=batchSize,)
            else:
                #CW = np.zeros([y.shape[0],1,1,y.shape[-1]])
                #CW[:,:,:,:]=1
                #CW /= CW.mean() 
                self.fit(x,y,batchSize=batchSize,)
            x=0
            y=0
            #print(self.layers[47].variables[-2][-10:],self.layers[47].variables[-1][-10:])
            if int(sampleDone.mean())>sampleTime or isSeparate:
                if isSeparate:
                    #self.config.lossFunc.focusChannel =-1
                    lr = K.get_value(self.optimizer.lr)
                    #self.compile(loss=self.config.lossFunc, optimizer='Adam')
                    K.set_value(self.optimizer.lr,lr )
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

class modelSq(Model):
    def __init__(self,weightsFile='',metrics=rateNp,\
        channelList=[1,2,3,4],onlyLevel=-1000,maxCount=-1):
        config=odelstmrConfig()
        config.channelSize=len(channelList) 
        if maxCount >0:
            config.padSize=maxCount
        self.genM(config)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        #self.compile(loss=self.config.lossFunc, optimizer='Nadam')
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
        print(self.summary())

    def genM(self,config):
        inputs, outputs = config.inAndOut()
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x):
        #print('inx')
        x = self.inx(x)
        #print('inx done')
        if self.config.mode in ['surf','surfdt']:
            return super().predict(x).astype(np.float16)
        else:
            return super().predict(x)
    def fit(self,x,y,batchSize=None):
        x=self.inx(x)
        if isinstance(x,tuple) or isinstance(x,list):
            print(x[0].shape,x[1].shape)
            if np.isnan(x[0]).sum()>0 or np.isinf(x[0]).sum()>0:
                print('bad record')
                return None
        else:
            print(x.shape)
            if np.isnan(x).sum()>0 or np.isinf(x).sum()>0:
                print('bad record')
                return None
        return super().fit(x=x ,y=(y,),batch_size=batchSize)
    def plot(self,filename='model.png'):
        plot_model(self, to_file=filename)
    def inx(self,x):
        x0 = x[0]
        x1 = x[1].copy()
        x1[:] = x1[:,1:2]-x1[:,0:1]
        x=x0
        if self.config.mode=='surf':
            if x.shape[-1] > len(self.channelList):
                x = x[:,:,self.channelList]
            timeN0 = np.float32(x.shape[1])
            timeN  = (x!=0).sum(axis=1,keepdims=True).astype(np.float32)
            timeN *= 1+0.2*(np.random.rand(*timeN.shape).astype(np.float32)-0.5)
            x/=(x.std(axis=(1),keepdims=True))*(timeN0/timeN)**0.5
        else:
            x/=x.std(axis=(1,2),keepdims=True)+np.finfo(x.dtype).eps
        return (x,x1)
    #def __call__(self,x,*args,**kwargs):
    #    return super(Model, self).__call__(self.inx(x))
    def train(self,x,y,**kwarg):
        if 't' in kwarg:
            t = kwarg['t']
        else:
            t = ''
        XYT = xyt(x,y,t)
        self.trainByXYT(XYT,**kwarg)
    def trainByXYT(self,XYT,N=2000000,perN=10,batchSize=None,xTest='',\
        yTest='',nTest='',k0 = 1e-6,t='',count0=3):
        if k0>-1:
            K.set_value(self.optimizer.lr, k0)
        indexL = range(len(XYT))
        sampleDone = np.zeros(len(XYT))
        #print(indexL)
        lossMin =100
        count   = count0
        w0 = self.get_weights()
        resStr=''
        trainTestLoss = []
        iL = random.sample(indexL,perN)
        xTrain, yTrain , nTrain,t0LTrain = XYT.newCall(iL)
        #print(self.metrics)
        for i in range(N):
            iL = random.sample(indexL,perN)
            for ii in iL:
                sampleDone[ii]+=1
            x, y ,n, t0L = XYT.newCall(iL)
            print(x.shape,y.shape,n.shape)
            #print(XYT.iL)
            self.fit((x,n),y,batchSize=batchSize)
            if i%10==0:
                if len(xTest)>0:
                    lossTrain = self.evaluate(x=self.inx((xTrain,nTrain)),y=yTrain)
                    lossTest    = self.evaluate(x=self.inx((xTest,nTest)),y=yTest)
                    print('train loss',lossTrain,'test loss: ',lossTest,\
                        'sigma: ',XYT.timeDisKwarg['sigma'],\
                        'w: ',self.config.lossFunc.w, \
                        'no sampleRate:', 1 - np.sign(sampleDone).mean(),\
                        'sampleTimes',sampleDone.mean())
                    resStr+='\n %d train loss : %f valid loss :%f'%(i,lossTrain,lossTest)
                    trainTestLoss.append([i,lossTrain,lossTest])
                    if lossTest >= lossMin:
                        count -= 1
                    if lossTest > 3*lossMin:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if lossTest < lossMin:
                        count = count0
                        lossMin = lossTest
                        w0 = self.get_weights()
                        print('find better')
                    if count ==0:
                        break
                    #print(self.metrics)
                    
                    if i%30==0:
                        youtTrain = 0
                        youtTest  = 0
                        youtTrain = self.predict((xTrain,nTrain))
                        youtTest  = self.predict((xTest,nTest))
                        resStr+='\ntrain '+printRes_sq(yTrain, youtTrain)
                        resStr+='\ntest '+printRes_sq(yTest, youtTest)
            if i%500==0:
                print('learning rate: ',self.optimizer.lr)
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.99)
            #if i>10 and i%5==0:
            #    perN += int(perN*0.05)
            #    perN = min(1000, perN)
        self.set_weights(w0)
        return resStr,trainTestLoss
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(19),fileStr='',\
        level=-1):
        y = self.predict(x)
        f = 1/T
        count = x.shape[1]
        for i in range(len(x)):
            #print('show',i)
            timeL = np.arange(count)*delta
            if len(time0L)>0:
                timeL+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            xlimNew=[0,500]
            #xlim=xlimNew
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0)
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
            plt.pcolor(timeL,f,y0[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            plt.plot(timeL[pos.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,3)
            plt.pcolor(timeL,f,y[i,:,level,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            #plt.clim(0,1)
            plt.plot(timeL[pos0.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,4)
            delta = timeL[1] -timeL[0]
            N = len(timeL)
            fL = np.arange(N)/N*1/delta
            for j in range(x.shape[-1]):
                spec=np.abs(np.fft.fft(self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0])).reshape([-1])
                plt.plot(fL,spec/(spec.max()+1e-16),'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            plt.xlabel('f/Hz')
            plt.ylabel('A')
            plt.xlim([fL[1],fL[-1]/2])
            #plt.gca().semilogx()
            plt.savefig('%s%s_%d_%d.jpg'%(outputDir,fileStr,level,i),dpi=200)
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


class modelDt(modelSq):
    def __init__(self,weightsFile='',metrics=rateNp,\
        channelList=[1,2,3,4],onlyLevel=-1000,maxCount=-1):
        #config.channelSize=len(channelList)
        config=fcnConfig(mode='surfdt')
        config.inputSize[-1]=len(channelList)
        self.genM(config)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        #self.compile(loss=self.config.lossFunc, optimizer='Nadam')
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
        print(self.summary())
    def inx(self,x):
        n = x[1]
        x = x[0]
        if self.config.mode in ['surf','surfdt']:
            if x.shape[-1] > len(self.channelList):
                x = x[:,:,self.channelList]
            timeN0 = np.float32(x.shape[1])
            timeN  = (x!=0).sum(axis=1,keepdims=True).astype(np.float32)
            timeN *= 1+0.2*(np.random.rand(*timeN.shape).astype(np.float32)-0.5)
            x/=(x.std(axis=(1),keepdims=True))*(timeN0/timeN)**0.5
            sinx = np.sin(2*n*np.pi)*x
            cosx = np.cos(2*n*np.pi)*x
            x = np.concatenate([sinx,cosx],axis=2)
            '''
            sinx = np.sin(2*n*np.pi)
            cosx = np.cos(2*n*np.pi)
            x = np.concatenate([x,sinx,cosx,sinx*cosx],axis=2)
            '''
            #x = x.reshape([x.shape[0],x.shape[1],1,x.shape[2]])
        else:
            x/=x.std(axis=(1,2),keepdims=True)+np.finfo(x.dtype).eps
        return  x
'''
def genM(self,config):
        inputs, outputs = config.inAndOut()
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x):
        #print('inx')
        x = self.inx(x)
        #print('inx done')
        if self.config.mode in ['surf','surfdt']:
            return super().predict(x).astype(np.float16)
        else:
            return super().predict(x)
    def fit(self,x,y,batchSize=None):
        x=self.inx(x)
        if np.isnan(x[0]).sum()>0 or np.isinf(x[0]).sum()>0:
            print('bad record')
            return None
        return super().fit(x=x ,y=(y,),batch_size=batchSize,epochs=1)
    def plot(self,filename='model.png'):
        plot_model(self, to_file=filename)
    #def __call__(self,x,*args,**kwargs):
    #    return super(Model, self).__call__(self.inx(x))
    def train(self,x,y,**kwarg):
        if 't' in kwarg:
            t = kwarg['t']
        else:
            t = ''
        XYT = xyt(x,y,t)
        self.trainByXYT(XYT,**kwarg)
    def trainByXYT(self,XYT,N=2000,perN=10,batchSize=None,xTest='',\
        yTest='',nTest='',k0 = 4e-3,t='',count0=3):
        if k0>1:
            K.set_value(self.optimizer.lr, k0)
        indexL = range(len(XYT))
        sampleDone = np.zeros(len(XYT))
        #print(indexL)
        lossMin =100
        count   = count0
        w0 = self.get_weights()
        resStr=''
        trainTestLoss = []
        iL = random.sample(indexL,perN)
        xTrain, yTrain , nTrain,t0LTrain = XYT.newCall(iL)
        #print(self.metrics)
        for i in range(N):
            iL = random.sample(indexL,perN)
            for ii in iL:
                sampleDone[ii]+=1
            x, y ,n, t0L = XYT.newCall(iL)
            print(x.shape,y.shape,n.shape)
            #print(XYT.iL)
            self.fit((x,n),y,batchSize=batchSize)
            if i%10==0:
                if len(xTest)>0:
                    lossTrain = self.evaluate(x=self.inx((xTrain,nTrain)),y=yTrain)
                    lossTest    = self.evaluate(x=self.inx((xTest,nTest)),y=yTest)
                    print('train loss',lossTrain,'test loss: ',lossTest,\
                        'sigma: ',XYT.timeDisKwarg['sigma'],\
                        'w: ',self.config.lossFunc.w, \
                        'no sampleRate:', 1 - np.sign(sampleDone).mean(),\
                        'sampleTimes',sampleDone.mean())
                    resStr+='\n %d train loss : %f valid loss :%f'%(i,lossTrain,lossTest)
                    trainTestLoss.append([i,lossTrain,lossTest])
                    if lossTest >= lossMin:
                        count -= 1
                    if lossTest > 3*lossMin:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if lossTest < lossMin:
                        count = count0
                        lossMin = lossTest
                        w0 = self.get_weights()
                        print('find better')
                    if count ==0:
                        break
                    #print(self.metrics)
                    
                    if i%30==0:
                        youtTrain = 0
                        youtTest  = 0
                        youtTrain = self.predict((xTrain,nTrain))
                        youtTest  = self.predict((xTest,nTest))
                        resStr+='\ntrain '+printRes_sq(yTrain, youtTrain)
                        resStr+='\ntest '+printRes_sq(yTest, youtTest)
            #if i%5==0:
            #    print('learning rate: ',self.optimizer.lr)
            #    K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.95)
            #if i>10 and i%5==0:
            #    perN += int(perN*0.05)
            #    perN = min(1000, perN)
        self.set_weights(w0)
        return resStr,trainTestLoss
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(19),fileStr='',\
        level=-1):
        y = self.predict(x)
        f = 1/T
        count = x.shape[1]
        for i in range(len(x)):
            #print('show',i)
            timeL = np.arange(count)*delta
            if len(time0L)>0:
                timeL+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            xlimNew=[0,500]
            #xlim=xlimNew
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0)
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
            plt.pcolor(timeL,f,y0[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            plt.plot(timeL[pos.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,3)
            plt.pcolor(timeL,f,y[i,:,level,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            #plt.clim(0,1)
            plt.plot(timeL[pos0.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,4)
            delta = timeL[1] -timeL[0]
            N = len(timeL)
            fL = np.arange(N)/N*1/delta
            for j in range(x.shape[-1]):
                spec=np.abs(np.fft.fft(self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0])).reshape([-1])
                plt.plot(fL,spec/(spec.max()+1e-16),'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            plt.xlabel('f/Hz')
            plt.ylabel('A')
            plt.xlim([fL[1],fL[-1]/2])
            #plt.gca().semilogx()
            plt.savefig('%s%s_%d_%d.jpg'%(outputDir,fileStr,level,i),dpi=200)
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
        '''
