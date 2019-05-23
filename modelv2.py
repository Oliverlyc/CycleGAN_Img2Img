# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:38:07 2019

@author: Oliver Lin
The Best or nothing!!!
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:08:09 2019

@author: Oliver Lin
The Best or nothing!!!
"""
from keras.models import Model
from keras.layers import Conv2D,ZeroPadding2D,Input,Add,Cropping2D,Dense
from keras.layers import LeakyReLU, Activation
from keras.layers import Conv2DTranspose
from keras.layers.core import Lambda
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.activations import tanh
import keras.backend as K
import tensorflow as tf

class CycleGAN(object):
    def __init__(self):
        self.img_H = 256
        self.img_W = 256
        self.img_C = 3
        
        pass
    #卷积层
    def conv_block(self, x, filters, size, stride=(2,2),has_norm_instance=True,
                   padding='valid',
                   has_activation_layer=True,
                   use_leaky_relu=False):
        x = Conv2D(filters, size, strides = stride, padding=padding, kernel_initializer=RandomNormal(0, 0.02))(x)
        
        if has_norm_instance:
            x = InstanceNormalization(axis=1)(x)
            
        if has_activation_layer:
            if use_leaky_relu:
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Activation('relu')(x)
        return x
        pass
    #res层
    def residual_block(self, x, filters=256):
        y = self.reflect_padding(x, 1)
        y = self.conv_block(x, filters, 3, (1, 1), padding='valid')
        y = self.reflect_padding(x, 1)
        y = self.conv_block(y, filters, 3, (1, 1), has_activation_layer=False)
        y = Add()([x, y])
        return y
        pass
    
    def deconv_block(self, x, filters, size ):
        x = Conv2DTranspose(filters, kernel_size=size, strides=2, padding='same', use_bias=False, kernel_initializer=RandomNormal(0, 0.02))(x)
        x = InstanceNormalization(axis=1)(x)
        x = Activation('relu')(x)
        return x
        pass
    def reflect_padding(self, x, kernel_size):
        x = Lambda(lambda x :tf.pad(x, [[0, 0],[kernel_size, kernel_size],[kernel_size, kernel_size],
                              [0, 0]], "REFLECT"))(x)
        return x
        pass
    def generator(self):
        inputs = Input(shape=(self.img_W,self.img_H, self.img_C))
        x = self.reflect_padding(inputs, 3)
        if(self.img_H == 256):
            res_block = 9
        else:
            res_block = 6
        
        x = self.conv_block(x, 64, (7, 7),(1, 1))
        x = self.conv_block(x, 128, (3, 3),(2, 2), padding='same')
        x = self.conv_block(x, 256, (3, 3), (2, 2), padding='same')
        
        for i in range(res_block):
            x = self.residual_block(x)
        
        x = self.deconv_block(x, 128, 3)
        x = self.deconv_block(x, 64, 3)
        x = self.conv_block(x, 3, (7, 7), stride=(1, 1), padding='same', has_norm_instance=False, has_activation_layer=False)
#        x = tanh(x)
        outputs = x
        return Model(inputs=inputs, outputs=outputs), inputs, outputs
        pass
    def discriminator(self):
        inputs = Input(shape=(self.img_W, self.img_H, self.img_C))
        #PatchGAN
        x = inputs
#        patch = int(self.img_W/2**4)
#        x = Cropping2D(cropping=(patch, patch), input_shape=(self.img_W, self.img_H, self.img_C))(inputs)
        x = self.conv_block(x, 64, (4, 4), padding='same', has_norm_instance=False, use_leaky_relu=True)
        x = self.conv_block(x, 128, (4, 4), padding='same', use_leaky_relu=True)
        x = self.conv_block(x, 256, (4, 4), padding='same', use_leaky_relu=True)
        x = self.conv_block(x, 512, (4, 4), padding='same', use_leaky_relu=True)
        x = self.conv_block(x, 1, (4, 4), padding='same', has_norm_instance=False, has_activation_layer=False)
#        x = Conv2D(1, (4, 4), activation='sigmoid', strides=(1, 1), padding='same')(x)
        outputs = x
        return Model(inputs=inputs, outputs=outputs)
        pass
    def build(self):
        pass