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
from keras.initializers import RandomNormal
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
import keras.backend as K

class CycleGAN(object):
    def __init__(self):
        self.img_H = 256
        self.img_W = 256
        self.img_C = 3
        
        pass
    #卷积层
    def conv_block(self, x, filters, size, stride=(2,2),has_norm_instance=True,
                   padding='same',
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
        y = self.conv_block(x, filters, 3, (1, 1))
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
    def generator(self):
        inputs = Input(shape=(self.img_W,self.img_H, self.img_C))
        x = inputs
        if(self.img_H == 256):
            res_block = 9
        else:
            res_block = 6
        
        x = self.conv_block(x, 64, (7, 7),(1, 1))
        x = self.conv_block(x, 128, (3, 3),(2, 2))
        x = self.conv_block(x, 256, (3, 3), (2, 2))
        
        for i in range(res_block):
            x = self.residual_block(x)
        
        x = self.deconv_block(x, 128, 3)
        x = self.deconv_block(x, 64, 3)
        x = self.conv_block(x, 3, (7, 7), stride=(1, 1), has_norm_instance=False)
        outputs = x
        return Model(inputs=inputs, outputs=outputs), inputs, outputs
        pass
    def discriminator(self):
        inputs = Input(shape=(self.img_W, self.img_H, self.img_C))
        #PatchGAN
        patch = int(self.img_W/2**5)
        x = Cropping2D(cropping=(patch, patch), input_shape=(self.img_W, self.img_H, self.img_C))(inputs)
        x = self.conv_block(x, 64, (4, 4), has_norm_instance=False, use_leaky_relu=True)
        x = self.conv_block(x, 128, (4, 4), use_leaky_relu=True)
        x = self.conv_block(x, 256, (4, 4), use_leaky_relu=True)
        x = self.conv_block(x, 512, (4, 4), use_leaky_relu=True)
        x = Conv2D(1, (4, 4), activation='sigmoid', strides=(1, 1))(x)
#        x = Dense(1, activation='sigmoid')(x)
        outputs = x
        return Model(inputs=inputs, outputs=outputs)
#        inputs = Input(shape=(self.img_W, self.img_H, self.img_C))
#        x = inputs
#        ndf = 64
#        hidden_layers = 3
#        x = ZeroPadding2D(padding=(1, 1))(x)
#        x = self.conv_block(x, ndf, 4, has_norm_instance=False, use_leaky_relu=True, padding='valid')
#
#        x = ZeroPadding2D(padding=(1, 1))(x)
#        for i in range(1, hidden_layers + 1):
#            nf = 2 ** i * ndf
#            x = self.conv_block(x, nf, 4, use_leaky_relu=True, padding='valid')
#            x = ZeroPadding2D(padding=(1, 1))(x)
#
#        x = Conv2D(1, (4, 4), activation='sigmoid', strides=(1, 1))(x)
#        outputs = x
#
#        return Model(inputs=[inputs], outputs=outputs)
        pass
    def build(self):
        pass