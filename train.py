# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:29:57 2019

@author: Oliver Lin
The Best or nothing!!!
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:30:10 2019

@author: Oliver Lin
The Best or nothing!!!
"""
import os
import time
import numpy as np
import keras.backend as K
from PIL import Image
from keras.models import Model, load_model
from keras.layers import Lambda, Input
from keras.optimizers import Adam
from model import CycleGAN
from keras.utils import plot_model
from utils import Dataloader,ImagePool

def gan_loss(output, target, use_lsgan=False):
    #使用lsgan的目的是为了解决生成图片质量和质量不高以及训练过程不稳定的问题
    if not use_lsgan:
        diff = output - target
        dims = list(range(1, K.ndim(diff)))
        return K.expand_dims(K.mean(K.square(diff), dims),0)
        pass
    else:
        #交叉熵损失
        return -K.mean(K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))
    pass

def cycle_loss(reconstruct, real):
    diff = K.abs(reconstruct - real)
    dims = list(range(1, K.ndim(diff)))
#    return K.expand_dims(K.mean(diff, dims), 0)
#    print(diff)
    return K.expand_dims((K.mean(diff, dims)), 0)
    pass

def netG_loss(G_tensors, loss_weight=10):
    netD_X_predict_fake, reconstruct_X, G_X_input, netD_Y_predict_fake, reconstruct_Y, G_Y_input = G_tensors
    
    loss_G_Y = gan_loss(netD_X_predict_fake, K.ones_like(netD_X_predict_fake))#fake越来越像real
    loss_cyc_X = cycle_loss(reconstruct_X, G_X_input)
    
    loss_G_X = gan_loss(netD_Y_predict_fake, K.ones_like(netD_Y_predict_fake))
    loss_cyc_Y = cycle_loss(reconstruct_Y, G_Y_input)
    
    loss_G = loss_G_X + loss_G_Y + loss_weight * (loss_cyc_X + loss_cyc_Y)
    return loss_G
#    return K.reshape(loss_G,shape=(1,1))
    pass

def netD_loss(D_list):
    netD_predict_real, netD_predict_fake = D_list
    netD_loss_real = gan_loss(netD_predict_real, K.ones_like(netD_predict_real))
    netD_loss_fake = gan_loss(netD_predict_fake, K.zeros_like(netD_predict_fake))
    
    loss_D = (1/2) * (netD_loss_fake + netD_loss_real)
    #不可以返回一个数字
    return loss_D
    pass

#获取生成图片
def get_G_function(netG):
    real_input = netG.inputs[0]
    fake_output = netG.outputs[0]
    
    function = K.function([real_input], [fake_output])
    return function
    pass

def save_image(file_name,image):
    print('Save {}.jpg'.format(file_name))
    image = (image[0]+1) * 127.5
    img = Image.fromarray(image.astype('uint8')).convert("RGB") 
    img.save('./pics/'+ file_name + ".jpg")
    pass

def train(epochs=100, batch_size=1):
    #生成器
#    img_shape = (256, 256, 3)
    netG = CycleGAN()
    netG_XY, real_X, fake_Y = netG.generator()
    netG_YX, real_Y, fake_X = netG.generator()

    reconstruct_X = netG_YX(fake_Y)
    reconstruct_Y = netG_XY(fake_X)
    #鉴别器
    netD = CycleGAN()
    netD_X = netD.discriminator()
    netD_Y = netD.discriminator()
    
    netD_X_predict_fake = netD_X(fake_X)
    netD_Y_predict_fake = netD_Y(fake_Y)
    netD_X_predict_real = netD_X(real_X)
    netD_Y_predict_real = netD_Y(real_Y)
#    netD_X.summary()
    #优化器
    optimizer = Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.01)
#    netG_XY.summary()
#    plot_model(netG_XY, to_file='./netG_XY_model_graph.png')
    #GAN
    netD_X.trainable = False#冻结
    netD_Y.trainable = False
    netG_loss_inputs = [netD_X_predict_fake, reconstruct_X, real_X, netD_Y_predict_fake, reconstruct_Y, real_Y]
    netG_train = Model([real_X, real_Y], Lambda(netG_loss)(netG_loss_inputs))
    netG_train.compile(loss='mae', optimizer=optimizer, metrics = ['accuracy'])
    

    _fake_X_inputs = Input(shape=(256,256,3))
    _fake_Y_inputs = Input(shape=(256,256,3))
    _netD_X_predict_fake = netD_X(_fake_X_inputs)
    _netD_Y_predict_fake = netD_Y(_fake_Y_inputs)
    netD_X.trainable = True
    netD_X_train = Model([real_X, _fake_X_inputs], Lambda(netD_loss)([netD_X_predict_real,  _netD_X_predict_fake]))
    netD_X_train.compile(loss='mae', optimizer=optimizer, metrics = ['accuracy'])#均方误差
    
    netD_X.trainable = False
    netD_Y.trainable = True
    netD_Y_train = Model([real_Y, _fake_Y_inputs], Lambda(netD_loss)([netD_Y_predict_real, _netD_Y_predict_fake]))
    netD_Y_train.compile(loss='mae', optimizer=optimizer, metrics = ['accuracy'])
    
    dataloader = Dataloader()
    fake_X_pool = ImagePool()
    fake_Y_pool = ImagePool()
    
    netG_X_function = get_G_function(netG_XY)
    netG_Y_function = get_G_function(netG_YX)
    if len(os.listdir('./weights')):
        netG_train.load_weights('./weights/netG.h5')
        netD_X_train.load_weights('./weights/netD_X.h5')
        netD_Y_train.load_weights
    
    print('Info: Strat Training\n')
    for epoch in range(epochs):
        
        target_label = np.zeros((batch_size, 1))
        
        for batch_i, (imgs_X, imgs_Y) in enumerate(dataloader.load_batch(batch_size)):
            start_time = time.time()
            num_batch = 0
            tmp_fake_X = netG_X_function([imgs_X])[0]
            tmp_fake_Y = netG_Y_function([imgs_Y])[0]
            
            #从缓存区读取图片
            _fake_X = fake_X_pool.action(tmp_fake_X)
            _fake_Y = fake_Y_pool.action(tmp_fake_Y)
            if epoch:
                save_image('fake_X_'+str(epoch), _fake_X[0])
                save_image('fake_Y_'+str(epoch), _fake_Y[0])
            _netG_loss = netG_train.train_on_batch([imgs_X, imgs_Y], target_label)
            netD_X_loss = netD_X_train.train_on_batch([imgs_X, _fake_X], target_label)
            netD_Y_loss = netD_Y_train.train_on_batch([imgs_Y, _fake_Y], target_label)
            num_batch += 1
            diff = time.time() - start_time
            print('Epoch:{}/{},netG_loss:{}, netD_loss:{},{}, time_cost_per_epoch:{}/epoch'\
              .format(epoch+1, epochs, _netG_loss, netD_X_loss, netD_Y_loss, diff, diff/num_batch))
       
        netG_train.save_weights('./weights/netG.h5')
        netD_X_train.save_weights('./weights/netD_X.h5')
        netD_Y_train.save_weights('./weights/netD_Y.hs')
        print('Model saved!\n')
    pass

def test(CycleGAN,model_path, batch_size=1):
    print('Load Model:')
    netG = load_model(model_path)
    dateloader = Dataloader()
    for batch_i, (imgs_X, imgs_Y) in enumerate(dataloader.load_batch(batch_size,for_testing=True)):
        preds = netG.predict()
        pass
    pass
if __name__ == "__main__":
    train()
    pass