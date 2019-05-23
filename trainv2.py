# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:29:25 2019

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
from modelv2 import CycleGAN
from keras.utils import plot_model
from utils import Dataloader,ImagePool
import matplotlib.pyplot as plt
#获取生成图片
def get_G_function(netG):
    real_input = netG.inputs[0]
    fake_output = netG.outputs[0]
    
    function = K.function([real_input], [fake_output])
    return function
    pass

def save_generate_image(file_name,image):
    print('Save {}.jpg'.format(file_name))
#    image = (image[0]+1) * 127.5
    image = ((image[0]) * 255.0).clip(0, 255)
    img = Image.fromarray(image.astype('uint8')).convert("RGB") 
#    print(img)
    img.save('./pics/'+ file_name + ".jpg")
    pass

def main(epochs=100, batch_size=1, test=False):
    
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.01)
    patch = int(256/2**5)
    #鉴别器
    netD = CycleGAN()
    netD_X = netD.discriminator()
    netD_Y = netD.discriminator()
    netD_X.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    netD_Y.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    
    #生成器
#    img_shape = (256, 256, 3)
    netG = CycleGAN()
    netG_XY, real_X, fake_Y = netG.generator()
    netG_YX, real_Y, fake_X = netG.generator()
    netG_XY.summary()
    netG_YX.summary()
    reconstruct_X = netG_YX(fake_Y)
    reconstruct_Y = netG_XY(fake_X)
    
    netD_X.trainable = False
    netD_Y.trainable = False
    
    netD_X_predict_fake = netD_X(fake_X)
    netD_Y_predict_fake = netD_Y(fake_Y)
    
    #训练生成网络
    combined_model = Model(inputs=[real_X, real_Y], outputs=[netD_X_predict_fake, netD_Y_predict_fake, reconstruct_X, reconstruct_Y])
    combined_model.compile(loss=['mse', 'mse', 'mae', 'mae'], loss_weights=[1, 1, 10, 10], optimizer=optimizer)
    #plot_model(combined_model, to_file='./combined_model_graph.png')
    
    #准备数据
    dataloader = Dataloader()
#    fake_X_pool = ImagePool()
#    fake_Y_pool = ImagePool()
#    
#    netG_X_function = get_G_function(netG_XY)
#    netG_Y_function = get_G_function(netG_YX)
    combined_model.summary()
    if len(os.listdir('./weights')):
        print('Info: Load weights.')
        combined_model.load_weights('./weights/netG.h5')
        netD_X.load_weights('./weights/netD_X.h5')
        netD_Y.load_weights('./weights/netD_Y.h5')
#    netD_X.summary()
    if test:
        sample_images(netG_XY, netG_YX,0, 0)
        pass
    else:
        print('Info: Strat Training\n')
        for epoch in range(epochs):
        
            real_label = np.ones((batch_size,) + (patch, patch, 1))
            fake_label = np.zeros((batch_size,) + (patch, patch, 1))
            for batch_i, (imgs_X, imgs_Y) in enumerate(dataloader.load_batch(batch_size)):
                start_time = time.time()
            
#                tmp_fake_X = netG_X_function([imgs_X])[0]
#                tmp_fake_Y = netG_Y_function([imgs_Y])[0]
#                save_image('real_X'+str(epoch)+'_'+str(batch_i), imgs_X)
                tmp_fake_Y = netG_XY.predict(imgs_X)
                tmp_fake_X = netG_YX.predict(imgs_Y)
#                print(tmp_fake_X.shape)
                #从缓存区读取图片
#                _fake_X = fake_X_pool.action(tmp_fake_X)
#                _fake_Y = fake_Y_pool.action(tmp_fake_Y)
#                _netG_loss = netG_train.train_on_batch([imgs_X, imgs_Y], target_label)
                netD_X_real_loss = netD_X.train_on_batch(imgs_X, real_label)
                netD_X_fake_loss = netD_X.train_on_batch(tmp_fake_X, fake_label)
                netD_X_loss = 0.5 * np.add(netD_X_real_loss, netD_X_fake_loss)
                
                netD_Y_real_loss = netD_Y.train_on_batch(imgs_Y, real_label)
                netD_Y_fake_loss = netD_Y.train_on_batch(tmp_fake_Y, fake_label)
                netD_Y_loss = 0.5 * np.add(netD_Y_real_loss, netD_Y_fake_loss)
                
                netD_loss = 0.5 * np.add(netD_X_loss, netD_Y_loss)
                
                netG_loss = combined_model.train_on_batch([imgs_X, imgs_Y], [real_label, real_label, imgs_X, imgs_Y])
                diff = time.time() - start_time
                print('[Epoch:{}/{}],[Batch:{}],[netD_loss:{:.4f}, acc:{:.4f}], [netG_loss:{:.4f}, adv:{:.4f}, rec:{:.4f}], [time_cost_per_epoch:{:.2f}s/batch]'\
                      .format(epoch+1, epochs, batch_i, netD_loss[0], netD_loss[1]*100, netG_loss[0], np.mean(netG_loss[1:3]), np.mean(netG_loss[3:5]), diff))
                if batch_i %200 == 0:
                    sample_images(netG_XY, netG_YX,epoch, batch_i)
            combined_model.save_weights('./weights/netG.h5')
            netD_X.save_weights('./weights/netD_X.h5')
            netD_Y.save_weights('./weights/netD_Y.h5')
            print('Model saved!\n')
            pass

def sample_images(g_AB, g_BA, epoch, batch_i):
        os.makedirs('pics/', exist_ok=True)
        r, c = 2, 3
        data_loader = Dataloader()
        imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = g_BA.predict(fake_B)
        reconstr_B = g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
#        gen_imgs = (gen_imgs + 1) * 127.5
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("pics/%d_%d.png" % (epoch, batch_i))
        plt.close()
        
def test(CycleGAN,model_path, batch_size=1):
    print('Load Model:')
    netG = load_model(model_path)
    dateloader = Dataloader()
    for batch_i, (imgs_X, imgs_Y) in enumerate(dataloader.load_batch(batch_size,for_testing=True)):
        preds = netG.predict()
        pass
    pass
if __name__ == "__main__":
    main(test=True)
    pass