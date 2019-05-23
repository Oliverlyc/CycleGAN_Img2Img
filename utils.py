# -*- coding: utf-8 -*-

"""
Created on Thu Apr 11 10:30:27 2019

@author: Oliver Lin
The Best or nothing!!!
"""
#%matplotlib inline
from glob import glob
from PIL import Image
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

train_X = r'./origin_data/horse2zebra/trainA'
train_Y = r'./origin_data/horse2zebra/trainB'
test_X = r'./origin_data/horse2zebra/testA'
test_Y = r'./origin_data/horse2zebra/testB'

class Dataloader(object):
    def __init__(self):    
        self.img_H = 256
        self.img_W = 256
        self.img_C = 3
        
    def load_dataset(self, path):
        path_list = []
        current_dir = os.path.dirname(__file__)
        for file in os.listdir(path):
            file_path = os.path.join(current_dir, path[2:], file)
            path_list.append(file_path)
        
        return path_list
        pass
    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('E:\PythonProject\Cycle_GAN_Image_to_Image\origin_data/%s/%s/*' % ('horse2zebra', data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.read_img(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, (self.img_W, self.img_H))

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, (self.img_W, self.img_H))
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs
    def load_batch(self, batch_size=1, for_testing = False):
        if for_testing:
            X_files = self.load_dataset(test_X)
            Y_files = self.load_dataset(test_Y)
            self.n_batch = int(min(len(X_files), len(Y_files))/batch_size)
        else:
            X_files = self.load_dataset(train_X)
            Y_files = self.load_dataset(train_Y)
            self.n_batch = int(min(len(X_files), len(Y_files))/batch_size)
        
    
        for i in range(self.n_batch - 1):
            batch_X = X_files[i * batch_size:(i+1) * batch_size]
            batch_Y = Y_files[i * batch_size:(i+1) * batch_size]
            imgs_X, imgs_Y = [], []
            for path_X, path_Y in zip(batch_X, batch_Y):
                img_X = np.array(self.read_img(path_X).resize((self.img_W, self.img_H)))
                img_Y = np.array(self.read_img(path_Y).resize((self.img_W, self.img_H)))
                
                if not for_testing and np.random.random() > 0.5:
                    img_X = np.fliplr(img_X)#图像增强，翻转
                    img_Y = np.fliplr(img_Y)
                
#                imgs_X.append(img_X/255.0)
#                imgs_Y.append(img_Y/255.0)
                imgs_X.append(img_X/127.5 - 1)
                imgs_Y.append(img_Y/127.5 - 1)
#            imgs_X = [np.stack(imgs_X, axis=0)]
#            imgs_Y = [np.stack(imgs_Y, axis=0)]
            #使用sigmoid函数，像素缩放到-1~1
            yield np.array(imgs_X), np.array(imgs_Y)
#            yield [np.stack(imgs_X,axis=0)], [np.stack(imgs_Y, axis=0)]
        pass
    def read_img(self, image):
        return Image.open(image).convert('RGB')
    
    #为了增强鉴别器的稳定性,使用历史生成的假图像，以防止出现忘记的现象，此为图片缓存区
class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.imgs = []
        pass
    
    def action(self, imgs):
        if self.pool_size == 0:
            return imgs
        return_imgs = []
        for index in range((imgs.shape)[0]):
            if self.num_imgs < self.pool_size:
                self.pool_size += 1
                self.imgs.append(imgs[index])
                return_imgs.append(imgs[index])
            else:
                if np.random.random() > 0.5:
                    #如果随机数大于0.5从缓存区读取图片，并替换
                    tmp_index = np.random.randint(0, self.pool_size - 1)
                    tmp_img = self.imgs[tmp_index]
                    self.imgs[tmp_index] = imgs[index]
                    return_imgs.append(tmp_img)
                else:
                    return_imgs.append(imgs[index])
#        return_imgs = np.stack(return_imgs, axis=0)
        return np.array(return_imgs)
        pass
if __name__ == "__main__":

    dataloader = Dataloader()
    img = dataloader.load_data(domain='A', batch_size=1, is_testing=True)
    img = 0.5 * img[0] + 0.5
    plt.imshow(img)