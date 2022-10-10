from utils.common import *
import tensorflow as tf
import numpy as np
import os
import random

class dataset:
    def __init__(self, dataset_dir_data, dataset_dir_labels):
        self.dataset_dir_data = dataset_dir_data
        self.dataset_dir_labels = dataset_dir_labels
        self.data = tf.convert_to_tensor([])
        self.labels = tf.convert_to_tensor([])
        self.data_file = os.path.join(self.dataset_dir_data, f"data.npy")
        self.labels_file = os.path.join(self.dataset_dir_labels, f"labels.npy")
        self.cur_idx = 0
    
    def generate(self, lr_crop_size, hr_crop_size, scale, samples):      
        if exists(self.data_file) and exists(self.labels_file):
            print(f"{self.data_file} and {self.labels_file} HAVE ALREADY EXISTED\n")
            return
        data = []
        labels = []
        step = hr_crop_size - 1

        ls_data = sorted_list(self.dataset_dir_data)
        ls_labels = sorted_list(self.dataset_dir_labels)
        
        for i in range(len(ls_data)):
            print(i)
            hr_image = read_image(ls_labels[i])
            hr_image = rgb2ycbcr(hr_image)

            lr_image = read_image(ls_data[i])
            lr_image = rgb2ycbcr(lr_image)
            # *Y chanel - shape = [h, w, 1]
            # hr_image = hr_image[:, :, 0, tf.newaxis]

            h = hr_image.shape[0]
            w = hr_image.shape[1]

            for i in range(samples):
                starting_y = random.randint(0, h - lr_crop_size)
                starting_x = random.randint(0, w - lr_crop_size)

                subim_label = hr_image[starting_y*scale:(starting_y+lr_crop_size)*scale, starting_x*scale:(starting_x+lr_crop_size)*scale]
                subim_data = lr_image[starting_y:starting_y+lr_crop_size, starting_x:starting_x+lr_crop_size]
                subim_data = gaussian_blur(subim_data, sigma=0.2)
                
                subim_label = norm01(subim_label)
                subim_data = norm01(subim_data)

                data.append(subim_data.numpy())
                labels.append(subim_label.numpy())

        data = np.array(data)
        labels = np.array(labels)
        data, labels = shuffle(data, labels)
        
        np.save(self.data_file, data)
        np.save(self.labels_file, labels)

    def load_data(self):
        if not exists(self.data_file):
            raise ValueError(f"\n{self.data_file} and {self.labels_file} DO NOT EXIST\n")
        self.data = np.load(self.data_file)
        self.data = tf.convert_to_tensor(self.data)
        self.labels = np.load(self.labels_file)
        self.labels = tf.convert_to_tensor(self.labels)
    
    def get_batch(self, batch_size, shuffle_each_epoch=True):
        if batch_size > self.data.shape[0]:
            return self.data, self.labels, True
        # Ignore remaining dataset because of  
        # shape error when run tf.reduce_mean()
        isEnd = False
        if self.cur_idx + batch_size > self.data.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                self.data, self.labels = shuffle(self.data, self.labels)
        
        data = self.data[self.cur_idx : self.cur_idx + batch_size]
        labels = self.labels[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size
        
        return data, labels, isEnd
