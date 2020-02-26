# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2

class ILSVRC2012(object):
    def __init__(self, ILSVRC_dir, classname_file):
        self.dirname_to_classnum = dict()
        self.classnum_to_classname = dict()
        with open(classname_file, 'r') as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                self.dirname_to_classnum[line[:9]] = i
                self.classnum_to_classname[i] = line[10:]
        
        self.img_paths = list()
        self.labels = list()
        for root, _, names in os.walk(ILSVRC_dir):
            for name in names:
                if name.endswith("JPEG"):
                    self.img_paths.append(os.path.join(root, name))
                    self.labels.append(self.dirname_to_classnum[root.split('/')[-1]])
    
    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        img_list = np.array([]).reshape(0, 227, 227, 3)
        label_list = list()
        for i in index:
            img = cv2.imread(self.img_paths[i//(2 * 33 * 33)]) # crop 0~32 on height & width, reflection on horizontal
            
            img = self.PCA(img)
            h, w = img.shape[:2]
            min_ratio = np.max(256/np.array([h, w]))
            img_resize = cv2.resize(img, (int(min_ratio*w+0.5), int(min_ratio*h+0.5)))
            img_center = img_resize[img_resize.shape[0]//2-128:img_resize.shape[0]//2+128, 
                                    img_resize.shape[1]//2-128:img_resize.shape[1]//2+128]
            
            y_start = (i % 33**2) // 33
            x_start = (i % 33**2) % 33
            img_crop = img_center[y_start:y_start+224, x_start:x_start+224]
            reflect = i % (33 * 33 * 2) // (33 * 33)
            if reflect:
                img_crop = img_crop[:, ::-1, :]
            img_pad = np.pad(img_crop, ((2,1), (2,1), (0,0)), "constant")
            img_pad = img_pad / 255
            label = self.labels[i//(2 * 33 * 33)]
            img_list = np.concatenate([img_list, np.expand_dims(img_pad, axis=0)], axis=0)
            label_list.append(label)
        img_list = np.array(img_list)
        label_list = np.array(label_list)
        return img_list, label_list
                
    def __len__(self):
        return len(self.labels) * 33 * 33 * 2 # crop with x, y from 0 to 32 and reflection on horizontal.

    def PCA(self, img):
        img_avg = np.average(img, axis=(0, 1))
        img_std = np.std(img, axis=(0, 1))
        img_norm = (img - img_avg) / img_std
        img_cov = np.zeros((3, 3))
        for data in img_norm.reshape(-1, 3):
            img_cov += data.reshape(3, 1) * data.reshape(1, 3)
        img_cov /= len(img_norm.reshape(-1, 3))
        
        eig_values, eig_vectors = np.linalg.eig(img_cov)
        alphas = np.random.normal(0, 0.1, 3)
        img_reconstruct_norm = img_norm + np.sum((eig_values + alphas) * eig_vectors, axis=1)
        img_reconstruct = img_reconstruct_norm * img_std + img_avg
        img_reboundary = np.maximum(np.minimum(img_reconstruct , 255), 0).astype(np.uint8)
        return img_reboundary

