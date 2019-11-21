# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os

def write_to_txt():
    dir_names = list()
    file_names = list()
    for dir_name, _, name in os.walk("./datasets/train"):
        dir_names.append(dir_name)
        file_names.append(name)
    
    f = open("training_list.txt", "w")
    for class_num in range(len(file_names)):
        print(class_num)
        for img_num in range(len(file_names[class_num])):
            if file_names[class_num][img_num].endswith(".JPEG"):
                img_path = os.path.join(dir_names[class_num], file_names[class_num][img_num])
                f.write(img_path+" "+str(class_num - 1)+"\n")
    
    f.close()

def parse_dataset():
    img_list = list()
    label_list = list()
    f = open("training_list.txt", "r")
    line = f.readline().splitlines()
    while(line):
        img_list.append(line[0].split(" ")[0])
        label_list.append(line[0].split(" ")[1])
        line = f.readline().splitlines()   
    f.close()
    return img_list, label_list
    
    
if __name__ == "__main__":
    write_to_txt()
