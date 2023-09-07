#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:33:09 2019

@author: yaching
"""

import os
import shutil

from os import walk
from os.path import join



input_path = "/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CWRU_data_image9bit_SNR-4"
output_path = "/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CWRU_data_image9bit_SNR-4/"

def read_image(path, output_path): 
    for root, dirs, files in walk(path):   
        for f in files:
            allImage = join(root, f)
            print(allImage)
            
            if 'Normal' in root:
                path = str(output_path) + "/0"
                if not os.path.isdir(path):
                    os.mkdir(path)
                shutil.copy(allImage, path)
            else:                  
                if 'B' in root:
                    if '007' in root:
                        path = str(output_path) + "/1"
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        shutil.copy(allImage, path)
                        
                    elif '014' in root:
                        path = str(output_path) + "/2"
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        shutil.copy(allImage, path)
                    elif '021' in root:
                        path = str(output_path) + "/3"
                        if not os.path.isdir(path):
                            os.mkdir(path)      
                        shutil.copy(allImage, path)
                        
                elif 'IR' in root:
                    if '007' in root:
                        path = str(output_path) + "/4"
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        shutil.copy(allImage, path)
                        
                    elif '014' in root:
                        path = str(output_path) + "/5"
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        shutil.copy(allImage, path)
                    elif '021' in root:
                        path = str(output_path) + "/6"
                        if not os.path.isdir(path):
                            os.mkdir(path)      
                        shutil.copy(allImage, path)   
                        
                elif 'OR' in root:
                    if '007' in root:
                        path = str(output_path) + "/7"
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        shutil.copy(allImage, path)
                        
                    elif '014' in root:
                        path = str(output_path) + "/8"
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        shutil.copy(allImage, path)
                    elif '021' in root:
                        path = str(output_path) + "/9"
                        if not os.path.isdir(path):
                            os.mkdir(path)      
                        shutil.copy(allImage, path)                           

    return str("End!")

saySomething = read_image(input_path, output_path)
print(saySomething)