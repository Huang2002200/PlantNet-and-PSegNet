# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:36:12 2021

@author: JS-L
"""

import numpy as np
import os

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


DATA_FILES = get_filelist(path='/train')   # origin file path


for i in range(len(DATA_FILES)):
    temp=np.loadtxt(DATA_FILES[i])
    shape=temp.shape
    temp[:,3]=temp[:,3]-2  #ins label-2
    zeros=np.zeros((shape[0],shape[1]+1))
    filename = os.path.basename(DATA_FILES[i])
    zeros[:,:4]=temp[:,:4]
    if filename.split("_")[0]=="benthi" : # add object label
        zeros[:,4]=0
    if filename.split("_")[0]=="m82D" :
        zeros[:,4]=1
    if filename.split("_")[0]=="sorghum":
         zeros[:,4]=2
    np.savetxt('save_path'+str(filename), zeros, fmt="%f %f %f %d %d")
    
    
    