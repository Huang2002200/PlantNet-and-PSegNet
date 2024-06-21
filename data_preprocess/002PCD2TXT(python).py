# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:46:48 2019

@author: 426-4
"""

import os
from os import listdir, path

path_str = ''  #  directory path
pcds = [f for f in listdir(path_str)
        if f.endswith('.pcd') and path.isfile(path.join(path_str, f))]

save_path_str = ''
for pcd in pcds:
    with open(os.path.join(path_str, pcd), 'r') as f:
        lines = f.readlines()

    with open(os.path.join(save_path_str,os.path.splitext(pcd)[0]+".txt"), 'w') as f:
        f.write(''.join(lines[11:]))