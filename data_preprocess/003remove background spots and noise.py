"""
Created on MON SEB 23 16:21:12 2022

@author: YC-W
"""
import os
from os import listdir, path


path_str = ''  #  directory path
save_path_str = ''

txts = [f for f in listdir(path_str)
        if f.endswith('.txt') and path.isfile(path.join(path_str, f))]

for txt in txts:
    with open(os.path.join(path_str, txt), 'r') as f:

        index = []
        lines = f.readlines()
        for line in lines:
            if(line.split()[0:3] != ['0', '0', '0']):   # remove [0, 0, 0] or any other point did not unreasonable
                index.append(line)
    with open(os.path.join(save_path_str,os.path.splitext(txt)[0]+".txt"), 'w') as f:
        f.write(''.join(index[0:]))