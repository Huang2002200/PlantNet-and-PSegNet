"""
Created on MON SEB 23 16:21:12 2022

@author: YC-W
Randomly sample 4096 points at the edge points and 4096 points at the center points,
and merge the collected points into one point cloud file
(repeat the sampling if there are less than 4096 points)
"""
import os
from os import listdir, path
import numpy as np

path_center = '/core_points_save_path'  # your directory path
path_edge = '/edge_points_save_path'
save_path = '/merge'

txt_cs = [f for f in listdir(path_center)
        if f.endswith('.txt') and path.isfile(path.join(path_center, f))]

txt_es = [f for f in listdir(path_edge)
        if f.endswith('.txt') and path.isfile(path.join(path_edge, f))]

#随机种子的选择
i = 0

for txt_c, txt_e in zip(txt_cs, txt_es):
        #暂时存储打乱顺序后的两部分点云
        c_temp = []
        e_temp = []
        end_temp = []
        i = i + 1
        with open(os.path.join(path_center, txt_c), 'r') as f:
                index = []
                lines = f.readlines()
                size_c = len(lines)
                #随机取出4096个点
                np.random.seed(i)
                indexs = np.random.randint(0, size_c, 4096, int)
                for index in indexs:
                        c_temp.append(lines[index])
        with open(os.path.join(path_edge, txt_e), 'r') as f:
                index = []
                lines = f.readlines()
                size_e = len(lines)
                np.random.seed(i)
                indexs = np.random.randint(0, size_e, 4096, int)
                for index in indexs:
                        e_temp.append(lines[index])

        end_temp = c_temp + e_temp

        #save
        output_file = os.path.join(save_path, os.path.splitext(txt_e)[0] + "m.txt")
        with open(output_file, 'w') as f:
                f.write(''.join(end_temp[0:]))