#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四种植物，保存obj标签
"""
import os
import sys
import numpy as np
import h5py

def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:, 0:3]
    ins_label = (data[:, 3]).astype(int)
    sem_label = np.zeros((data.shape[0]), dtype=int)
    obj_label = data[:, 4]
    return point_xyz, ins_label, sem_label, obj_label


def change_scale(data):

    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    data[:, 0:3] = data[:, 0:3] - xyz_move
    # scale
    scale = np.max(data[:, 0:3])
    return data[:, 0:3] / scale


if __name__ == "__main__":
    DATA_ALL = []
    num_sample = 4096
    base_path = '/train'
    DATA_FILES = os.listdir(base_path)
    for fn in range(len(DATA_FILES)):
        # loadDataFile 返回点云坐标 实例标签 全零的语义标签
        current_data, current_ins_label, current_sem_label, current_obj_label = loadDataFile(os.path.join(base_path, DATA_FILES[fn]))
        # create current_sem_label
        if DATA_FILES[fn].split("_")[0] == 'sorghum':
            current_sem_label[np.where(current_ins_label == 0)] = 0
            current_sem_label[np.where(current_ins_label >= 1)] = 1
        if DATA_FILES[fn].split("_")[0] == 'benthi':
            current_sem_label[np.where(current_ins_label == 0)] = 2
            current_sem_label[np.where(current_ins_label >= 1)] = 3
        if DATA_FILES[fn].split("_")[0] == 'm82D':
            current_sem_label[np.where(current_ins_label == 0)] = 4
            current_sem_label[np.where(current_ins_label >= 1)] = 5
        change_data = change_scale(current_data)
        data_label = np.column_stack((change_data, current_ins_label, current_sem_label, current_obj_label))
        DATA_ALL.append(data_label)

    output = np.vstack(DATA_ALL)
    output = output.reshape(-1, num_sample, 6)
    f = h5py.File("/save_path/save_file_name", "w")
    f['data'] = output[:, :, 0:3]
    f['pid'] = output[:, :, 3]  # ins label
    f['seglabel'] = output[:, :, 4]
    f['obj'] = output[:, :, 5]

    f.close()
