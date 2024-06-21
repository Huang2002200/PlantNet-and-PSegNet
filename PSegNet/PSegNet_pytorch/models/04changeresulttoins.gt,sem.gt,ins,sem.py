# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:28:20 2021

@author: JS-L
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:27:49 2021

@author: JS-L
"""
###########################此程序专门针对PlantNet跑出的结果，使其保存为一个个的txt文件，分别为gt-语义，gt-实例，pred-语义，pred-实例
import numpy as np
import os

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

def file_sorting(files):
    n = len(files)
    for i in range(n):
        for j in range(n-i-1):
            if int(files[j].split('_')[0]) > int(files[j+1].split('_')[0]):
                files[j], files[j+1] = files[j+1], files[j]
    return files

data=np.loadtxt("/test_pred.txt")  # 获取预测的数据 predict data
gtdata=np.loadtxt("/test_gt.txt")  # 获取gt  ground truth data
pcd=[]
seg=[]
ins=[]
gt=[]
gt_ins=[]
gt_seg=[]

file_path="/test"  # origin test data
ins_path="/test_produce/predict/ins"  #save path
sem_path="/test_produce/predict/sem"
gt_ins_path="/test_produce/gt/ins_gt"
gt_sem_path="/test_produce/gt/sem_gt"
files=os.listdir(file_path)


for i in range(1820):  # 1820 is testdata number
    pcd=data[i*4096:((i+1)*4096)]
    gt=gtdata[i*4096:((i+1)*4096)]
    data1=pcd[:,:3]    # xyz坐标数据
    data2=pcd[:,4]      # 语义标签

    seg=np.concatenate((pcd[:,:3], pcd[:, 4].reshape(4096, 1)), axis=-1)
    np.savetxt(os.path.join(sem_path, files[i]), seg, fmt="%f %f %f %d", delimiter=" ")

    gt_seg=np.concatenate((pcd[:,:3],gt[:,0].reshape(4096,1)), axis=-1)
    np.savetxt(os.path.join(gt_sem_path,files[i]),gt_seg,fmt="%f %f %f %d",delimiter=" ")

    ins=np.concatenate((pcd[:,:3],pcd[:,5].reshape(4096,1)),axis=-1)
    np.savetxt(os.path.join(ins_path,files[i]),ins,fmt="%f %f %f %d",delimiter=" ")

    gt_ins=np.concatenate((pcd[:,:3],gt[:,1].reshape(4096,1)),axis=-1)
    np.savetxt(os.path.join(gt_ins_path,files[i]),gt_ins,fmt="%f %f %f %d",delimiter=" ")
