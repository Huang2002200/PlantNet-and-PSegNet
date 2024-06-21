import os
import shutil
import random

o_list = list(range(546))
select_list=random.sample(o_list, 182)

file_dir = ''
for root, dirs, files in os.walk(file_dir, topdown=False):
    print(root)     # 当前目录路径
    print(dirs)     # 当前目录下所有子目录
    print(files)        # 当前路径下所有非目录子文件

number=1820
filter=[0]*1820
k = 0
for i in range(182):
    a=select_list[i]*10
    filter[k] = a
    for j in range(10):
        a = files[select_list[i]*10 + j]
        filter[k] = a
        k = k + 1

dir_root=''
dir_save='/test file'

for i in filter:
     # 目录的拼接
        full_path = os.path.join(dir_root,i)
        shutil.move(full_path, dir_save)
