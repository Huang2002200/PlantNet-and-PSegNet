# To estimate the mean instance size of each class in training set
import os
import sys
import numpy as np
from scipy import stats
import argparse
import provider
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


parser = argparse.ArgumentParser()
parser.add_argument('--test_area', type=int, default=5, help='The areas except this one will be used to estimate the mean instance size')
FLAGS = parser.parse_args()

def estimate(area):
    LOG_DIR = '/models'.format(area)
    num_classes = 20
    file_path = "/data/train_file_list.txt"

    train_file_list = provider.getDataFiles(os.path.join(BASE_DIR,file_path))

    mean_ins_size = np.zeros(num_classes)
    ptsnum_in_gt = [[] for itmp in range(num_classes)]

    for h5_filename in train_file_list:
        cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
        for i in range(cur_data.shape[0]):
            cur_group_batch = np.reshape(cur_group[i,...], [-1]) #327680x1
            cur_sem_batch = np.reshape(cur_sem[i,...], [-1])
    
            un = np.unique(cur_group_batch)
            for ig, g in enumerate(un):
                tmp = (cur_group_batch == g)
                sem_seg_g = int(stats.mode(cur_sem_batch[tmp])[0])
                ptsnum_in_gt[sem_seg_g].append(np.sum(tmp))

    for idx in range(num_classes):
        if ptsnum_in_gt[idx] != []:
            mean_ins_size[idx] = np.mean(ptsnum_in_gt[idx]).astype(int)

    print(mean_ins_size)
    np.savetxt(os.path.join(LOG_DIR, 'mean_ins_size.txt'),mean_ins_size)


if __name__ == "__main__":
    estimate(FLAGS.test_area)
