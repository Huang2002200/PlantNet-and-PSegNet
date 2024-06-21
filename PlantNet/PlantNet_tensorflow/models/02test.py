import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
from scipy import stats
from IPython import embed
import random

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
from model import *
from test_utils import *
from clustering import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--verbose', default=True, help='if specified, output color-coded seg obj files')
parser.add_argument('--log_dir', default='out', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--bandwidth', type=float, default=0.6, help='Bandwidth for meanshift clustering [default: 1.]')
parser.add_argument('--input_list', type=str, default='/PlantNet/data/test_file_list.txt', help='Input data list file')
parser.add_argument('--model_path', type=str, default='/PlantNet/models/epoch_199.ckpt', help='Path of model')
FLAGS = parser.parse_args()


BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
TEST_FILE_LIST = FLAGS.input_list
BANDWIDTH = FLAGS.bandwidth
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
mean_num_pts_in_group = np.loadtxt("/PlantNet/models/mean_ins_size.txt")#loadtxt(os.path.join(MODEL_PATH.split('/')[5], '../mean_ins_size.txt'))

output_verbose = FLAGS.verbose  # If true, output all color-coded segmentation obj files

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

OUTPUT_DIR = os.path.join(LOG_DIR, 'log_6_BS74_newdata')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

os.system('cp inference_merge.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_inference.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


NUM_CLASSES = 6

HOSTNAME = socket.gethostname()

ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(os.path.join(ROOT_DIR, FLAGS.input_list))]
len_pts_files = len(ROOM_PATH_LIST)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # add by Shi Guoliang 20201219
            ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph = \
                placeholder_onehot_inputs(BATCH_SIZE, NUM_POINT)
                
            # Get model and loss
            pred_sem, pred_ins, fuse_catch = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
            pred_sem_softmax = tf.nn.softmax(pred_sem)
            pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)

            loss, sem_loss, disc_loss, l_var, l_dist, l_reg, simmat_loss = \
                get_loss(pred_ins, labels_pl, pred_sem, sem_labels_pl, fuse_catch, ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph)


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        is_training = False
        loader = tf.train.Saver()
        # Restore variables from disk.
        loader.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sem_labels_pl': sem_labels_pl,
               'is_training_pl': is_training_pl,
               'ptsseglabel_ph':ptsseglabel_ph,
               'ptsgroup_label_ph':ptsgroup_label_ph,
               'pts_seglabel_mask_ph':pts_seglabel_mask_ph,
               'pts_group_mask_ph':pts_group_mask_ph,
               'pred_ins': pred_ins,
               'pred_sem_label': pred_sem_label,
               'pred_sem_softmax': pred_sem_softmax,
               'loss': loss,
               'l_var': l_var,
               'l_dist': l_dist,
               'l_reg': l_reg,
               'simmat_loss': simmat_loss}

#        total_acc = 0
#        total_seen = 0
        
        output_filelist_f = os.path.join(LOG_DIR, 'output_filelist.txt')
        fout_out_filelist = open(output_filelist_f, 'w')
        for shape_idx in range(len_pts_files):
            room_path = ROOM_PATH_LIST[shape_idx]
            log_string('%d / %d ...' % (shape_idx, len_pts_files))
            log_string('Loading train file ' + room_path)
            out_data_label_filename = os.path.basename(room_path)[:-3] + str(shape_idx) + '_pred.txt'
            out_data_label_filename = os.path.join(OUTPUT_DIR, out_data_label_filename)
            out_gt_label_filename = os.path.basename(room_path)[:-3] + str(shape_idx) + '_gt.txt'
            out_gt_label_filename = os.path.join(OUTPUT_DIR, out_gt_label_filename)
            fout_data_label = open(out_data_label_filename, 'w')
            fout_gt_label = open(out_gt_label_filename, 'w')

            fout_out_filelist.write(out_data_label_filename+'\n')

            cur_data, cur_group, cur_sem, cur_obj = provider.load_h5_data_label_seg(room_path)
            cur_data = cur_data[:, :, :]
            cur_sem = np.squeeze(cur_sem)
            cur_group = np.squeeze(cur_group)
            cur_obj = np.squeeze(cur_obj)
            

            cur_pred_sem = np.zeros_like(cur_sem)
            cur_pred_sem_softmax = np.zeros([cur_sem.shape[0], cur_sem.shape[1], NUM_CLASSES])
            group_output = np.zeros_like(cur_group)
            group_obj = np.zeros_like(cur_obj)
            
            
            num_data = cur_data.shape[0]
            for j in range(num_data):
                log_string("Processsing: File [%d] Batch[%d]"%(shape_idx, j))

                pts = cur_data[j,...]
                group = cur_group[j,...]
                sem = cur_sem[j,...]
                obj = cur_obj[j,...]

                pts_group_label, pts_group_mask = convert_groupandcate_to_one_hot(cur_group[j:j+1])

                feed_dict = {ops['pointclouds_pl']: np.expand_dims(pts, 0),
                             ops['labels_pl']: np.expand_dims(group, 0),
                             ops['sem_labels_pl']: np.expand_dims(sem, 0),
                             ops['ptsgroup_label_ph']: pts_group_label,
                             ops['pts_group_mask_ph']: pts_group_mask,
                             ops['is_training_pl']: is_training}

                loss_val, l_var_val, l_dist_val, l_reg_val, pred_ins_val, pred_sem_label_val, pred_sem_softmax_val = sess.run(
                    [ops['loss'], ops['l_var'], ops['l_dist'], ops['l_reg'], ops['pred_ins'], ops['pred_sem_label'], ops['pred_sem_softmax']],
                    feed_dict=feed_dict)
                pred_val = np.squeeze(pred_ins_val, axis=0)# data_number x 5
                pred_sem = np.squeeze(pred_sem_label_val, axis=0)
                pred_sem_softmax = np.squeeze(pred_sem_softmax_val, axis=0)
                cur_pred_sem[j, :] = pred_sem
                cur_pred_sem_softmax[j, ...] = pred_sem_softmax# batch_size x data_number x num_class
                
                # cluster
                bandwidth = BANDWIDTH
                num_clusters, labels, cluster_centers = cluster(pred_val, bandwidth)

                groupids_block = labels
                group_obj[j,:] = obj
                
                un = np.unique(groupids_block)
                pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
                group_pred_final = -1 * np.ones_like(groupids_block)
                grouppred_cnt = 0
                for ig, g in enumerate(un): #each object in prediction
                    if g == -1:
                        continue
                    tmp = (groupids_block == g)
                    sem_seg_g = int(stats.mode(pred_sem[tmp])[0])
                    if np.sum(tmp) > 0.01 * mean_num_pts_in_group[sem_seg_g]:
                        group_pred_final[tmp] = grouppred_cnt
                        pts_in_pred[sem_seg_g] += [tmp]
                        grouppred_cnt += 1
                
                group_output[j, :] = group_pred_final

            group_pred = group_output.reshape(-1)
            seg_pred = cur_pred_sem.reshape(-1)
            seg_pred_softmax = cur_pred_sem_softmax.reshape([-1, NUM_CLASSES])
            pts = cur_data.reshape([-1, 3])
            
            obj_gt = group_obj.reshape(-1)
            seg_gt = cur_sem.reshape(-1)

            if output_verbose:
                ins = group_pred.astype(np.int32)
                sem = seg_pred.astype(np.int32)
                sem_softmax = seg_pred_softmax
                sem_gt = seg_gt
                ins_gt = cur_group.reshape(-1)
                for i in range(pts.shape[0]):
                    fout_data_label.write('%f %f %f %f %d %d\n' % (
                    pts[i, 0], pts[i, 1], pts[i, 2], sem_softmax[i, sem[i]], sem[i], ins[i]))   #预测txt的文件组成形式
                    fout_gt_label.write('%d %d %d\n' % (sem_gt[i], ins_gt[i], obj_gt[i]))  #gt txt的文件组成形式

            fout_data_label.close()
            fout_gt_label.close()


        fout_out_filelist.close()

if __name__ == "__main__":
    test()
    LOG_FOUT.close()
