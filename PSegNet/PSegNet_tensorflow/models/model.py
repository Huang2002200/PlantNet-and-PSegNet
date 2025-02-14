import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module_1, pointnet_fp_module, add_module
from loss import *

NUM_CATEGORY = 6
NUM_GROUPS = 40
def attention(point_cloud,out_channel,is_training,is_dist,bn_decay,name,attn_dropout=0.5):
   temperature=out_channel**0.5
   q_map=tf_util.conv1d(point_cloud,out_channel,1,padding='VALID',bn=True,is_training=is_training,is_dist=is_dist,scope=name+'1',bn_decay=bn_decay)
   k_map=tf_util.conv1d(point_cloud,out_channel,1,padding='VALID',bn=True,is_training=is_training,is_dist=is_dist,scope=name+'2',bn_decay=bn_decay)
   v_map=tf_util.conv1d(point_cloud,out_channel,1,padding='VALID',bn=True,is_training=is_training,is_dist=is_dist,scope=name+'3',bn_decay=bn_decay)
   attn=tf.matmul(tf.transpose(q_map,[0,2,1]) / temperature,k_map)#tf.transpose(q_map,[0,2,1,3])
   attn=tf.nn.softmax(attn,axis=-1)
   attn=tf_util.dropout(attn,keep_prob=attn_dropout,is_training=is_training,scope=name+'4')
   y=tf.matmul(attn,tf.transpose(v_map,[0,2,1]))
   return tf.transpose(y,[0,2,1])

def attention1(channels,points,is_training,is_dist,bn_decay):
        points=tf.transpose(points, [0,2,1])
        q_conv = tf_util.conv1d(points, channels//4, 1, padding='VALID', bn=True, 
                                is_training=is_training, is_dist=is_dist, scope='attention1', bn_decay=bn_decay)
        x_q=tf.transpose(q_conv, [0,2,1])
        
        
        k_conv = tf_util.conv1d(points, channels // 4, 1, padding='VALID', bn=True,
                                is_training=is_training, is_dist=is_dist, scope='attention2', bn_decay=bn_decay)
        x_k=k_conv
        
        # self.q_conv.conv.weight = self.k_conv.conv.weight 
        v_conv =tf_util.conv1d(points, channels,1, padding='VALID', bn=True,
                              is_training=is_training, is_dist=is_dist, scope='attention3', bn_decay=bn_decay)
        x_v=v_conv
        
        energy = tf.matmul(x_q, x_k)
        
        attention =tf.nn.softmax(energy,dim=-1)
        attention = attention / (1e-9 + tf.reduce_sum(attention,axis=1, keepdims=True))
        x_r = tf.matmul(x_v, attention)
        
        trans_conv = tf_util.conv1d(points-x_r, channels,1, padding='VALID', bn=True,
                              is_training=is_training, is_dist=is_dist, scope='attention4', bn_decay=bn_decay)
        x_r = tf.relu(tf.BatchNorm1d(trans_conv,channels))
        
        points = points + x_r
        return points
        # return attention


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl

def placeholder_onehot_inputs(batch_size, num_point, num_group = NUM_GROUPS, num_cate = NUM_CATEGORY):
    pts_seglabels_ph = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_cate))
    pts_grouplabels_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_group))
    pts_seglabel_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    pts_group_mask_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    return pts_seglabels_ph, pts_grouplabels_ph, pts_seglabel_mask_ph, pts_group_mask_ph

def convert_seg_to_one_hot(labels):
    # labels:BxN
    labels = labels.astype(int)
    label_one_hot = np.zeros((labels.shape[0], labels.shape[1], NUM_CATEGORY))
    pts_label_mask = np.zeros((labels.shape[0], labels.shape[1]))

    un, cnt = np.unique(labels, return_counts=True)
    label_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in label_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(labels.shape[0]):
        for jdx in range(labels.shape[1]):
            if labels[idx, jdx] != -1:
                label_one_hot[idx, jdx, labels[idx, jdx]] = 1
                # pts_label_mask[idx, jdx] = float(totalnum) / float(label_count_dictionary[labels[idx, jdx]]) # 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum
                pts_label_mask[idx, jdx] = 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum

    return label_one_hot, pts_label_mask


def convert_groupandcate_to_one_hot(grouplabels):
    # grouplabels: BxN
    grouplabels = grouplabels.astype(int)
    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))

    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in group_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                group_one_hot[idx, jdx, grouplabel_dictionary[grouplabels[idx, jdx]]] = 1
                # pts_group_mask[idx, jdx] = float(totalnum) / float(group_count_dictionary[grouplabels[idx, jdx]]) # 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum
                pts_group_mask[idx, jdx] = 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

    return group_one_hot, pts_group_mask

def get_model(point_cloud, is_training, num_class, bn_decay=None, is_dist=False):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, :3]
    end_points['l0_xyz'] = l0_xyz
    
    # l0_points = add_module(l0_points, mlp0=None, mlp1=[8,8,16], mlp2=[16, 16], k=64, d=3, is_training=is_training, bn_decay=bn_decay, scope='AddlLayer')
    
    # Layer 1
    # l5_xyz, l5_points = pointnet_sa_module_1(l0_xyz, l0_points, npoint=2048, mlp1=[32, 32, 64], k=16, d=1, mlp2=None,
    #                                          is_training=is_training, bn_decay=bn_decay, scope='layer5')
    l1_xyz, l1_points = pointnet_sa_module_1(l0_xyz, l0_points, npoint=1024, mlp1=[32,32,64],k=16,d=1, mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer1')           
    l2_xyz, l2_points = pointnet_sa_module_1(l1_xyz, l1_points, npoint=256, mlp1=[64,64,128],k=16,d=1, mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points = pointnet_sa_module_1(l2_xyz, l2_points, npoint=128, mlp1=[128,128,256],k=8,d=1, mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points = pointnet_sa_module_1(l3_xyz, l3_points, npoint=128, mlp1=[256,256,512],k=8,d=1, mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points_sem = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='sem_fa_layer1')
    l2_points_sem = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_sem, [256,256], is_training, bn_decay, scope='sem_fa_layer2')
    l1_points_sem = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_sem, [256,128], is_training, bn_decay, scope='sem_fa_layer3')
    l0_points_sem = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_sem, [128,128,128], is_training, bn_decay, scope='sem_fa_layer4')

    # ins
    l3_points_ins = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='ins_fa_layer1')
    l1_points_ins = pointnet_fp_module(l1_xyz, l3_xyz, l1_points, l3_points_ins, [256,128], is_training, bn_decay, scope='ins_fa_layer3')
    l0_points_ins = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_ins, [128,128,128], is_training, bn_decay, scope='ins_fa_layer4')

    net_sem_cache = tf_util.conv1d(l0_points_sem, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache', bn_decay=bn_decay)
    net_ins_cache = tf_util.conv1d(l0_points_ins, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache', bn_decay=bn_decay)

    ins_avg=tf.reduce_mean(net_ins_cache, axis=-1, keep_dims=True, name='ins_reduce')
    sum_feature=tf.concat([net_sem_cache,ins_avg],axis=-1)
    sum_feature=tf_util.conv1d(sum_feature, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fusion', bn_decay=bn_decay)
    sum_feature=tf.concat([sum_feature,net_ins_cache],axis=-1)
    sum_feature=tf_util.conv1d(sum_feature, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fusion2', bn_decay=bn_decay)
    net_ins_2=sum_feature
    net_sem_2=sum_feature
   
   
    # net_fuse_cache = net_ins_cache + net_sem_cache
    # net_fuse_cache_1 = tf_util.conv1d(net_fuse_cache, 6, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sasdem_cache', bn_decay=bn_decay)
    
    # Similarity matrix
    Fsim = sum_feature
    r = tf.reduce_sum(Fsim * Fsim, 2)
    r = tf.reshape(r, [batch_size, -1, 1])
    D = r - 2 * tf.matmul(Fsim, tf.transpose(Fsim, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])
    simmat_logits = tf.maximum(10 * D, 0.)
    
    # Ins
    #net_ins_2 = tf.concat([l0_points_ins, net_ins_2], axis=-1, name='net_ins_2_concat')#################################shan
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten')
    # # net_ins_atten=attention(256,net_ins_2,is_training,is_dist,bn_decay)
    #net_ins_atten= tf.nn.softmax(net_ins_2, axis=1)
    net_ins_3 = net_ins_2 * net_ins_atten
    #net_ins_3=attention(net_ins_2,256,is_training,is_dist,bn_decay,'ins_attention',attn_dropout=0.5)

    # Sem
    #net_sem_2 = tf.concat([l0_points_sem, net_sem_2], axis=-1, name='net_ins_2_concat')##################################shan
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')
    # # net_sem_atten=attention(256,net_sem_2,is_training,is_dist,bn_decay)
    #net_sem_atten= tf.nn.softmax(net_sem_2, axis=1)
    net_sem_3 = net_sem_2 * net_sem_atten
    #net_sem_3 = attention(net_sem_2, 256, is_training, is_dist, bn_decay, 'sem_attention', attn_dropout=0.5)
    ##############################gai

    # Output
    #net_ins_3 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    # net_ins_atten= tf.nn.relu(tf.reduce_max(net_ins_3, axis=-2, keep_dims=True, name='ins_reduce2'))
    # 
    # net_ins_3 = net_ins_3 * net_ins_atten
    max_pool=tf.reduce_max(net_ins_3, axis=-2, keep_dims=True, name='max_pool')
    avg_pool=tf.reduce_mean(net_ins_3, axis=-2, keep_dims=True, name='avg_pool')
    max_pool=tf_util.conv1d(max_pool, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='max_pool1', bn_decay=bn_decay)
    avg_pool=tf_util.conv1d(avg_pool, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='avg_pool1', bn_decay=bn_decay)
    ins_fusion=max_pool+avg_pool
    net_ins_atten = tf.sigmoid(ins_fusion, name='ins_atten1')
    net_ins_3= net_ins_3 * net_ins_atten
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, 5, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')

    #net_sem_3 = tf_util.conv1d(net_sem_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    # net_sem_atten = tf.nn.relu(tf.reduce_max(net_sem_3, axis=-2, keep_dims=True, name='sem_reduce2'))
    # net_sem_3 = net_sem_3 * net_sem_atten
    max_pool=tf.reduce_max(net_sem_3, axis=-2, keep_dims=True, name='max_pool_2')
    avg_pool=tf.reduce_mean(net_sem_3, axis=-2, keep_dims=True, name='avg_pool_2')
    max_pool=tf_util.conv1d(max_pool, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='max_pool3', bn_decay=bn_decay)
    avg_pool=tf_util.conv1d(avg_pool, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='avg_pool3', bn_decay=bn_decay)
    sem_fusion=max_pool+avg_pool
    net_sem_atten = tf.sigmoid(sem_fusion, name='sem_atten1')
    net_sem_3 = net_sem_3 * net_sem_atten
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_sem_4, net_ins_4, simmat_logits


def get_loss(pred, ins_label, pred_sem, sem_label, sem_ins_fuse, pts_semseg_label, pts_group_label, pts_seg_label_mask, pts_group_mask):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    # double hinge loss add by Shi Guoliang 20201219
    group_mask = tf.expand_dims(pts_group_mask, dim=2)

    # pred_simmat = tf_util.pairwise_distance_l1(sem_ins_fuse)
    pred_simmat = sem_ins_fuse

    # Similarity Matrix loss
    B = pts_group_label.get_shape()[0]
    N = pts_group_label.get_shape()[1]

    onediag = tf.ones([B,N], tf.float32)

    group_mat_label = tf.matmul(pts_group_label,tf.transpose(pts_group_label, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j in the same group
    group_mat_label = tf.matrix_set_diag(group_mat_label,onediag)

    sem_mat_label = tf.cast(tf.matmul(pts_semseg_label,tf.transpose(pts_semseg_label, perm=[0, 2, 1])), tf.float32) #BxNxN: (i,j) if i and j are the same semantic category
    sem_mat_label = tf.matrix_set_diag(sem_mat_label,onediag)

    samesem_mat_label = sem_mat_label
    diffsem_mat_label = tf.subtract(1.0, sem_mat_label)

    samegroup_mat_label = group_mat_label
    diffgroup_mat_label = tf.subtract(1.0, group_mat_label)
    diffgroup_samesem_mat_label = tf.multiply(diffgroup_mat_label, samesem_mat_label)
    diffgroup_diffsem_mat_label = tf.multiply(diffgroup_mat_label, diffsem_mat_label)

    # Double hinge loss
    alpha = 10.
    C_same = 10.
    C_diff = 80.

    neg_diffsem =  tf.multiply(diffgroup_diffsem_mat_label, pred_simmat) # minimize distances if in the same group
    neg_samesem = alpha * tf.multiply(diffgroup_samesem_mat_label, tf.maximum(tf.subtract(C_same, pred_simmat), 0))
    pos = tf.multiply(samegroup_mat_label, tf.maximum(tf.subtract(C_diff, pred_simmat), 0))


    simmat_loss = neg_samesem + neg_diffsem + pos
    group_mask_weight = tf.matmul(group_mask, tf.transpose(group_mask, perm=[0, 2, 1]))
    # simmat_loss = tf.add(simmat_loss, pos)
    simmat_loss = tf.multiply(simmat_loss, group_mask_weight)

    simmat_loss = tf.reduce_mean(simmat_loss)
    
    # sementic segmentation loss
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem)
    tf.summary.scalar('classify loss', classify_loss)
    
    # discrimination loss function
    feature_dim = pred.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.
    param_reg = 0.000
    
    disc_loss, l_var, l_dist, l_reg = discriminative_loss(pred, ins_label, feature_dim,
                                         delta_v, delta_d, param_var, param_dist, param_reg)

    loss = 100.*classify_loss + 100.*disc_loss + 10*simmat_loss

    tf.add_to_collection('losses', loss)
    return loss, 100.*classify_loss, 100.*disc_loss, l_var, l_dist, l_reg,  10*simmat_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
