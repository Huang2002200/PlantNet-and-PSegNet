# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:17:35 2021

@author: JS-L
"""
from tf_sampling import gather_point
import tensorflow as tf
import tf_util

def att_pooling(feature_set, d_out, name, is_training):#dout 通道数
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
    
def relative_pos_encoding( xyz,k,d):
    adj_matrix = tf_util.pairwise_distance(xyz)  # [4096,4096]
    neigh_idx = tf_util.dg_knn(adj_matrix, k=k, d=d)
    neighbor_xyz = gather_point(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
    relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
    return relative_feature   

    
def dilated_res_block(feature, xyz, neigh_idx, d_out, name, is_training):##总的聚合特征
    f_pc = tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
    f_pc = building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
    f_pc = tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
    shortcut = tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
    return tf.nn.leaky_relu(f_pc + shortcut)

def building_block(xyz, feature, neigh_idx, d_out, name, is_training):##聚合特征
    d_in = feature.get_shape()[-1].value
    f_xyz =relative_pos_encoding(xyz, neigh_idx)
    f_xyz = tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
    f_neighbours = gather_point(tf.squeeze(feature, axis=2), neigh_idx)
    f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
    f_pc_agg =att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

    f_xyz = tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
    f_neighbours =gather_point(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
    f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
    f_pc_agg = att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
    return f_pc_agg