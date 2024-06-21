#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:55:37 2019
@author: dell
"""

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import tf_util
import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_grouping import query_ball_point, group_point, knn_point


# -----------------------------------------------------------------------------
# PREPARE multyscale DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------


def sample_and_group(radius, nsample, xyz, points, knn, use_xyz):
    '''
    Input:
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    if knn:
        _, idx = knn_point(nsample, xyz, xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    # translation normalization ??????
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, nsample, 1])
    if points is not None:
        # (batch_size, npoint, nsample, channel)
        grouped_points = group_point(points, idx)
        if use_xyz:
            # (batch_size, npoint, nample, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_points

#3个N*8 Concat 多k 多d
def add_model_2(input_data, input_feature, nsample_number, point_scale, num_out_channel, is_training, bn_decay, scope,
              bn=False, use_nchw=False):
    #    data_format = 'NCHW' if use_nchw else 'NHWC'
    # with tf.variable_scope(scope) as sc:
    # Grouping


    adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=3)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope, bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=7)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "2", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=11)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "3", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    net = tf.concat([net1, net2, net3], axis=-1)

    net = tf.reduce_max(
        net, axis=[2], keepdims=True, name='maxpool')
    net = tf.squeeze(net, [2])  # (batch_size, npoints, 64)

    return net  # (batch_size, npoints, 96)

#3个N*8 Concat d 变化
def add_model_3(input_data, input_feature, nsample_number, num_out_channel, is_training, bn_decay, scope,
              bn=False, use_nchw=False):
    #    data_format = 'NCHW' if use_nchw else 'NHWC'
    # with tf.variable_scope(scope) as sc:
    # Grouping


    adj_matrix = tf_util.pairwise_distance(input_data)  #[4096,4096]
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=3)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number) #[N,k,2*d]
    #[4096,16,6]

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope, bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    adj_matrix = tf_util.pairwise_distance(net1)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=7)
    edge_feature = tf_util.get_edge_feature(net1, nn_idx=nn_idx, k=nsample_number)
    # [4096,16,16]
    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "2", bn_decay=bn_decay)
    #[4096,16,8]
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util.pairwise_distance(net2)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=11)
    edge_feature = tf_util.get_edge_feature(net2, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "3", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    net = tf.concat([net1, net2, net3], axis=-1)

    net = tf.reduce_max(
        net, axis=[2], keepdims=True, name='maxpool')
    net = tf.squeeze(net, [2])  # (batch_size, npoints, 64)

    return net  # (batch_size, npoints, 96)

#3个N*8 Concat
def add_del_1(input_data, input_feature, nsample_number, point_scale, num_out_channel, is_training, bn_decay, scope,
              bn=False, use_nchw=False):
    #    data_format = 'NCHW' if use_nchw else 'NHWC'
    # with tf.variable_scope(scope) as sc:
    # Grouping


    adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=3)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope, bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    adj_matrix = tf_util.pairwise_distance(net1)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=3)
    edge_feature = tf_util.get_edge_feature(net1, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "2", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util.pairwise_distance(net2)
    nn_idx = tf_util.dg_knn(adj_matrix, k=nsample_number,d=3)
    edge_feature = tf_util.get_edge_feature(net2, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "3", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    net = tf.concat([net1, net2, net3], axis=-1)

    net = tf.reduce_max(
        net, axis=[2], keepdims=True, name='maxpool')
    net = tf.squeeze(net, [2])  # (batch_size, npoints, 64)

    return net  # (batch_size, npoints, 96)


#多半径DGCNN
def add_model(input_data, input_feature, nsample_number, point_scale, num_out_channel, is_training, bn_decay, scope,
              bn=False, use_nchw=False):
    #    data_format = 'NCHW' if use_nchw else 'NHWC'
    # with tf.variable_scope(scope) as sc:
    # Grouping
    """
    new_points0 = sample_and_group(point_scale[0], nsample_number,
                                   input_data, input_feature, knn=False, use_xyz=True)  # (batch_size, npoint, nsample_number, 3+channel)
    new_points1 = sample_and_group(point_scale[1], nsample_number,
                                   input_data, input_feature, knn=False, use_xyz=True)  # (batch_size, npoint, nsample_number, 3+channel)
    new_points2 = sample_and_group(point_scale[2], nsample_number,
                                   input_data, input_feature, knn=False, use_xyz=True)  # (batch_size, npoint, nsample_number, 3+channel)
    # Point Feature Embedding output(B N NSAMLE 64)
    """

    """
    adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.knn(adj_matrix, k=nsample_number,radius=point_scale[0])
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    """

    # adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.myknn(input_data, radius=point_scale[0], k=nsample_number)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope, bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    # adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.myknn(input_data, radius=point_scale[1], k=nsample_number)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "2", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    # adj_matrix = tf_util.pairwise_distance(input_data)
    nn_idx = tf_util.myknn(input_data, radius=point_scale[2], k=nsample_number)
    edge_feature = tf_util.get_edge_feature(input_data, nn_idx=nn_idx, k=nsample_number)

    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope + "3", bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    net = tf.concat([net1, net2, net3], axis=-1)

    net = tf.reduce_max(
        net, axis=[2], keepdims=True, name='maxpool')
    net = tf.squeeze(net, [2])  # (batch_size, npoints, 64)

    return net  # (batch_size, npoints, 96)


"""
def feature_nearst(feature_data, nsample_point, mlp1, mlp2, mlp3, mlp4,scope, is_training, bn_decay, radius=None, bn=False):
    feature_points = feature_data
#    point_size = feature_points.get_shape()[1].value
    # (batch_size, npoints, 1, 96)
    feature_points = tf.expand_dims(feature_points, 2)
    with tf.variable_scope(scope) as sc:
        # point feature embedding
        for i, num_out_channel in enumerate(mlp1):
            feature_points = tf_util.conv2d(feature_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv3%d' % (i), bn_decay=bn_decay)
        # pooling in globle feature
        feature_points1 = tf.squeeze(feature_points, [2])
#        net_globle = tf.reduce_max(feature_points, axis=[1], epdims=True, name='maxpool2')#Bx1x1x64
#        net_globle = tf.squeeze(net_globle, [2])#Bx1x64
#        net_globle = tf.tile(net_globle, [1, point_size, 1])
        # POINT FEATURE EMBEDDING for finding nearst feature points
        for i, num_out_channel1 in enumerate(mlp2):
            feature_points = tf_util.conv2d(feature_points, num_out_channel1, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv4%d' % (i), bn_decay=bn_decay)

        # (batch_size, npoints, 3)
        feature_points = tf.squeeze(feature_points, [2])
        # Grouping
        net_group = sample_and_group(radius, nsample_point, feature_points,
                                     feature_points1, knn=True, use_xyz=False)  # B X N x 64 x 64
        # point feature embedding
        for j, num_out_channel2 in enumerate(mlp3):
            net_group = tf_util.conv2d(net_group, num_out_channel2, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='conv5%d' % (j), bn_decay=bn_decay)
        # Pooling in Local Regions
        net_group = tf.reduce_max(
            net_group, axis=[2], keepdims=True, name='maxpool')  # B X N x 1 x 32
        net_group = tf.squeeze(net_group, [2])  # B X N x 32
        # concta
#        net_group = tf.concat(axis=-1, values=[feature_data, net_group])
#        net_group = tf.concat(axis=-1, values=[net_group, net_globle])#128
        return feature_points, net_group

"""


def feature_nearst(feature_data, nsample_point, mlp1, mlp2, mlp3, mlp4, scope, is_training, bn_decay, radius=None,
                   bn=False):
    feature_points = feature_data
    #    point_size = feature_points.get_shape()[1].value
    # (batch_size, npoints, 1, 96)
    feature_points = tf.expand_dims(feature_points, 2)
    with tf.variable_scope(scope) as sc:
        # point feature embedding
        for i, num_out_channel in enumerate(mlp1):
            feature_points = tf_util.conv2d(feature_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv3%dn' % (i), bn_decay=bn_decay)
        # pooling in globle feature
        feature_points1 = tf.squeeze(feature_points, [2])
        #        net_globle = tf.reduce_max(feature_points, axis=[1], keepdims=True, name='maxpool2')#Bx1x1x64
        #        net_globle = tf.squeeze(net_globle, [2])#Bx1x64
        #        net_globle = tf.tile(net_globle, [1, point_size, 1])
        # POINT FEATURE EMBEDDING for finding nearst feature points
        for i, num_out_channel1 in enumerate(mlp2):
            feature_points = tf_util.conv2d(feature_points, num_out_channel1, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv4%dn' % (i), bn_decay=bn_decay)

        # (batch_size, npoints, 3)
        feature_points = tf.squeeze(feature_points, [2])
        # Grouping
        net_group = sample_and_group(radius, nsample_point, feature_points,
                                     feature_points1, knn=True, use_xyz=False)  # B X N x 64 x 64

        # point feature embedding
        for k, num_out_channel3 in enumerate(mlp3):
            net_group = tf_util.conv2d(net_group, num_out_channel3, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='conv5%dn' % (k), bn_decay=bn_decay)
        # Pooling in Local Regions
        net_group = tf.reduce_max(
            net_group, axis=[2], keepdims=True, name='maxpool')  # B X N x 1 x 32

        # concta
        #        net_group = tf.concat(axis=-1, values=[feature_data, net_group])
        #        net_group = tf.concat(axis=-1, values=[net_group, net_globle])#128

        # point feature embedding
        for k, num_out_channel3 in enumerate(mlp4):
            net_group = tf_util.conv2d(net_group, num_out_channel3, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=bn, is_training=is_training,
                                       scope='conv6%dn' % (k), bn_decay=bn_decay)
        # Pooling in Local Regions
        net_group = tf.reduce_max(
            net_group, axis=[2], keepdims=True, name='maxpool')  # B X N x 1 x 6

        net_group = tf.squeeze(net_group, [2])  # B X N x 6

        return feature_points, net_group


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature