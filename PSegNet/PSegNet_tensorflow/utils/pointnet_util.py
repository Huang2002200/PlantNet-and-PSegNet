""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import random

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
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

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def my_dgcnn(new_points, k,d,num_out_channel,is_training,scope,bn_decay):
    adj_matrix = tf_util.pairwise_distance(new_points)
    nn_idx = tf_util.dg_knn(adj_matrix, k=k, d=d)
    edge_feature = tf_util.get_edge_feature(new_points, nn_idx=nn_idx, k=k)
    net = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope=scope, bn_decay=bn_decay)
    return net


def new_group_point(xyz, points, npoint, use_xyz):
    ''' ADD New operation for point group by Shi Guoliang 20201216
    Input:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor
        npoint: int32 -- #points sampled in farthest point sampling
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Return:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    point_num = points.get_shape()[1].value
    batch_size = points.get_shape()[0].value
    
    max_feature = tf.reduce_max(points, axis=-2, keep_dims=True)#B x 1 x F
    max_feature = tf.tile(max_feature, [1, point_num, 1])
    
    differ_feature = max_feature - points# B X N X F
    differ_feature = tf.reduce_sum(differ_feature, 2, keep_dims=True)#B X N X 1
    differ_feature = tf.squeeze(differ_feature, [2])#B X N
    
    sort_index = tf.argsort(differ_feature, axis=-1, direction='DESCENDING')
    sample_index = sort_index[:,0:npoint]#B X npoint
    
    new_xyz = gather_point(xyz, sample_index) # (batch_size, npoint, 3)
    if points.get_shape()[2]>0:
        sample_points = []
        for b in range(batch_size):
            sample_points.append(tf.gather(points[b,...],sample_index[b,:]))
        sample_points = tf.stack(sample_points)
        
        if use_xyz:
            new_points = tf.concat([new_xyz, sample_points], axis=-1)
        else:
            new_points = sample_points
    else:
        new_points = new_xyz
    
    return new_xyz, new_points, sample_index

def random_group_point(xyz, points, npoint, use_xyz):
    ''' ADD New operation for point group by Shi Guoliang 20201218
    Input:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor
        npoint: int32 -- #points sampled in farthest point sampling
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Return:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    point_num = points.get_shape()[1].value
    batch_size = points.get_shape()[0].value
    
    index = random.sample(range(0, point_num),npoint)
    sample_index = np.tile(index,(batch_size,1))
    
    new_xyz = gather_point(xyz, sample_index) # (batch_size, npoint, 3)
    if points.get_shape()[2]>0:
        sample_points = []
        for b in range(batch_size):
            sample_points.append(tf.gather(points[b,...],sample_index[b,:]))
        sample_points = tf.stack(sample_points)
        
        if use_xyz:
            new_points = tf.concat([new_xyz, sample_points], axis=-1)
        else:
            new_points = sample_points
    else:
        new_points = new_xyz
    
    return new_xyz, new_points, sample_index
    
    

def pointnet_sa_module_1(xyz, points, npoint, mlp1, k,d,mlp2, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling

            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region

            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        data_format = 'NCHW' if use_nchw else 'NHWC'
        # Type 1
        batch_size = xyz.get_shape()[0].value
        index = farthest_point_sample(npoint, xyz)
        new_xyz = gather_point(xyz, index) # (batch_size, npoint, 3)
        if points.get_shape()[2]>0:
            sample_points = []
            for b in range(batch_size):
                sample_points.append(tf.gather(points[b,...],index[b,:]))
            sample_points = tf.stack(sample_points)
            
            if use_xyz:
                new_points = tf.concat([new_xyz, sample_points], axis=-1)
            else:
                new_points = sample_points
        else:
            new_points = new_xyz
        
        # Type 2
        # new_xyz, new_points, index = new_group_point(xyz, points, npoint, use_xyz)
        
        # Type 3
        # new_xyz, new_points, index = random_group_point(xyz, points, npoint, use_xyz)

        # Point Feature Embedding
        # new_points1=relative_pos_encoding(xyz,k,d,new_xyz)
        #_,neigh_idx=knn_point(k, xyz,new_xyz)
        # adj_matrix = tf_util.pairwise_distance(new_points)  # [4096,4096]
        # nn_idx = tf_util.dg_knn(adj_matrix, k=k, d=d)
        #neigh_idx=tf.reshape(neigh_idx,[10,npoint*k])
        # new_points1=relative_pos_encoding( new_xyz, neigh_idx)
        #new_points1=tf.reduce_max(new_points1,-2,keepdims=True)
        #new_points1 = tf.squeeze(new_points1, -2)
        if mlp1 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp1):
                adj_matrix = tf_util.pairwise_distance(new_points)  # [4096,4096]
                nn_idx = tf_util.dg_knn(adj_matrix, k=k, d=d)
                new_points1=relative_pos_encoding( new_xyz, nn_idx)
                new_points = my_dgcnn(new_points, k,d,num_out_channel,is_training,scope='conv%d'%(i), bn_decay=bn_decay)
                new_points=tf.concat([new_points,new_points1],axis=-1)
                new_points=att_pooling(new_points,num_out_channel, 'lG%d'%(i), is_training,bn_decay)
                new_points=tf.squeeze(new_points,axis=-2)
                if i == 0:
                    cur_points = new_points
                elif i==1:
                    cur_points = tf.concat([cur_points, new_points], axis=-1)
                    new_points = cur_points
                else:
                        cur_points = cur_points+new_points
            new_points = cur_points
        return new_xyz, new_points
        # if mlp1 is not None:
        #     if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        #     for i, num_out_channel in enumerate(mlp1):
               
        #         new_points = my_dgcnn(new_points, k,d,num_out_channel,is_training,scope='conv%d'%(i), bn_decay=bn_decay)
        #         #new_points=tf.concat([new_points1,new_points],axis=-1)
        #         #new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
        #                                     # padding='VALID', stride=[1,1],
        #                                     # bn=bn, is_training=is_training,
        #                                     # scope='conv_post_%d'%(i), bn_decay=bn_decay,
        #                                     # data_format=data_format)
        #         #new_points=att_pooling(new_points, num_out_channel, 'lg', is_training)
        #        # new_points = my_AFA(new_points,num_out_channel,is_training,scope='conv%d'%(i),bn_decay=bn_decay,use_softmax=False)
        #         new_points = tf.squeeze(new_points, -2)
        ###     woshizhjishi hhhhhhhhhhhhhhhhhhhhhhhhhhhh
        #         if i==0:
        #             cur_points = new_points
        #         elif i==1:
        #             cur_points = tf.concat([cur_points, new_points], axis=-1)
        #             new_points = cur_points
        #         else:
        #             cur_points = cur_points+new_points
        #     new_points = cur_points

        # [Optional] Further Processing
        if mlp2 is not None:
            new_points = tf.expand_dims(new_points, 2)
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for j, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(j), bn_decay=bn_decay,
                                            data_format=data_format)
            new_points = tf.squeeze(new_points, [2])

        return new_xyz, new_points
   

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
    
def add_module(new_points, mlp0, mlp1, mlp2, k, d, is_training, bn_decay, scope='AddlLayer'):
    if mlp0 is not None:
        new_points = tf.expand_dims(new_points, -1)
        for j, num_out_channel in enumerate(mlp0):
            if j==0:
                c = new_points.get_shape()[2].value
            else:
                c = 1
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,c],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conhv_qpost_%d'%(j), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2])
        
        
    if mlp1 is not None:
        for i, num_out_channel in enumerate(mlp1):
            new_points = my_dgcnn(new_points, k,d,num_out_channel,is_training,scope='cosnv%d'%(i), bn_decay=bn_decay)
            new_points = tf.squeeze(new_points, -2)
            if i==0:
                cur_points = new_points
            elif i==1:
                cur_points = tf.concat([cur_points, new_points], axis=-1)
                new_points = cur_points
            else:
                cur_points = cur_points+new_points
        new_points = cur_points
        
    # [Optional] Further Processing
    if mlp2 is not None:
        new_points = tf.expand_dims(new_points, 2)
        for j, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='conhv_post_%d'%(j), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2])
        
    return new_points
#---------------------------------------------------------------------------------------
def att_pooling(feature_set, d_out, name, is_training,bn_decay):#dout 通道数
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
        f_agg = tf_util.conv2d(f_agg, d_out, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope=name+'mlp', bn_decay=bn_decay)
        
        return f_agg
    
def relative_pos_encoding( xyz, neigh_idx):
    neighbor_xyz = gather_neighbour(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
    relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
    return relative_feature   

def gather_neighbour(pc, neighbor_idx):
    # gather the coordinates or features of neighboring points
    batch_size = tf.shape(pc)[0]
    num_points = tf.shape(pc)[1]
    d = pc.get_shape()[2].value
    index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
    features = gather_point(pc, index_input)
    features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
    return features