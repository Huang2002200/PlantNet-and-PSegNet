import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import torch_util

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.sqrt(dist)
    idx = torch.argmin(dist, dim=-1)
    return dist, idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(npoint, xyz):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def get_sample_points(xyz,points,npoints,use_xyz=True):

    batch_size = xyz.shape[0]
    index = farthest_point_sample(npoints, xyz)
    new_xyz = index_points(xyz, index)  # (batch_size, npoint, 3)
    if points.shape[2] > 0:
        sample_points = []
        for b in range(batch_size):
             sample_points.append(points[b, ...].index_select(0, index[b, :]))
        sample_points = torch.stack(sample_points)
        if use_xyz:
            new_points = torch.cat([new_xyz, sample_points], dim=-1)
        else:
            new_points = sample_points
    else:
            new_points = new_xyz  # (B,N,C)
    return new_xyz,new_points

def relative_pos_encoding(xyz, neigh_idx):
    neighbor_xyz = index_points(xyz, neigh_idx)# (B,N,k,3)
    xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)# (B,N,k,3)
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = torch.sqrt(torch.sum(relative_xyz ** 2, dim=-1, keepdim=True))# (B,N,k,1)
    relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)# (B,N,k,10)
    return relative_feature

def gather_neighbour(pc, neighbor_idx):
    batch_size = pc.size(0)
    num_points = pc.size(1)
    d = pc.size(2)
    index_input = neighbor_idx.view(batch_size, -1)
    features = index_points(pc, index_input)
    features = features.view(batch_size, num_points, neighbor_idx.size(-1), d)
    return features

class position_encode(nn.Module):
    def __init__(self, k, d):
        super(position_encode, self).__init__()
        self.k1 = k
        self.d1 = d
    def forward(self, new_xyz,new_points):
        adj_matrix = torch_util.pairwise_distance(new_points)#(B,N,N)
        nn_idx = torch_util.dg_knn(adj_matrix, k=self.k1, d=self.d1)#(B,N,2*K)
        new_xyz1 = relative_pos_encoding(new_xyz, nn_idx)  # (B,N,K,C)
        return new_xyz1
class edge_conv(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(edge_conv, self).__init__()
        self.CONV2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2))
    def forward(self, new_points, k, d):
        adj_matrix = torch_util.pairwise_distance(new_points)  # [B,N,N]
        nn_idx = torch_util.dg_knn(adj_matrix, k=k, d=d)# [B,N,K]
        feature_set = torch_util.get_edge_feature(new_points, nn_idx=nn_idx, k=k)
        new_points = self.CONV2d(feature_set.transpose(0, 1).transpose(1, 3))#(N,2*C,k,B)
        new_points1 = new_points.transpose(1, 3).transpose(0, 1)  # (B,N,K,2*C)
        return new_points1

class att_pooling(nn.Module):
    def __init__(self,  out_channel):
        super(att_pooling, self).__init__()
        in_channel = out_channel+10
        self.l1 = nn.Linear(in_channel, out_channel+10)
        self.CONV2d = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(negative_slope=0.2))
    def forward(self,new_xyz1, new_points1):
        '''
        Args:
            new_xyz1: position_encode得到的特征
            new_points1: edge_conv得到的特征
        '''
        points_xyz = torch.cat([new_points1, new_xyz1], dim=-1)  # (B,N,K,C+10)
        batch_size = points_xyz.shape[0]  # 获取形状信息
        num_points = points_xyz.shape[1]
        num_neigh = points_xyz.shape[2]
        d = points_xyz.shape[3]
        f_reshaped = points_xyz.reshape(-1, num_neigh, d)
        att_activation = self.l1(f_reshaped)
        att_scores = torch.nn.functional.softmax(att_activation, dim=1)
        f_agg = f_reshaped * att_scores
        f_agg = torch.sum(f_agg, dim=1)  # (B*N,C)
        f_agg = f_agg.reshape(batch_size, num_points, 1, d)
        f_agg = self.CONV2d(f_agg.transpose(0, 1).transpose(1, 3))
        f_agg = f_agg.transpose(0, 1).transpose(0, 3)
        og_batch = f_agg.shape[0]
        f_agg = f_agg.squeeze()  # (B,N,C)
        if og_batch == 1:
            f_agg = f_agg.unsqueeze(dim=0)
        return f_agg

class PointNetSetAbstraction(nn.Module):
    def __init__(self, mlp1, in_channel, k, d):
        super(PointNetSetAbstraction, self).__init__()
        self.k1 = k
        self.d1 = d
        self.mlp = mlp1
        self.PE = position_encode(k, d)
        self.mlp1_aps = nn.ModuleList()
        self.mlp1_ecs = nn.ModuleList()
        for i, out_channel in enumerate(mlp1):
            self.mlp1_aps.append(att_pooling(out_channel))
            if i == 0:
                self.mlp1_ecs.append(edge_conv(2*in_channel, out_channel))
            elif i == 1:
                self.mlp1_ecs.append(edge_conv(2*mlp1[0], out_channel))
            else:
                self.mlp1_ecs.append(edge_conv(2*2*mlp1[1], out_channel))
    def forward(self, new_xyz, new_points):
        new_xyz1 = self.PE(new_xyz, new_points)
        for i, num_out_channel in enumerate(self.mlp):
            AP = self.mlp1_aps[i]
            EC = self.mlp1_ecs[i]
            if i == 0:
                new_points1 = EC(new_points, self.k1, self.d1)#(B,N,K,32)
                f_agg1 = AP(new_xyz1, new_points1)#(B,N,32)
            elif i == 1:
                new_points2 = EC(f_agg1, self.k1, self.d1)#(B,N,K,32)
                f_agg2 = AP(new_xyz1, new_points2)
                f_agg2 = torch.cat([f_agg1, f_agg2], dim=-1)
            else:
                new_points3 = EC(f_agg2, self.k1, self.d1)
                f_agg3 = AP(new_xyz1, new_points3)
                f_agg3 = f_agg2 + f_agg3
        return f_agg3

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, mlp, in_channel):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp = mlp
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N,C]
            xyz2: sampled input points position data, [B, S,C]
            points1: input points data, [B, N,C]
            points2: input points data, [B, N,C]
        Return:
            new_points: (batch_size, ndataset1, mlp[-1])
        """

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists,idx= square_distance(xyz1, xyz2)
            dists, idx = torch.sort(dists)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)# B,ndataset1,nchannel1+nchannel2
        else:
            new_points = interpolated_points
        new_points = torch.unsqueeze(new_points, 2)#(B,N,1,C)
        new_points = new_points.transpose(0, 1).transpose(1, 3) # (NC1B)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        og_batch=new_points.shape[-1]
        new_points = new_points.squeeze()  # ndataset1,mlp[-1],B
        if og_batch==1:
            new_points = new_points.unsqueeze(-1)
        new_points= new_points.transpose(0,2).transpose(1,2)# new_points: (batch_size, ndataset1, mlp[-1])
        return new_points


class SimmatModel(nn.Module):
    def __init__(self):
        super(SimmatModel, self).__init__()
        # 在这里定义你的模型结构，包括可能的层和操作

    def forward(self, Fsim, batch_size):
        r = torch.sum(Fsim * Fsim, dim=2)
        r = r.view(batch_size, -1, 1)
        D = r - 2 * torch.matmul(Fsim, Fsim.transpose(1, 2)) + r.transpose(1, 2)
        zero_tensor = torch.tensor(0)
        simmat_logits = torch.maximum(10 * D, zero_tensor)

        return simmat_logits


