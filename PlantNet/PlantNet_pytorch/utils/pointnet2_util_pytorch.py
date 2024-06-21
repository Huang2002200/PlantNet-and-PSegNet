import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import torch_util


class GetEdgeFeature(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GetEdgeFeature, self).__init__()
        self.l1 = nn.Linear(in_channel, in_channel, device='cuda')
        self.CONV2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2))
    def forward(self, new_points, k, d):
        adj_matrix = torch_util.pairwise_distance(new_points)  # [4096,4096]
        nn_idx = torch_util.dg_knn(adj_matrix, k=k, d=d)
        edge_points = torch_util.index_points(new_points, nn_idx)
        new_points = new_points.unsqueeze(dim=-2)
        new_points = new_points.repeat(1, 1, k, 1)
        feature_set = torch.cat([new_points, new_points - edge_points], dim=-1)
        batch_size = feature_set.shape[0]
        num_points = feature_set.shape[1]
        num_neigh = feature_set.shape[2]
        d = feature_set.shape[3]
        f_reshaped = feature_set.reshape(-1, num_neigh, d)
        att_activation = self.l1(f_reshaped)
        att_scores = F.softmax(att_activation, dim=1)
        f_agg = f_reshaped * att_scores
        f_agg = torch.sum(f_agg, dim=1)  # (B*N,C)
        f_agg = f_agg.reshape(batch_size, num_points, 1, d)
        f_agg = f_agg.transpose(0, 1).transpose(1, 3)  # (NC1B)
        f_agg = self.CONV2d(f_agg)  # (num_points, d, 1, B)
        f_agg = f_agg.transpose(0, 1).transpose(0, 3)  # (B,N,1,mlp[0])
        og_batch = f_agg.shape[0]
        f_agg = f_agg.squeeze()  # (B,N,C)
        if og_batch == 1:
            f_agg = f_agg.unsqueeze(dim=0)
        return f_agg


def get_sample_points(xyz,points,npoints,use_xyz=True):

    batch_size = xyz.shape[0]
    index = torch_util.farthest_point_sample(npoints, xyz)
    new_xyz = torch_util.index_points(xyz, index)  # (batch_size, npoint, 3)
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
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, mlp1, in_channel):
        super(PointNetSetAbstraction, self).__init__()
        self.mlp_dgcnn = mlp1
        self.ecs = nn.ModuleList()
        for i, out_channel in enumerate(mlp1):
            if i == 0:
                self.ecs.append(GetEdgeFeature(in_channel, mlp1[0]))
            if i == 1:
                self.ecs.append(GetEdgeFeature(2*(mlp1[0]), mlp1[1]))
            if i == 2:
                self.ecs.append(GetEdgeFeature(4*(mlp1[1]), mlp1[2]))

    def forward(self, new_points, k, d):
        for i, mlp in enumerate(self.mlp_dgcnn):
            ec = self.ecs[i]
            if i == 0:
                new_points = ec(new_points, k=k, d=d)#b,n,32
                cur_points = new_points
            elif i == 1:
                new_points = ec(new_points, k=k, d=d)
                cur_points = torch.cat([cur_points, new_points], dim=-1)
                new_points = cur_points
            else:
                new_points = ec(new_points, k=k, d=d)
                cur_points = cur_points + new_points
        new_points = cur_points
        return new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, mlp,in_channel):
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
            dists,idx= torch_util.square_distance(xyz1, xyz2)
            dists, idx = torch.sort(dists)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(torch_util.index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

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

    def forward(self, Fsim, batch_size):
        r = torch.sum(Fsim * Fsim, dim=2)
        r = r.view(batch_size, -1, 1)
        D = r - 2 * torch.matmul(Fsim, Fsim.transpose(1, 2)) + r.transpose(1, 2)
        zero_tensor = torch.tensor(0)
        simmat_logits = torch.maximum(10 * D, zero_tensor)

        return simmat_logits
