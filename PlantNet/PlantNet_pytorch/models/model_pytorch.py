import numpy as np
from utils.pointnet2_util_pytorch import PointNetFeaturePropagation, get_sample_points, PointNetSetAbstraction
from utils.loss_pytorch import *
from utils.torch_util import *
import torch.nn as nn

NUM_CATEGORY = 6
NUM_GROUPS = 40  # larger than gt instance numbers

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
                pts_label_mask[idx, jdx] = 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum

    return label_one_hot, pts_label_mask
def convert_groupandcate_to_one_hot(grouplabels):
    # grouplabels: BxN
    grouplabels = grouplabels.astype(int)
    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], NUM_GROUPS))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))

    un, cnt = np.unique(grouplabels, return_counts=True)#un:ndarry(7,),cnt:ndarry(7,)
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
                pts_group_mask[idx, jdx] = 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

    return group_one_hot, pts_group_mask
class plantnet_model(nn.Module):
    def __init__(self, num_classes):
        super(plantnet_model, self).__init__()
        """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
        self.num_classes = num_classes
        self.sa1 = PointNetSetAbstraction(mlp1=[32, 32, 64], in_channel=(3+3)*2)
        self.sa2 = PointNetSetAbstraction(mlp1=[64, 64, 128], in_channel=(64+3)*2)
        self.sa3 = PointNetSetAbstraction(mlp1=[128, 128, 256], in_channel=(128+3)*2)
        self.sa4 = PointNetSetAbstraction(mlp1=[256, 256, 512], in_channel=(256+3)*2)
        self.fp4 = PointNetFeaturePropagation(mlp=[256, 256],in_channel=512+256)
        self.fp3 = PointNetFeaturePropagation(mlp=[256, 256],in_channel=128+256)
        self.fp2 = PointNetFeaturePropagation(mlp=[256, 128],in_channel=64+256)
        self.fp1 = PointNetFeaturePropagation(mlp=[128, 128, 128],in_channel=3+128)
        self.fp8 = PointNetFeaturePropagation(mlp=[256, 256], in_channel=512 + 256)
        self.fp7 = PointNetFeaturePropagation(mlp=[256, 256], in_channel=128 + 256)
        self.fp6 = PointNetFeaturePropagation(mlp=[256, 128], in_channel=64 + 256)
        self.fp5 = PointNetFeaturePropagation(mlp=[128, 128, 128], in_channel=3 + 128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.CONV1 = nn.Sequential(
               nn.Conv1d(128,128, 1, device='cuda'),
               nn.BatchNorm1d(128),
               nn.LeakyReLU(negative_slope=0.2))
        self.CONV3 = nn.Sequential(
               nn.Conv1d(256, 128, 1, device='cuda'),
               nn.BatchNorm1d(128),
               nn.LeakyReLU(negative_slope=0.2))
        self.CONV2 = nn.Sequential(
               nn.Conv1d(128, 128, 1, device='cuda'),
               nn.BatchNorm1d(128),
               nn.LeakyReLU(negative_slope=0.2))
        self.CONV4 = nn.Sequential(
               nn.Conv1d(256, 128, 1, device='cuda'),
               nn.BatchNorm1d(128),
               nn.LeakyReLU(negative_slope=0.2))
        self.CONV5 = nn.Conv1d(128, self.num_classes, 1, device='cuda')
        self.CONV6 = nn.Conv1d(128, 5, 1, device='cuda')

    def forward(self, point_cloud):
            batch_size = point_cloud.shape[0]
            l0_xyz = point_cloud
            l0_points = point_cloud

            #layer1
            l1_xyz, l1_points = get_sample_points(l0_xyz, l0_points, npoints=1024)
            l1_points = self.sa1(l1_points,k=16, d=1)#(8,1024,64)

            l2_xyz, l2_points = get_sample_points(l1_xyz, l1_points, npoints=256)
            l2_points = self.sa2(l2_points,k=16, d=1)#(8,256,128)

            l3_xyz, l3_points = get_sample_points(l2_xyz, l2_points, npoints=128)
            l3_points = self.sa3(l3_points,k=8, d=1)#(8,128,256)

            l4_xyz, l4_points = get_sample_points(l3_xyz, l3_points,  npoints=128)
            l4_points = self.sa4(l4_points,k=8, d=1)#(8,128,512)
            # Feature Propagation layers
            #sem
            l3_points_sem = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
            l2_points_sem = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_sem)
            l1_points_sem = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_sem)
            l0_points_sem = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_sem)
            net_sem_cache = self.CONV1(l0_points_sem.transpose(1, 2)).transpose(1, 2)
            #ins
            l3_points_ins = self.fp8(l3_xyz, l4_xyz, l3_points, l4_points)
            l2_points_ins = self.fp7(l2_xyz, l3_xyz, l2_points, l3_points_ins)
            l1_points_ins = self.fp6(l1_xyz, l2_xyz, l1_points, l2_points_ins)
            l0_points_ins = self.fp5(l0_xyz, l1_xyz, l0_points, l1_points_ins)
            net_ins_cache = self.CONV2(l0_points_ins.transpose(1, 2)).transpose(1, 2)
            net_fuse_cache = net_ins_cache + net_sem_cache
            # # Similarity matrix
            Fsim = net_fuse_cache
            r = torch.sum(Fsim * Fsim, dim=2)
            r = r.view(batch_size, -1, 1)
            D = r - 2 * torch.matmul(Fsim, Fsim.transpose(1, 2)) + r.transpose(1, 2)
            zero_tensor = torch.tensor(0)
            simmat_logits = torch.maximum(10 * D, zero_tensor)
            # Ins
            net_ins_2 = torch.cat([l0_points_ins, net_fuse_cache], dim=-1)
            net_ins_atten = torch.sigmoid(torch.mean(net_ins_2, dim=-1, keepdim=True))
            net_ins_3 = net_ins_2 * net_ins_atten#(16,4096,256)
            # Sem
            net_sem_2 = torch.cat([l0_points_sem, net_fuse_cache], dim=-1)
            net_sem_atten = torch.sigmoid(torch.mean(net_sem_2, dim=-1, keepdim=True))
            net_sem_3 = net_sem_2 * net_sem_atten
            # Output
            net_ins_3 = self.CONV3(net_ins_3.transpose(1, 2))
            net_ins_4 = self.drop1(net_ins_3)
            net_ins_4 = self.CONV6(net_ins_4).transpose(1, 2)
            net_sem_3 = self.CONV4(net_sem_3.transpose(1, 2))
            net_sem_4 = self.drop2(net_sem_3)
            net_sem_4 = self.CONV5(net_sem_4).transpose(1, 2)
            return net_sem_4, net_ins_4, simmat_logits


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred, ins_label, pred_sem, sem_ins_fuse, pts_semseg_label, pts_group_label,pts_group_mask, device):
        """ pred:   BxNxE,（16,4096,5）
            ins_label:  BxN（16，4096）
            pred_sem: BxNx13（16,4096,6）
            sem_label: BxN（16，4096）
            sem_ins_fuse：（16,4096,4096）
            pts_semseg_label（16,4096,6）
            pts_group_label（16,4096,40）
            pts_group_mask（16,4096）
        """
        pts_semseg_label = torch.from_numpy(pts_semseg_label).to(device)
        pts_group_label = torch.from_numpy(pts_group_label).to(device)
        pts_group_mask = torch.from_numpy(pts_group_mask).to(device)
        # double hinge loss add by Shi Guoliang 20201219
        group_mask = torch.unsqueeze(pts_group_mask, dim=2)#（16,4096，1）
        pred_simmat = sem_ins_fuse
        # Similarity Matrix loss
        group_mat_label = torch.matmul(pts_group_label, pts_group_label.transpose(1, 2))
        sem_mat_label = torch.matmul(pts_semseg_label, pts_semseg_label.transpose(1, 2)).float()
        samesem_mat_label = sem_mat_label
        diffsem_mat_label = torch.sub(1.0, sem_mat_label)
        samegroup_mat_label = group_mat_label
        diffgroup_mat_label = torch.sub(1.0, group_mat_label)#BxNxN: (i,j)=0， if i and j in the same group，不是同一类则（i,j)=1
        diffgroup_samesem_mat_label = torch.mul(diffgroup_mat_label, samesem_mat_label)
        diffgroup_diffsem_mat_label = torch.mul(diffgroup_mat_label, diffsem_mat_label)

        # Double hinge loss
        alpha = 10.
        C_same = 10.
        C_diff = 80.
        neg_diffsem = torch.mul(diffgroup_diffsem_mat_label, pred_simmat)
        zero_tensor = torch.tensor(0)
        neg_samesem = alpha * torch.mul(diffgroup_samesem_mat_label, torch.maximum(torch.sub(C_same, pred_simmat),  zero_tensor))
        pos = torch.mul(samegroup_mat_label, torch.maximum(torch.sub(C_diff, pred_simmat),  zero_tensor))

        simmat_loss = neg_samesem + neg_diffsem + pos

        group_mask_weight = torch.matmul(group_mask, group_mask.transpose(1, 2))
        simmat_loss = torch.mul(simmat_loss, group_mask_weight)
        simmat_loss = torch.mean(simmat_loss)

        # sementic segmentation loss
        pred_sem = pred_sem.transpose(1, 2)
        classify_loss = self.loss_fun(pred_sem, pts_semseg_label.transpose(1, 2))

       # discrimination loss function
        feature_dim = pred.shape[-1]#pred:ins feature
        delta_v = 0.5
        delta_d = 1.5
        param_var = 1.
        param_dist = 1.
        param_reg = 0.001

        disc_loss,l_var,l_dist,l_reg = discriminative_loss(pred, ins_label, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg)#pred(16,4096,5),ins_label(16,4096)

        loss = 10. * classify_loss + 10. * disc_loss + simmat_loss
        return loss, 10.*classify_loss, 10.*disc_loss, l_var, l_dist, l_reg, simmat_loss


