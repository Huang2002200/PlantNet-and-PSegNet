import torch
import torch.nn as nn
import torch.nn.functional as F


def dg_knn(adj_matrix, k=20,d=3):
    """Get KNN based on the pairwise distance.
    Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

    Returns:
    nearest neighbors: (batch_size, num_points, k)
    """
    k1 = k*d+1
    neg_adj = -adj_matrix
    _, nn_idx = torch.topk(neg_adj, k=k1)

    res = nn_idx[:,:,1:k1:d]

    return res


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.shape[0]
    point_cloud = torch.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)

    point_cloud_transpose = point_cloud.transpose(-1 ,-2)
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(torch.square(point_cloud), dim=-1, keepdim=True)
    point_cloud_square_transpose = point_cloud_square.permute(0, 2, 1)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def get_edge_feature(point_cloud, nn_idx, k=20):
    """
    Construct edge feature for each point.

    Args:
        point_cloud: (batch_size, num_points, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        edge features: (batch_size, num_points, k, 2*num_dims)
    """
    og_batch_size = point_cloud.shape[0]
    point_cloud = point_cloud.squeeze()
    if og_batch_size == 1:
        point_cloud = point_cloud.unsqueeze(0)

    point_cloud_central = point_cloud

    batch_size, num_points, num_dims = point_cloud.shape
    device = point_cloud.device
    idx_ = torch.arange(batch_size) * num_points
    idx_ = idx_.view(batch_size, 1, 1).to(device)

    point_cloud_flat = point_cloud.contiguous().view(-1, num_dims)
    point_cloud_neighbors = point_cloud_flat[nn_idx + idx_]#nn_idx + idx_（n,1024,k),point_cloud_flat(8192,6)
    point_cloud_central = point_cloud_central.unsqueeze(dim=-2)

    point_cloud_central = point_cloud_central.repeat(1, 1, k, 1)

    edge_feature = torch.cat([point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=-1)

    return edge_feature


def knn(x, k):
    ## 计算点之间的内积
    # The transpose(2, 1) function transposes the tensor to switch the second and third dimensions,
    # so that the matrix multiplication is performed between the last two dimensions of x .
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    # 计算每个点的平方和
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    # 计算
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # 找到每个点的k个最近邻的索引
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


