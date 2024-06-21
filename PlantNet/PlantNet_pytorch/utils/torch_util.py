import torch


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
    return dist,idx


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
      (i,j)对应点云内部第i个点与第 j 个点之间的距离
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
