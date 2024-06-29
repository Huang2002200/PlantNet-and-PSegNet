import numpy as np
import os
import torch
from os import listdir, path



def farthest_point_sample(xyz, npoint, z):
    device = xyz.device
    N, C = xyz.shape
    torch.manual_seed(z)
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distance = torch.ones(N, dtype=torch.float64).to(device) * 1e10
    farthest = torch.randint(0, N, (1,),dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def main():
    source_path = '/train'
    txt_files = [f for f in os.listdir(source_path) if f.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(source_path, txt_file)
        points = np.loadtxt(file_path)
        ply_array = np.asarray(points)
        s = len(points)
        if s < 4096:
            ply_array = np.tile(ply_array, (2, 1))

        b = torch.from_numpy(ply_array)
        b = b[:, : 3]
        # expand 10
        for z in range(0, 10):
            sampled_points_index = farthest_point_sample(b, 4096, z)
            # save path
            np.savetxt('/train_aug/' + str(txt_file)+str(z),
                       ply_array[sampled_points_index], fmt="%f %f %f %d %d %d")

if __name__ == '__main__':
    main()
    