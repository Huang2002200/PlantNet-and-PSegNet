import numpy as np
import math
import time
import sys
import os

class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)  # Returns the sum of squared Euclidean distances between the set of sample points and other points

    def _call__(self, pts, k):  # PTS is the input point cloud,  K is the number of downsampling
        farthest_pts = np.zeros((k, 6),
                                dtype=np.float32)  # The first three columns are coordinates xyz, and the fourth column is labels
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3]))
        return farthest_pts


if __name__ == '__main__':
    path = '/train'
    saved_path = '/train_aug'
    Filelist = os.listdir(path)
    n = len(Filelist)
    for idx in range(n):
        points = np.loadtxt(os.path.join(path, Filelist[idx]), dtype=float, delimiter=' ')
        pcd_array = np.array(points)
        print("pcd_array.shape:", pcd_array.shape)
        sample_count = 4096
        for z in range(10):  # do 10 times data augmentation
            # Fixed number of points after FPS downsampling
            # Perform FPS Downsampling for center point set and edge point set respectively
            FPS = FarthestSampler()
            sample_points = FPS._call__(pcd_array, sample_count)
            file_nameR = Filelist[idx].split(".")[0] + "_aug_" + str(z) + ".txt"
            np.savetxt(os.path.join(saved_path, file_nameR), sample_points, fmt='%f %f %f %d %d %d')
