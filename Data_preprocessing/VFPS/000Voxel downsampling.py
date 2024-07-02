import os
import numpy as np
import open3d as o3d

def create_open3d_point_cloud(points, labels=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if labels is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(labels)
    return point_cloud

def main():
    source_path = '/raw_data_path'
    output_file = '/datasample'
    txt_files = [f for f in os.listdir(source_path) if f.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(source_path, txt_file)
        data = np.loadtxt(file_path)
        points = data[:, :3]  # xyz信息
        labels = data[:, 3:]  # label信息
        pc = create_open3d_point_cloud(points, labels)

        voxel_size = 0.3
        downsampled_pc = pc.voxel_down_sample(voxel_size)

        # 准备保存的数据
        points_to_save = np.asarray(downsampled_pc.points)
        labels_to_save = np.asarray(downsampled_pc.colors) if downsampled_pc.has_colors() else None

        # 将标签信息写入.txt文件
        save_file = os.path.join(output_file, txt_file.replace('.txt', '_downsampled.txt'))
        if labels_to_save is not None:
            data_to_save = np.concatenate((points_to_save, labels_to_save), axis=1)
        else:
            data_to_save = points_to_save

        np.savetxt(save_file, data_to_save, fmt='%f %f %f %d %d %d')

    print("Voxel downsampling complete.")

if __name__ == "__main__":
    main()
