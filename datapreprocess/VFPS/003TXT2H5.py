import os
import numpy as np
import h5py


def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:, 0:3]
    label = (data[:, 3:6]).astype(int)
    return point_xyz, label


def change_scale(data):

    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    data[:, 0:3] = data[:, 0:3] - xyz_move
    # scale
    scale = np.max(data[:, 0:3])
    return data[:, 0:3] / scale


if __name__ == "__main__":
    DATA_ALL = []
    num_sample = 4096
    base_path = '/train_aug'
    DATA_FILES = os.listdir(base_path)

    for fn in range(len(DATA_FILES)):
         current_data, current_label= loadDataFile(os.path.join(base_path, DATA_FILES[fn]))
         change_data = change_scale(current_data)
         data_label = np.column_stack((change_data, current_label))
         DATA_ALL.append(data_label)

    output = np.vstack(DATA_ALL)
    output = output.reshape(-1, num_sample, 6)
    f = h5py.File("/train_h5/train.h5", "w")
    f['data'] = output[:, :, 0:3]
    f['pid'] = output[:, :, 3]  # 实例标签
    f['seglabel'] = output[:, :, 4]
    f['obj'] = output[:, :, 5]

    f.close()



