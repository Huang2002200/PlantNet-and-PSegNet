import os
import h5py
import numpy as np
from torch.utils.data import Dataset


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


class PlantnetDataset(Dataset):
    def __init__(self, file_list):
        f = h5py.File(file_list[0], 'r')
        self.data = f['data'][:].astype('float32')
        self.label = f['pid'][:].astype('int64')
        self.seg = f['seglabel'][:].astype('int64')
        self.objlabel = f['obj'][:].astype('int64')
        f.close()

    def __getitem__(self, index):
        data1 = self.data[index]
        label1 = self.label[index]
        seg1 = self.seg[index]
        obj1 = self.objlabel[index]
        sample = {'data': data1, 'label': label1, 'seg': seg1, 'obj': obj1}
        return sample

    def __len__(self):
        return len(self.label)


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['pid'][:]
    seg = f['seglabel'][:]
    objlabel = f['obj'][:]
    return (data, label, seg, objlabel)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx, ...], idx


def loadDataFile_with_groupseglabel_stanfordindoor(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    group = f['pid'][:].astype(np.int32)#NxG
    if 'label' in f:
        label = f['label'][:].astype(np.int32)
    else :
        label = []
    if 'seglabel' in f:
        seg = f['seglabel'][:].astype(np.int32)
    else:
        seg = f['seglabels'][:].astype(np.int32)
    return (data, group, label, seg)


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud
