from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 shape_list_filename,
                 npoints=2500,
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation
        self.shape_list_filename = shape_list_filename

        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        self.datapath = []
        # get basic information for each shape in the dataset
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            normals_filename = os.path.join(self.root, shape_name+'.normals')
            normals = np.loadtxt(normals_filename).astype('float32')
            np.save(normals_filename+'.npy', normals)
            
            self.datapath.append((shape_ind, point_filename+'.npy', normals_filename+'.npy'))

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.load(fn[1])
        normals = np.load(fn[2])
        print(point_set.shape, normals.shape)

        choice = np.random.choice(len(normals), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        normals = normals[choice]
        point_set = torch.from_numpy(point_set)
        normals = torch.from_numpy(normals)
        
        return point_set, normals

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)


