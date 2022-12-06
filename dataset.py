import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial


def load_shape(point_filename, normals_filename):
    pts = np.load(point_filename+'.npy')
    normals = np.load(normals_filename+'.npy')
    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals)

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, kdtree, normals=None, pidx=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding b ox diagonal of a shape
    def __init__(self, root, shape_list_filename, patch_radius, points_per_patch,
                 seed=None, point_count_std=0.0, sparse_patches=False):

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.sparse_patches = sparse_patches
        self.point_count_std = point_count_std
        self.seed = seed
        self.include_normals = True
            
        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(100, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            self.patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius])

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3, dtype=torch.float)
        patch_pts_valid = []
        scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
            patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad))

            
            point_count = min(self.points_per_patch, len(patch_point_inds))

            # randomly decrease the number of points to get patches with different point densities
            if self.point_count_std > 0:
                point_count = max(5, round(point_count * self.rng.uniform(1.0-self.point_count_std*2)))
                point_count = min(point_count, len(patch_point_inds))

            # if there are too many neighbors, pick a random subset
            if point_count < len(patch_point_inds):
                patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]

            start = s*self.points_per_patch
            end = start+point_count
            scale_ind_range[s, :] = [start, end]

            patch_pts_valid += list(range(start, end))

            # convert points to torch tensors
            patch_pts[start:end, :] = torch.from_numpy(shape.pts[patch_point_inds, :])

            patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])
            
            # normalize size of patch (scale with 1 / patch radius)
            patch_pts[start:end, :] = patch_pts[start:end, :] / rad


        if self.include_normals:
            patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])

        trans = torch.eye(3).float()
        return (patch_pts,) +(patch_normal,) + (trans,)

    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        return load_shape(point_filename, normals_filename)
