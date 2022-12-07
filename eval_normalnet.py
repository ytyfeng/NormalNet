
import argparse
import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from dataset import PointcloudPatchDataset, SequentialPointcloudPatchSampler
from normalnet import NormalNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='./pclouds')
    parser.add_argument('--model', type=str, default='./models/Normal_estimation_model_99.pth', help='model path')
    parser.add_argument('--outdir', type=str, default='./results', help='output results folder')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--feature_transform', type=int, default=True, help='use feature transform')
    parser.add_argument('--sym_op', type=str, default='sum', help='symmetry operation')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay value for L2 regularization (eg. 0.01)')
    return parser.parse_args()

def eval_normalnet(opt):
    print(opt)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0")

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename='validationset_whitenoise.txt',
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed)

    test_datasampler = SequentialPointcloudPatchSampler(test_dataset)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=opt.workers)

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    normalnet = NormalNet(
        num_points=opt.points_per_patch,
        output_dim=3,
        feature_transform=opt.feature_transform,
        sym_op=opt.sym_op)

    normalnet.load_state_dict(torch.load(opt.model, map_location=device))
    normalnet.to(device)
    normalnet.eval()

    num_batch = len(test_dataloader)

    shape_ind = 0
    shape_patch_offset = 0
    shape_patch_count = test_dataset.shape_patch_count[shape_ind]
    shape_properties = torch.zeros(shape_patch_count, 3, dtype=torch.float, device=device)
    for i, data in enumerate(test_dataloader, 0):

        # get trainingset batch and upload to GPU
        points = data[0]
        # target = data[1:-1]

        points = points.transpose(2, 1)
        points = points.to(device)

        with torch.no_grad():
            pred, trans, _ = normalnet(points)
        o_pred = pred[:, 0:3]
        o_pred_len = torch.max(o_pred.new_tensor([sys.float_info.epsilon*100]), o_pred.norm(p=2, dim=1, keepdim=True))
        o_pred = o_pred / o_pred_len
        batch_offset = 0
        while batch_offset < pred.size(0):

            shape_remaining = shape_patch_count-shape_patch_offset
            batch_remaining = pred.size(0)-batch_offset

            # append estimated patch properties batch to properties for the current shape
            shape_properties[shape_patch_offset:shape_patch_offset+min(shape_remaining, batch_remaining), :] = pred[
                batch_offset:batch_offset+min(shape_remaining, batch_remaining), :]

            batch_offset += min(shape_remaining, batch_remaining)
            shape_patch_offset += min(shape_remaining, batch_remaining)

            if shape_remaining <= batch_remaining:
                # save normals
                normal_prop = shape_properties[:, 0:3]
                np.savetxt(os.path.join(opt.outdir, test_dataset.shape_names[shape_ind]+'.normals'), normal_prop.cpu().numpy())

            # start new shape
            if shape_ind + 1 < len(test_dataset.shape_names):
                shape_patch_offset = 0
                shape_ind += 1
                shape_patch_count = test_dataset.shape_patch_count[shape_ind]
                shape_properties = shape_properties.new_zeros(shape_patch_count, 3)


if __name__ == '__main__':
    eval_opt = parse_arguments()
    eval_normalnet(eval_opt)
