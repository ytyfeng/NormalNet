
import argparse
import os
import random
import math
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler
from normalnet import NormalNet


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./models', help='output folder (trained models)')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')

    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    parser.add_argument('--feature_transform', type=int, default=True, help='use feature transform')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()

def train_normalnet(opt):
    print(opt)
    
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)
    if opt.gpu_idx < 0 and torch.backends.mps.is_available():
        device = torch.device("mps")

    params_filename = os.path.join(opt.outdir, 'normal_estimation_params.pth')
    model_filename = os.path.join(opt.outdir, 'normal_estimation_model.pth')

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename='trainingset_whitenoise.txt',
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed)

    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed)
   
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=opt.workers)

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename='validationset_whitenoise.txt',
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed)

    test_datasampler = RandomPointcloudPatchSampler(
        test_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed,)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=opt.workers)

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    normalnet = NormalNet(
        num_points=opt.points_per_patch,
        output_dim=3,
        feature_transform=opt.feature_transform,
        sym_op=opt.sym_op)
    optimizer = optim.SGD(normalnet.parameters(), lr=opt.lr, momentum=opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1) # milestones in number of optimizer iterations
    normalnet.to(device)

    train_num_batch = len(train_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    losses = []
    for epoch in range(opt.nepoch):
        loss_avg = 0.0
        for i, data in enumerate(train_dataloader, 0):

            # set to training mode
            normalnet.train()

            # get trainingset batch and upload to GPU
            points = data[0]
            target = data[1:-1]

            points = points.transpose(2, 1)
            points = points.to(device)

            target = tuple(t.to(device) for t in target)

            optimizer.zero_grad()
            pred, _ = normalnet(points)

            loss = compute_loss(pred, target, opt.normal_loss)

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            # update learning rate
            scheduler.step()

            # print info and update log file
            print('[Normal Estimation %d: %d/%d] Train loss: %f' % (epoch, i, train_num_batch-1, loss.item()))

            if i % 10 == 0:
                j, data = next(enumerate(test_dataloader, 0))
                # set to evaluation mode
                normalnet.eval()

                # get testset batch and upload to GPU
                points = data[0]
                target = data[1:-1]

                points = points.transpose(2, 1)
                points = points.to(device)

                target = tuple(t.to(device) for t in target)

                # forward pass
                with torch.no_grad():
                    pred, _ = normalnet(points)

                loss = compute_loss(pred, target, opt.normal_loss)
                loss_avg += loss.item()
                # print info and update log file
                print('[Normal Estimation %d: %d/%d] Test loss: %f' % (epoch, i, train_num_batch-1, loss.item()))
        loss_avg = 10* loss_avg / train_num_batch 
        losses.append(loss_avg)
        # save model, overwriting the old model
        if epoch == opt.nepoch-1:
            torch.save(normalnet.state_dict(), model_filename)
        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(normalnet.state_dict(), os.path.join(opt.outdir, 'Normal_estimation_model_%d.pth' % epoch))
    # write losses to file for graph
    arr = np.array(losses)
    np.savetxt('losses.txt', arr)

def cos_angle(v1, v2):
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

def compute_loss(pred, target, normal_loss):
    loss = 0
    o_pred = pred[:, 0:3]
    o_target = target[0]
    if normal_loss == 'ms_euclidean':
        loss += (o_pred - o_target).pow(2).sum(1).mean()
    elif normal_loss == 'ms_oneminuscos':
        loss += (1-cos_angle(o_pred, o_target)).pow(2).mean()
    return loss

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_normalnet(train_opt)
