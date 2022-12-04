from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train_normalnet(opt):
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ShapeNetDataset(
        root=opt.dataset,
        shape_list_filename='trainingset_no_noise.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        shape_list_filename='testset_no_noise.txt',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    model_filename = os.path.join(opt.outf, 'normal_model.pth')

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = NormalNet(sym_op = opt.sym_op, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=opt.momentum)
    # optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1) 
    if (mps_device != None):
        classifier.to(mps_device)
    else:
        classifier.cuda()

    num_batch = len(dataloader)
    losses = []
    for epoch in range(opt.nepoch):
        loss_avg = 0.0
        for i, data in enumerate(dataloader, 0):
            scheduler.step(epoch * num_batch + i)
            points, target = data
            points = points.transpose(2, 1)
            if (mps_device != None):
                points, target = points.to(mps_device), target.to(mps_device)
            else: 
                points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            
            # pred = pred.view(-1, num_classes)
            # target = target.view(-1, 1)[:, 0] - 1

            #print(pred.size(), target.size())
            loss = compute_loss(pred, target, opt.normal_loss)

            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            
            print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                if (mps_device != None):
                    points, target = points.to(mps_device), target.to(mps_device)
                else: 
                     points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = compute_loss(pred, target, opt.normal_loss)
                loss_avg += loss.item()
                print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
        loss_avg = loss_avg / num_batch
        losses.append(loss_avg)

        if epoch == opt.nepoch-1:
            torch.save(pcpnet.state_dict(), model_filename)
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(classifier.state_dict(), '%s/normal_model_%d.pth' % (opt.outf, epoch))
    # write losses to file for graph
    losses_file = open("losses.txt", "w")
    f.write(losses)
    f.close()

def cos_angle(v1, v2):
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

def compute_loss(pred, target, normal_loss):
    loss = 0
    if normal_loss == 'ms_euclidean':
        loss += (pred - target).pow(2).sum(1).mean()
    elif normal_loss == 'ms_oneminuscos':
        loss += (1-cos_angle(pred, target)).pow(2).mean()

    return loss


if __name__ == "__main__":
    mps_device = None
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--feature_transform', default=False, help="use feature transform")
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation: max or sum')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    opt = parser.parse_args()

    train_normalnet(opt)
    