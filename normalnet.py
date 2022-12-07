import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, sym_op="sum", num_points=500):
        super(STN3d, self).__init__()
        self.mps_device = None
        if torch.backends.mps.is_available():
            self.mps_device = torch.device("mps")
        self.sym_op = sym_op
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op == 'sum':
            x = torch.sum(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        if self.mps_device != None:
            iden = iden.to(self.mps_device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64, sym_op="sum", num_points=500):
        super(STNkd, self).__init__()
        self.mps_device = None
        if torch.backends.mps.is_available():
            self.mps_device = torch.device("mps")
        self.sym_op = sym_op
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op == 'sum':
            x = torch.sum(x, 2, keepdim=True)
        
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        if self.mps_device != None:
            iden = iden.to(self.mps_device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points=500, feature_transform=True, sym_op='max', global_feature=False):
        super(PointNetfeat, self).__init__()
        self.stn1 = STN3d(sym_op=sym_op)
        self.num_points = num_points
        self.feature_transform = feature_transform
        self.sym_op = sym_op
        self.global_feature = global_feature
        if self.feature_transform:
            self.fstn = STNkd(k=64, sym_op=sym_op)

        self.conv0a = torch.nn.Conv1d(3, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.mp1 = torch.nn.MaxPool1d(num_points)

        
    def forward(self, x):
        n_pts = x.size()[2]
        # input transform
        trans = self.stn1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        
        # feature transform
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # local point feature    
        pointFeat = x
        # symmetric max operation over all points
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op == 'sum':
            x = torch.sum(x, 2, keepdim=True)

        # use both point features and global feature for normal estimation
        if self.global_feature == False:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            newX = torch.cat([x, pointFeat], 1)
            newX = torch.sum(newX, 2, keepdim=True)
            newX = newX.view(-1, 2048)
            return newX, trans, trans_feat
        else:
            # only use global feature
            x = x.view(-1, 1024)
            return x, trans, trans_feat

class NormalNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, feature_transform=True, sym_op='max', global_feature=False):
        super(NormalNet, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(
            num_points=num_points,
            feature_transform=feature_transform,
            sym_op=sym_op,
            global_feature=global_feature)
        self.global_feature = global_feature
        self.output_dim = output_dim
        self.fc0 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.do = nn.Dropout(p=0.3)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        if self.global_feature == False:
            x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.do(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.do(x)
        x = self.fc5(x)
        return x, trans, trans_feat
