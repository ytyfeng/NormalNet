import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

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
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STN(nn.Module):
    def __init__(self, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

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

        # symmetric operation over all points
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op == 'sum':
            x = torch.sum(x, 2, keepdim=True)
        
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=500, feature_transform=True, sym_op='max'):
        super(PointNetfeat, self).__init__()
        #self.stn3d = STN3d()
        self.num_points = num_points
        self.feature_transform = feature_transform
        self.sym_op = sym_op
        self.stn1 = QSTN(num_points=num_points, dim=3, sym_op=self.sym_op)
        if self.feature_transform:
            self.stn = STN(num_points=num_points, dim=64, sym_op=self.sym_op)

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

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        
    def forward(self, x):
        n_pts = x.size()[2]
        # input transform
        # from tuples to list of single points
        x = x.view(x.size(0), 3, -1)
        trans = self.stn1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = x.contiguous().view(x.size(0), 3, -1)

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        
        # feature transform
        if self.feature_transform:
            trans_feat = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        pointFeat = x
        # symmetric max operation over all points
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op == 'sum':
            x = torch.sum(x, 2, keepdim=True)

        #x = x.view(-1, 1024)
        #print(x.size())
        #return x, trans

        # use both point features and global feature for normal estimation
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        newX = torch.cat([x, pointFeat], 1)
        newX = torch.sum(newX, 2, keepdim=True)
        newX = newX.view(-1, 2048)
        return newX, trans, trans_feat

class NormalNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, feature_transform=True, sym_op='max'):
        super(NormalNet, self).__init__()
        self.num_points = num_points
        
        self.feat = PointNetfeat(
            num_points=num_points,
            feature_transform=feature_transform,
            sym_op=sym_op)
        self.output_dim = output_dim
        self.fc0 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, output_dim)
        self.bn0 = nn.BatchNorm1d(1024)
        self.do = nn.Dropout(p=0.3)
        '''
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
        '''


    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn0(self.fc0(x)))
        x = self.do(x)
        x = self.fc1(x)
        '''
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
        '''
        return x, trans, trans_feat

class QSTN(nn.Module):
    def __init__(self, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

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

        # symmetric operation over all points
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = batch_quat_to_rotmat(x)

        return x

# quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out
