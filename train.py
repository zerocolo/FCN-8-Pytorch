import gc

from torch.autograd import Variable

from dataset import *
from FCN8s_pytorch import *
import FCN8s_pytorch

from dataset import MyData

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='./ECSSD/train')  # training dataset
parser.add_argument('--val_dir', default='./ECSSD/val')  # validation dataset
parser.add_argument('--check_dir', default='./parameters')  # save parameters
parser.add_argument('--m', default='conv')  # fully connected or convolutional region embedding
parser.add_argument('--e', type=int, default=36)  # epoches
parser.add_argument('--b', type=int, default=3)  # batch size
parser.add_argument('--p', type=int, default=5)  # probability of random flipping during training
opt = parser.parse_args()
print(opt)


def validation(feature,loader):
    total_loss = 0
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        lbl = lbl.float()
        lbl = Variable(lbl.unsqueeze(1)).cuda()

        feats = feature(inputs)

        loss = criterion(feats, lbl)
        total_loss += loss.data[0]
    return total_loss / len(loader)


check_root = opt.check_dir
train_data = opt.train_dir
val_data = opt.val_dir
p = opt.p
epoch = opt.e
bsize = opt.b
is_fc = False

if not os.path.exists(check_root):
    os.mkdir(check_root)

# models
feature = FCN()
feature.cuda()

loader = torch.utils.data.DataLoader(
            MyData(train_data, transform=True),
            batch_size=bsize*5, shuffle=True, num_workers=2, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)


if __name__ == '__main__':
    for it in range(epoch):
        for ib, (data, lbl) in enumerate(loader):
            inputs = Variable(data).cuda()
            lbl = Variable(lbl.float().unsqueeze(1)).cuda()

            feats = feature(inputs)

            loss = criterion(feats, lbl)

            feature.zero_grad()
            loss.backward()

            optimizer_feature.step()
            print('loss: %.4f (epoch: %d, step: %d)' % (loss.item(), it, ib))
            del inputs, lbl, loss, feats
            gc.collect()

    loader = torch.utils.data.DataLoader(
                MyData(train_data, transform=True),
                batch_size=bsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
                MyData(val_data, transform=True),
                batch_size=bsize, shuffle=True, num_workers=2,pin_memory=True)
    min_loss = 1000.0
    for it in range(epoch):
        for ib, (data, lbl) in enumerate(loader):
            inputs = Variable(data).cuda()
            lbl = lbl.float()

            lbl = Variable(lbl.unsqueeze(1)).cuda()
            feats = feature(inputs)
            loss = criterion(feats, lbl)

            feature.zero_grad()
            loss.backward()
            optimizer_feature.step()

            print('loss: %.4f, min-loss: %.4f (epoch: %d, step: %d)' % (loss.item(), min_loss, it, ib))

            del inputs, lbl, loss, feats
            gc.collect()

        vb = validation(feature, val_loader)
        if vb < min_loss:
            filename = ('%s/feature.pth' % (check_root))
            torch.save(feature.state_dict(), filename)
            print('save: (epoch: %d, step: %d)' % (it, ib))
            min_loss = vb