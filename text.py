import torch
from torch.autograd import Variable
from torch.nn import functional
import os
from FCN8s_pytorch import FCN
import FCN8s_pytorch
from dataset import MyTestData
import time
import PIL.Image as Image
import argparse

import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./MSRAK/')
parser.add_argument('--prior_map', default='prior')  # set prior_map to the name of the directory of proir maps
parser.add_argument('--output_dir', default='./pubcode/')  # save checkpoint parameters
parser.add_argument('--m', default='conv')  # fully connected or convolutional region embedding
parser.add_argument('--f', default='./parameters/feature.pth')  # set it to None to download my trained parameters



def main():
    opt = parser.parse_args()
    print(opt)

    input_dir = opt.input_dir
    prior_map = opt.prior_map
    output_dir = opt.output_dir

    # parameters

    param_feature = opt.f

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    loader = torch.utils.data.DataLoader(
        MyTestData(input_dir, transform=True, ptag=prior_map),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    feature = FCN()
    feature.cuda()
    feature.load_state_dict(torch.load(param_feature))
    feature.eval()

    test(loader, feature, output_dir)

def test(loader, feature, output_dir):
    feature.train(False)
    print('start')
    start_time = time.time()
    it = 0
    for ib, (data, msk, img_name, img_size) in enumerate(loader):
        print(it)

        inputs = Variable(data).cuda()

        msk = Variable(msk.unsqueeze(1)).cuda()

        feats = feature(inputs)

        msk = functional.sigmoid(feats)
        mask = msk.data[0, 0].cpu().numpy()
        mask = (mask*255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((img_size[0][0], img_size[1][0]))
        mask.save(os.path.join(output_dir, img_name[0] + '.png'),'png')
        it += 1
    print('end, cost %.2f seconds for %d images' % (time.time() - start_time, it))


if __name__ == '__main__':
    main()