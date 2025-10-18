from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DFAN import DFAN

from data import get_training_set
from data import get_eval_set
import pdb
import socket
import time
# from test import *
from torchsummary import summary

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--data_dir', type=str, default='/media/smile/新加卷/liu/BDE/DFAN/datasets')  # revise
parser.add_argument('--hr_train_dataset', type=str, default='SINTEL_1000')
parser.add_argument('--hbd', type=int,default=16, help='bit-depth of GT image')
parser.add_argument('--lbd', type=int,default=4, help='bit-depth of low bit-depth image')
parser.add_argument('--stride', type=int, default=1, help='stride of convolution operation')
parser.add_argument('--bit_gap', type=int, default=12, help="bit-depth gap between HBD and LBD")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')   # origin
parser.add_argument('--snapshots', type=int, default=20, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='DBPN')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=96, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='SINTEL_train_HRDESKTOP-3I3OD01DBPNtpami_residual_filter8_epoch_659.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())   # 获取运行主机的名字以及ip地址
# print(hostname)
cudnn.benchmark = True   # 内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        # print(input)
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()   # 统计模型参数量
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    model_out_path = opt.save_folder+opt.hr_train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading traindatasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.bit_gap, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
model = DFAN(in_nc=3, nf=36, num_modules=12, upscale=opt.bit_gap)
    
model = torch.nn.DataParallel(model, device_ids=gpus_list)  # 多GPU训练的时候
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
# print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    print('++++++++++++++++++++++++++++++++')
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)
    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (200) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
