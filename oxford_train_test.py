# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import time
import sys
import os
from shutil import copytree, copy
from oxford_model import RansacNet
from data.oxford_dataloader import TrainDatasetMultiSeq
from torch.utils.tensorboard import SummaryWriter
import torchvision

from torch.utils.data import random_split
from tqdm import tqdm
import cv2
import open3d as o3d
#import copy
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path






global global_step
global_step = 0

BATCH_SIZE=10
num_epochs=10
num_workers=10

dataset_root = '/mnt/Disk1/training_data_local_vel_i2o1_with_tf/data_aug/2019-01-10-11-46-21-radar-oxford-10k'
num_past_frames = 2
out_seq_len = 1
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
cell_category_num = 2
val_percent=0.05

model_save_path='./'

def main():
    start_epoch = 1

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    trainset = TrainDatasetMultiSeq(dataset_root=dataset_root, future_frame_skip=0, num_past_frames=num_past_frames, num_future_frames=1, voxel_size=voxel_size,
                                    area_extents=area_extents, num_category=cell_category_num)

    n_val = int(len(trainset) * val_percent)
    n_train = len(trainset) - n_val
    train_set, val_set = random_split(trainset, [n_train, n_val])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    #print(type(trainloader))
    print("Training dataset size:", len(trainset))

    model = RansacNet()
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss(reduce='True', reduction='sum')
    #criterion = nn.L1Loss(reduction='sum')
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0002) # 0.0016
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    writer = SummaryWriter(comment='RansacNet')

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        #scheduler.step()
        model.train()
        pred_odom, loss_odom = train(model, criterion, trainloader, optimizer, device, epoch, writer, voxel_size)
        # save model
        if (epoch % 1 == 0 or epoch == num_epochs or epoch == 1 or epoch > 20):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         #'scheduler_state_dict': scheduler.state_dict(),
                         'loss': loss_odom.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

def train(model, criterion, trainloader, optimizer, device, epoch, writer, voxel_size):
    running_loss_odom = AverageMeter('Odom', ':.6f')  # for odometry estimation error

    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        global global_step
        global_step += 1

        raw_radars, pixel_radar_map_gt, pixel_lidar_map_gt, pixel_moving_map_gt, all_disp_field_gt, tf_gt = data

        all_disp_field_gt = all_disp_field_gt.view(-1, out_seq_len, 256, 256, 2) # [1, future_num*bs, 256, 256, 2]
        disp_gt = all_disp_field_gt[:, -out_seq_len:, ...].contiguous() # [1, future_num*bs, 256, 256, 2]
        disp_gt = disp_gt.view(-1, disp_gt.size(2), disp_gt.size(3), disp_gt.size(4)) # [future_num*bs, 256, 256, 2]
        disp_gt = disp_gt.permute(0, 3, 1, 2).to(device) # [future_num*bs, 2, 256, 256]

        valid_mask = torch.from_numpy(np.zeros((disp_gt.shape[0], 1, 256, 256))) # [future_num*bs, 1, 256, 256]
        valid_mask[:,0] = pixel_lidar_map_gt[:,0] # pixel_radar_map_gt # torch.Size([bs, 256, 256])
        motion_mask = torch.from_numpy(np.zeros((disp_gt.shape[0], 1, 256, 256)).astype(np.float32))
        motion_mask[:,0] = pixel_moving_map_gt[:,0,:,:,0] # torch.Size([bs, 256, 256])
        motion_mask = motion_mask.to(device)

        #print('disp_gt.shape', disp_gt.shape)
        #print('motion_mask.shape', motion_mask.shape)
        disp_gt = disp_gt * torch.logical_not(motion_mask)

        tf_gt = tf_gt.to(device)

        # predict
        odom_pred = model(disp_gt)

        odom_pred = odom_pred[...,0,0].double()
        #print('tf_gt.shape', tf_gt.shape)
        #print('odom_pred.shape', odom_pred.shape)
        print('tf_gt', tf_gt[0])
        print('odom_pred', odom_pred[0])

#        fig, ax = plt.subplots(1, 3, figsize=(5, 5))
#        ax[0].imshow(np.linalg.norm(disp_gt[0].cpu().detach().numpy(), axis=0))
#        ax[1].imshow(disp_gt[0,0].cpu().detach().numpy())
#        ax[2].imshow(disp_gt[0,1].cpu().detach().numpy())
#        plt.show()

        # compute loss
        #print('odom_pred.dtype', odom_pred.dtype)
        #print('tf_gt.dtype', tf_gt.dtype)
        loss_odom = criterion(odom_pred, tf_gt)
        #print('loss_odom.dtype', loss_odom.dtype)
        loss_odom = loss_odom/tf_gt.shape[0]
        #print('loss_odom.dtype', loss_odom.dtype)
        loss_odom_value = loss_odom.item()

        # bp
        loss_odom.backward()
        optimizer.step()

        writer.add_scalar('loss_odom/train', loss_odom_value, global_step)

        running_loss_odom.update(loss_odom_value)
        print("[{}/{}]\t{}".format(epoch, i, running_loss_odom))
    return odom_pred, running_loss_odom



if __name__ == "__main__":
    main()
