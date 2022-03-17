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
from oxford_model import RaMNet
from data.oxford_dataloader import TrainDatasetMultiSeq
from torch.utils.tensorboard import SummaryWriter
import torchvision

from torch.utils.data import random_split
from tqdm import tqdm
import cv2
import open3d as o3d

import matplotlib.pyplot as plt

#import copy

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






use_weighted_loss = True

use_class_loss = True # True
use_motion_loss = False # True
use_disp_loss = False # True

use_odom_loss = False
use_odom_net = False

use_temporal_info = True
num_past_frames = 2
out_seq_len = 1  # The number of future frames we are going to predict

val_percent = 0.05

# static
height_feat_size = 1 #13  # The size along the height dimension
cell_category_num = 2  # The number of object categories (including the background)
# no use
pred_adj_frame_distance = True  # Whether to predict the relative offset between frames
trans_matrix_idx = 1  # Among N transformation matrices (N=2 in our experiment), which matrix is used for alignment (see paper)

global global_step
global_step = 0

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--pretrain', default='', type=str, help='The path to the saved model that is loaded as pretrained model')
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=45, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')

parser.add_argument('--reg_weight_bg_tc', default=0.1, type=float, help='Weight of background temporal consistency term')
parser.add_argument('--reg_weight_fg_tc', default=2.5, type=float, help='Weight of instance temporal consistency')
parser.add_argument('--reg_weight_sc', default=15.0, type=float, help='Weight of spatial consistency term')

parser.add_argument('--use_bg_tc', action='store_true', help='Whether to use background temporal consistency loss')
parser.add_argument('--use_fg_tc', action='store_true', help='Whether to use foreground loss in st.')
parser.add_argument('--use_sc', action='store_true', help='Whether to use spatial consistency loss')

parser.add_argument('--nn_sampling', action='store_true', help='Whether to use nearest neighbor sampling in bg_tc loss')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')
parser.add_argument('--board', action='store_true', help='Whether to show in tensorboard')
parser.add_argument('-sv', '--spatial_val_num', default=-1, type=int, help='Section number for Spatial vaidation')

args = parser.parse_args()
print(args)

need_log = args.log
need_board = args.board

BATCH_SIZE = args.batch
num_epochs = args.nepoch
num_workers = args.nworker

reg_weight_bg_tc = args.reg_weight_bg_tc  # The weight of background temporal consistency term
reg_weight_fg_tc = args.reg_weight_fg_tc  # The weight of foreground temporal consistency term
reg_weight_sc = args.reg_weight_sc  # The weight of spatial consistency term

use_bg_temporal_consistency = args.use_bg_tc
use_fg_temporal_consistency = args.use_fg_tc
use_spatial_consistency = args.use_sc

use_nn_sampling = args.nn_sampling


def main():
    start_epoch = 1

    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_multi_seq'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.write("use_class_loss: {}\n use_motion_loss: {}\n use_disp_loss: {}\n use_odom_loss: {}\n use_odom_net: {}\n use_temporal_info: {}\n use_weighted_loss: {}\n".format(use_class_loss,use_motion_loss,use_disp_loss,use_odom_loss,use_odom_net,use_temporal_info,use_weighted_loss))
            saver.flush()

            # Copy the code files as logs
            copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
            copytree('data', os.path.join(model_save_path, 'data'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"
            #model_save_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-03-28_21-32-13_as_pretrain'

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")

            saver.flush()

        #if arg.pretrain != '':


    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])

    trainset = TrainDatasetMultiSeq(dataset_root=args.data, spatial_val_num=args.spatial_val_num, future_frame_skip=0, num_past_frames=num_past_frames, num_future_frames=1, voxel_size=voxel_size,
                                    area_extents=area_extents, num_category=cell_category_num)
    n_val = int(len(trainset) * val_percent)
    n_train = len(trainset) - n_val
    train_set, val_set = random_split(trainset, [n_train, n_val])

    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    #print(type(trainloader))
    print("Training dataset size:", len(trainset))

    model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames, use_odom_net=use_odom_net)
    model = nn.DataParallel(model)
    model = model.to(device)

    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='sum')
    #optimizer = optim.Adam(model.parameters(), lr=0.0016)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20, 30, 40, 50], gamma=0.5)

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    if args.pretrain != '':
        checkpoint = torch.load(args.pretrain)
        start_epoch = 1
        #print('checkpoint[model_state_dict]', checkpoint['model_state_dict'] )
        model.load_state_dict(checkpoint['model_state_dict'], False) # strict=False
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.pretrain, checkpoint['epoch']))

    if need_board:
      writer = SummaryWriter()
    else:
      writer=-1

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        scheduler.step()
        model.train()

        loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc, disp_pred, class_pred, motion_pred \
            = train(model, criterion, trainloader, optimizer, device, epoch, writer, voxel_size)

        loss_disp_val, loss_class_val, loss_motion_val, loss_odom_val = eval(model, criterion, valloader, device, epoch, writer, voxel_size)
        if need_board:
          if use_class_loss:
            writer.add_scalar('loss_class/val', loss_class_val, epoch)
          if use_motion_loss:
            writer.add_scalar('loss_motion/val', loss_motion_val, epoch)
          if use_disp_loss:
            writer.add_scalar('loss_disp/val', loss_disp_val, epoch)
          if use_odom_loss:
            writer.add_scalar('loss_odom/val', loss_odom_val, epoch)

        if need_log:
            saver.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(loss_disp, loss_class, loss_motion, loss_motion, loss_bg_tc,
                                                          loss_fg_tc, loss_sc))
            saver.flush()

        # save model
        if need_log and (epoch % 1 == 0 or epoch == num_epochs or epoch == 1 or epoch > 20):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'loss': loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_board:
      writer.close()
    if need_log:
        saver.close()

def eval(model, criterion, valloader, device, epoch, writer, voxel_size):
  n_val = len(valloader)
  model.eval()
  loss_class_tot=0
  loss_motion_tot=0
  loss_disp_tot=0
  loss_odom_tot=0
  with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for i, data in enumerate(valloader, 0):
      raw_radars, pixel_radar_map_gt, pixel_lidar_map_gt, pixel_moving_map_gt, all_disp_field_gt, tf_gt = data
      # Move to GPU/CPU
      raw_radars = raw_radars.view(-1, num_past_frames, 256, 256, height_feat_size)
      raw_radars = raw_radars.to(device)
      # Make prediction
      with torch.no_grad():
        if use_temporal_info == True:
          if use_odom_net:
            disp_pred, class_pred, motion_pred, odom_pred = model(raw_radars)
          else:
            disp_pred, class_pred, motion_pred = model(raw_radars)
            odom_pred = -1
        else:
          raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 256, 256, 1)).astype(np.float32))
          raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
          raw_radars_curr = raw_radars_curr.to(device)
          if use_odom_net:
            disp_pred, class_pred, motion_pred, odom_pred = model(raw_radars_curr)
          else:
            disp_pred, class_pred, motion_pred = model(raw_radars_curr)
            odom_pred = -1
      # Compute the losses
      optimizer=-1
      loss_class, loss_motion, loss_disp, loss_odom, loss_bg_tc, loss_sc, loss_fg_tc = \
          compute_and_bp_loss(optimizer, device, out_seq_len, pixel_radar_map_gt, pixel_lidar_map_gt, pixel_moving_map_gt, all_disp_field_gt, tf_gt,
          criterion, class_pred, motion_pred, disp_pred, odom_pred, raw_radars, voxel_size, bp_loss=False)
      loss_class_tot += loss_class
      loss_motion_tot += loss_motion
      loss_odom_tot += loss_odom
      if loss_disp > 0:
        loss_disp_tot += loss_disp
#      else:
#        n_val = n_val-1

      pbar.update()
  model.train()
  return loss_disp_tot/n_val, loss_class_tot/n_val, loss_motion_tot/n_val, loss_odom_tot/n_val

def train(model, criterion, trainloader, optimizer, device, epoch, writer, voxel_size):
    running_loss_bg_tc = AverageMeter('bg_tc', ':.7f')  # background temporal consistency error
    running_loss_fg_tc = AverageMeter('fg_tc', ':.7f')  # foreground temporal consistency error
    running_loss_sc = AverageMeter('sc', ':.7f')  # spatial consistency error
    running_loss_disp = AverageMeter('Disp', ':.6f')  # for motion prediction error
    running_loss_class = AverageMeter('Obj_Cls', ':.6f')  # for cell classification error
    running_loss_motion = AverageMeter('Motion_Cls', ':.6f')  # for state estimation error
    running_loss_odom = AverageMeter('Odom', ':.6f')  # for odometry estimation error

    for i, data in enumerate(trainloader, 0):
        global global_step
        global_step += 1

        raw_radars, pixel_radar_map_gt, pixel_lidar_map_gt, pixel_moving_map_gt, all_disp_field_gt, tf_gt = data
#        print('---trainloader output---')
#        print(raw_radars.shape)
#        print(pixel_radar_map_gt.shape)
#        print(pixel_moving_map_gt.shape)
#        torch.Size([bs, 1, history_frame_num, 256, 256, 1])
#        torch.Size([bs, 1, 256, 256, 2])
#        torch.Size([bs, 1, 256, 256, 2])

        # Move to GPU/CPU
        raw_radars = raw_radars.view(-1, num_past_frames, 256, 256, height_feat_size)
        raw_radars = raw_radars.to(device)

        # Make prediction
        if use_temporal_info == True:
          #print('---network input--- \n',raw_radars.shape) #input shape: torch.Size([bs*1, 5, 256, 256, 1])
          if use_odom_net:
            disp_pred, class_pred, motion_pred, odom_pred = model(raw_radars)
          else:
            disp_pred, class_pred, motion_pred = model(raw_radars)
            odom_pred = -1
        else:
          raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 256, 256, 1)).astype(np.float32))
          raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
          raw_radars_curr = raw_radars_curr.to(device)
          #print('---network input--- \n', raw_radars_curr.shape) #input shape: torch.Size([bs*1, 1, 256, 256, 1])
          if use_odom_net:
            disp_pred, class_pred, motion_pred, odom_pred = model(raw_radars_curr)
          else:
            disp_pred, class_pred, motion_pred = model(raw_radars_curr)
            odom_pred = -1
#        print('---network output---')
#        print(disp_pred.shape)
#        print(class_pred.shape)
#        print(motion_pred.shape)
        # [20*bs*2, 2, 256, 256]
        # [bs*2, 5, 256, 256]
        # [bs*2, 2, 256, 256]

        # Compute and back-propagate the losses
        loss_class, loss_motion, loss_disp, loss_odom, loss_bg_tc, loss_sc, loss_fg_tc = \
            compute_and_bp_loss(optimizer, device, out_seq_len, pixel_radar_map_gt, pixel_lidar_map_gt, pixel_moving_map_gt, all_disp_field_gt, tf_gt,
            criterion, class_pred, motion_pred, disp_pred, odom_pred, raw_radars, voxel_size, bp_loss=True)

        if need_board:
          if loss_class>0:
            writer.add_scalar('loss_class/train', loss_class, global_step)
          if loss_motion>0:
            writer.add_scalar('loss_motion/train', loss_motion, global_step)
          if loss_disp>0:
            writer.add_scalar('loss_disp/train', loss_disp, global_step)
          if loss_odom>0:
            writer.add_scalar('loss_odom/train', loss_odom, global_step)

          if global_step%100==0:
            raw_radars_viz = torch.from_numpy(np.zeros((raw_radars.shape[0],3,raw_radars.shape[2],raw_radars.shape[3])))
            raw_radars_viz[:,0,:,:] = raw_radars[:,0,:,:,0]
            raw_radars_viz[:,1,:,:] = raw_radars[:,0,:,:,0]
            raw_radars_viz[:,2,:,:] = raw_radars[:,0,:,:,0]

            writer.add_images('raw_radars', raw_radars_viz, global_step)

            class_pred_viz = torch.from_numpy(np.zeros((class_pred.shape[0],3,class_pred.shape[2],class_pred.shape[3])))
            class_pred_viz[:,0,:,:] = class_pred[:,0,:,:]>0.5
            class_pred_viz[:,1,:,:] = class_pred[:,0,:,:]>0.5
            class_pred_viz[:,2,:,:] = class_pred[:,0,:,:]>0.5
            writer.add_images('class_pred', class_pred_viz, global_step)

            motion_pred_viz = torch.from_numpy(np.zeros((motion_pred.shape[0],3,motion_pred.shape[2],motion_pred.shape[3])))
            motion_pred_viz[:,0,:,:] = motion_pred[:,0,:,:]>0.5
            motion_pred_viz[:,1,:,:] = motion_pred[:,0,:,:]>0.5
            motion_pred_viz[:,2,:,:] = motion_pred[:,0,:,:]>0.5
            writer.add_images('motion_pred', motion_pred_viz, global_step)

            disp_pred0_viz = torch.from_numpy(np.zeros((disp_pred.shape[0],3,disp_pred.shape[2],motion_pred.shape[3])))
            disp_pred0_viz[:,0,:,:] = torch.abs(disp_pred[:,0,:,:])
            disp_pred0_viz[:,1,:,:] = torch.abs(disp_pred[:,0,:,:])
            disp_pred0_viz[:,2,:,:] = torch.abs(disp_pred[:,0,:,:])
            disp_pred1_viz = torch.from_numpy(np.zeros((disp_pred.shape[0],3,disp_pred.shape[2],motion_pred.shape[3])))
            disp_pred1_viz[:,0,:,:] = torch.abs(disp_pred[:,1,:,:])
            disp_pred1_viz[:,1,:,:] = torch.abs(disp_pred[:,1,:,:])
            disp_pred1_viz[:,2,:,:] = torch.abs(disp_pred[:,1,:,:])
            writer.add_images('disp_pred0', disp_pred0_viz, global_step)
            writer.add_images('disp_pred1', disp_pred1_viz, global_step)

        if not all((loss_class, loss_motion)):
            print("{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".
                  format(running_loss_class, running_loss_motion, epoch, i))
            continue

        running_loss_bg_tc.update(loss_bg_tc)
        running_loss_fg_tc.update(loss_fg_tc)
        running_loss_sc.update(loss_sc)
        running_loss_disp.update(loss_disp)
        running_loss_class.update(loss_class)
        running_loss_motion.update(loss_motion)
        running_loss_odom.update(loss_odom)
        print("[{}/{}]\t{}, \t{}, \t{}, \t{}, \t{}, \t{}, \t{}".
              format(epoch, i, running_loss_disp, running_loss_class, running_loss_motion, running_loss_odom,
              running_loss_bg_tc, running_loss_sc, running_loss_fg_tc))

    return running_loss_disp, running_loss_class, running_loss_motion, running_loss_bg_tc, \
        running_loss_sc, running_loss_fg_tc, disp_pred, class_pred, motion_pred


# Compute and back-propagate the loss
def compute_and_bp_loss(optimizer, device, future_frames_num, pixel_radar_map_gt, pixel_lidar_map_gt, motion_gt, all_disp_field_gt, tf_gt,
                        criterion, class_pred, motion_pred, disp_pred, odom_pred, raw_radars, voxel_size, bp_loss):
    if bp_loss:
      optimizer.zero_grad()

    # ---------------------------------------------------------------------
    pixel_radar_map_gt = pixel_radar_map_gt.view(-1, 256, 256, cell_category_num)
    pixel_radar_map_gt = pixel_radar_map_gt.permute(0, 3, 1, 2).to(device)
    #print('pixel_radar_map_gt.shape:', pixel_radar_map_gt.shape) # torch.Size([bs, 2, 256, 256]) # non_empty_map
    pixel_lidar_map_gt = pixel_lidar_map_gt.to(device) # torch.Size([bs, 1, 256, 256])
    #print('pixel_lidar_map_gt.shape:', pixel_lidar_map_gt.shape)

    ### power thres
    power_thres_map = torch.clone(raw_radars[:,0,:,:,0])
    power_thres_map[power_thres_map>0.08] = 1
    power_thres_map[power_thres_map<=0.08] = 0

    pixel_radar_map_gt_thres = torch.clone(pixel_radar_map_gt)
    pixel_radar_map_gt_thres[:,0,:,:] = pixel_radar_map_gt_thres[:,0,:,:]*power_thres_map
    pixel_radar_map_gt_thres[:,1,:,:] = torch.logical_not(pixel_radar_map_gt_thres[:,0,:,:])

    pixel_lidar_map_gt_thres = torch.clone(pixel_lidar_map_gt)
    pixel_lidar_map_gt_thres[:,0,:,:] = pixel_lidar_map_gt_thres[:,0,:,:]*power_thres_map

#    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#    ax[0].imshow(power_thres_map[0].detach().cpu().numpy())
#    ax[1].imshow(pixel_radar_map_gt[0,0].detach().cpu().numpy())
#    ax[2].imshow(pixel_radar_map_gt_thres[0,0].detach().cpu().numpy())
#    plt.show()

    # ---------------------------------------------------------------------
    # -- Compute the grid cell classification loss
    if use_class_loss:

      log_softmax_probs = F.log_softmax(class_pred, dim=1)
      #print('log_softmax_probs.shape:', log_softmax_probs.shape)
      if use_weighted_loss:
          #map_shape = cat_weight_map.size()
          #cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
          #loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1) * cat_weight_map

          ### power weight
          power_weight_map = torch.clone(raw_radars[:,0,:,:,0])
          power_weight_mean = torch.mean(power_weight_map,dim=(1,2))
          power_weight_mean = torch.unsqueeze(torch.unsqueeze(power_weight_mean,1),2)
#          print(power_weight_mean.shape)
#          print(power_weight_mean)
#          fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#          ax[0].imshow(power_weight_map[0].detach().cpu().numpy())
#          ax[1].imshow((torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)*power_weight_map)[0].detach().cpu().numpy())
#          ax[2].imshow(torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)[0].detach().cpu().numpy())
#          plt.show()

          #loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1) # no weight
          #loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1) * power_weight_map / power_weight_mean  # weight by returned power
          loss_class = torch.sum(- pixel_radar_map_gt_thres * log_softmax_probs, dim=1) # use thres gt
      else:
          loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
      loss_class = torch.sum(loss_class) / (class_pred.shape[2]*class_pred.shape[3])
      loss_class_value = loss_class.item()
    else:
      loss_class_value = -1

    # ---------------------------------------------------------------------
    # -- Compute the motion loss
    if use_motion_loss:
      motion_gt_ = motion_gt.view(-1, 256, 256, 2)
      motion_gt_numpy = motion_gt_.numpy()
      motion_gt_ = motion_gt_.permute(0, 3, 1, 2).to(device)

      log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

      #valid_mask = pixel_radar_map_gt[:,0] # torch.Size([bs, 256, 256])
      valid_mask = pixel_radar_map_gt_thres[:,0]
      valid_mask = valid_mask.to(device)

      if use_weighted_loss:
          motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
          motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
          weight_vector = [1.0, 0.500]  # [moving, static]
          for k in range(len(weight_vector)):
              mask = motion_gt_numpy == (k + 1)
              motion_weight_map[mask] = weight_vector[k]

          motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
          loss_speed = torch.sum(- motion_gt_ * log_softmax_motion_pred, dim=1) * motion_weight_map
      else:
          loss_speed = torch.sum(- motion_gt_ * log_softmax_motion_pred, dim=1) # torch.Size([bs, 256, 256])

      #loss_motion = torch.sum(loss_speed) / (motion_pred.shape[2]*motion_pred.shape[3])
      loss_motion = torch.sum(loss_speed * valid_mask) / torch.nonzero(valid_mask).size(0)
      loss_motion_value = loss_motion.item()
    else:
      loss_motion_value = -1

    # ---------------------------------------------------------------------
    # -- Compute the displacement loss
    if use_disp_loss:
      all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, 256, 256, 2) # [1, future_num*bs, 256, 256, 2]
      gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous() # [1, future_num*bs, 256, 256, 2]
      gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4)) # [future_num*bs, 256, 256, 2]
      gt = gt.permute(0, 3, 1, 2).to(device) # [future_num*bs, 2, 256, 256]

      valid_mask = torch.from_numpy(np.zeros((gt.shape[0], 1, 256, 256))) # [future_num*bs, 1, 256, 256] # !!! only work when future_num = 1 !!!!!!!!!!!!!!!!
      #valid_mask[:,0] = pixel_lidar_map_gt[:,0] # pixel_radar_map_gt # torch.Size([bs, 256, 256])
      valid_mask[:,0] = pixel_lidar_map_gt_thres[:,0]
      motion_mask = torch.from_numpy(np.zeros((gt.shape[0], 1, 256, 256)))
      motion_mask[:,0] = motion_gt[:,0,:,:,0] # torch.Size([bs, 256, 256])

      #valid_mask = valid_mask * torch.logical_not(motion_mask) ###### ////////////////////////////////////////////////////////// Ablation test...
      valid_mask = valid_mask.to(device)

      #pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, cell_category_num)
      if use_weighted_loss:
          loss_disp = criterion(gt*valid_mask, disp_pred*valid_mask)
          axis_weight_map = torch.zeros_like(loss_disp, dtype=torch.float32)
          axis_weight_map[:,0,:,:] = 1 # right
          axis_weight_map[:,1,:,:] = 1 # 0.1 # top
          if torch.nonzero(valid_mask).size(0) != 0:
            loss_disp = torch.sum(loss_disp * axis_weight_map) / torch.nonzero(valid_mask).size(0)
            loss_disp_value = loss_disp.item()
            #print('loss_disp', loss_disp)
          else:
            print('no err')
            return -1,-1,-1,-1,-1,-1
      else:
          criterion = nn.SmoothL1Loss(reduction='sum')
          loss_disp = criterion(gt*valid_mask, disp_pred*valid_mask) / torch.nonzero(valid_mask).size(0)
    else:
      loss_disp_value = -1

    # ---------------------------------------------------------------------
    # -- Compute the Odometry loss
    if use_odom_loss:
      # TF gt
      #tf_gt_numpy = np.array(tf_gt)
      # disp gt
      all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, 256, 256, 2) # [1, future_num*bs, 256, 256, 2]
      gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous() # [1, future_num*bs, 256, 256, 2]
      gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4)) # [future_num*bs, 256, 256, 2]
      gt = gt.permute(0, 3, 1, 2).to(device) # [future_num*bs, 2, 256, 256]
      # gt mask for ransac
      valid_mask = torch.from_numpy(np.zeros((gt.shape[0], 1, 256, 256))) # [future_num*bs, 1, 256, 256] # !!! only work when future_num = 1 !!!!!!!!!!!!!!!!
      valid_mask[:,0] = pixel_lidar_map_gt[:,0] # pixel_radar_map_gt # torch.Size([bs, 256, 256])
      motion_mask = torch.from_numpy(np.zeros((gt.shape[0], 1, 256, 256)))
      motion_mask[:,0] = motion_gt[:,0,:,:,0] # torch.Size([bs, 256, 256])
      #valid_mask = valid_mask * torch.logical_not(motion_mask) # ///////////// only static

      if use_odom_net:
        #print('use_odom_net !!!')
        tf_gt = tf_gt.to(device)
        loss = nn.L1Loss(reduction='sum')
        loss_odom = loss(odom_pred, tf_gt)/tf_gt.shape[0]
        loss_odom_value = loss_odom.item()
      else:
        res_tf = torch.empty(0, 3)
        for i in range(tf_gt.shape[0]):
          M = calc_odom_by_disp_map(disp_pred[i,0].cpu().detach().numpy(), disp_pred[i,1].cpu().detach().numpy(), valid_mask[i,0].cpu().detach().numpy(), motion_mask[i,0], voxel_size[0])
          #M = calc_odom_by_disp_map(gt[i,0].cpu().detach().numpy(), gt[i,1].cpu().detach().numpy(), valid_mask[i,0].cpu().detach().numpy(), motion_mask[i,0], voxel_size[0])
          res_tf = torch.cat((res_tf, torch.tensor([[M[1,2], M[0,2], -np.arctan(M[1,0] / M[0,0])]]) ), dim=0)
        res_tf.requires_grad = True
        res_tf = res_tf.to(device)
        tf_gt = tf_gt.to(device)
        #print('res_tf', res_tf)
        loss = nn.L1Loss(reduction='sum')
        loss_odom = loss(res_tf, tf_gt)/tf_gt.shape[0]
        #print('loss_odom:', loss_odom)
        loss_odom_value = loss_odom.item()
    else:
      loss_odom_value = -1

    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    if use_class_loss and use_disp_loss and use_motion_loss:
      loss = loss_class + loss_motion + loss_disp
    elif use_class_loss and use_disp_loss and (not use_motion_loss):
      loss = loss_class + loss_disp
    elif use_class_loss and (not use_disp_loss) and use_motion_loss:
      loss = loss_class + loss_motion
    elif (not use_class_loss) and use_disp_loss and (not use_motion_loss):
      loss = loss_disp
    elif use_class_loss and (not use_disp_loss) and (not use_motion_loss):
      loss = loss_class
    else:
      loss = 0

    if use_odom_loss: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if loss==0:
        loss = loss_odom
      else:
        loss += loss_odom

    if bp_loss:
      loss.backward()
      optimizer.step()

    if use_bg_temporal_consistency:
        bg_tc_loss_value = bg_tc_loss.item()
    else:
        bg_tc_loss_value = -1

    if use_spatial_consistency or use_fg_temporal_consistency:
        sc_loss_value = instance_spatial_loss_value
        fg_tc_loss_value = instance_temporal_loss_value
    else:
        sc_loss_value = -1
        fg_tc_loss_value = -1

    return loss_class_value, loss_motion_value, loss_disp_value, loss_odom_value, bg_tc_loss_value, \
        sc_loss_value, fg_tc_loss_value

def gen_corr_line_set(src, dst, corres, color):
    viz_points = np.concatenate((src, dst), axis=1)
    viz_lines = list()
    for corr in corres:
      associate_shift = corr[1]-corr[0]
      viz_lines.append([corr[0],corr[0]+src.shape[1]])
    colors = [color for i in range(len(viz_lines))]
    line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(viz_points.T),
            lines=o3d.utility.Vector2iVector(viz_lines),
        )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def calc_odom_by_disp_map(disp0, disp1, radar_mask, moving_mask, cart_resolution):
  N=256
  center=N/2-0.5

#  pointcloud = list()
#  pointcloud_ = list()
#  for i in range(N): # row
#    for j in range(N): # col
#      if radar_mask[i,j]>0: # and moving_mask[i,j]<0:
#        point = np.array([center-i, j-center, 0]) # x, y in ego-motion frame
#        delta = np.array([disp1, disp0, 0])
#        point_ = point + delta
#        pointcloud.append(point)
#        pointcloud_.append(point_)
#  pc = np.array(pointcloud)
#  pc_ = np.array(pointcloud_)

  radar_mask[radar_mask>0] = 1
  radar_mask[radar_mask<0] = 0
  valid_mask = radar_mask.astype(np.bool) # (N x N)

  N_vec = np.arange(N).reshape(1,-1)
  c_mat = np.full((N, N), center)

  # x, y w.r.t. ego-motion frame
  point_x = c_mat - N_vec.T
  point_y = -(c_mat - N_vec.T).T
  point_x = np.expand_dims(point_x, axis=2)
  point_y = np.expand_dims(point_y, axis=2)
  point = np.concatenate((point_x, point_y), axis=2) # (N x N x 2)

  disp_map = np.zeros((256,256,2)) # (N x N x 2)
  disp_map[:,:,0] = disp1
  disp_map[:,:,1] = disp0

  point_trans = point + disp_map # (N x N x 2)

  point = point.reshape(-1, 2) # (N^2 x 2)
  point_trans = point_trans.reshape(-1, 2) # (N^2 x 2)
  valid_mask = valid_mask.reshape(-1, 1) # (N^2 x 1)

  # mask by valid mask
  point = np.delete(point, np.where(valid_mask==0), axis=0)
  point_trans = np.delete(point_trans, np.where(valid_mask==0), axis=0)

  pc = np.concatenate((point, np.zeros((point.shape[0], 1))), axis=1)
  pc_ = np.concatenate((point_trans, np.zeros((point_trans.shape[0], 1))), axis=1)
#  print(pc.shape)
#  print(pc_.shape)
#  pcd = o3d.geometry.PointCloud()
#  pcd.points = o3d.utility.Vector3dVector(pc)
#  pcd.paint_uniform_color([1, 0, 0])
#  pcd_ = o3d.geometry.PointCloud()
#  pcd_.points = o3d.utility.Vector3dVector(pc_)
#  pcd_.paint_uniform_color([0, 1, 0])
#  arr = np.expand_dims(np.arange(pc.shape[0]),axis=0)
#  np_corres = np.concatenate((arr, arr), axis=0).T
#  corres = o3d.utility.Vector2iVector(np_corres)
#  line_set = gen_corr_line_set(pc.T, pc_.T, corres, [0,0,1])
#  o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  M, mask = cv2.findHomography(point, point_trans, cv2.RANSAC, 3.0) # 1~10 => strict~loose # result tf is w.r.t. disp frame
#  print(M[1,2], M[0,2], -np.arctan(M[1,0]/M[0,0])) # w.r.t. disp frame
#  np_corres_new = np_corres[mask.squeeze().astype(np.bool),:]
#  corres_new = o3d.utility.Vector2iVector(np_corres_new)
#  line_set = gen_corr_line_set(pc.T, pc_.T, corres_new, [0,0,1])
#  o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])

  M[0,2] = M[0,2]*cart_resolution
  M[1,2] = M[1,2]*cart_resolution
  return M




def background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices):
    """
    disp_pred: Should be relative displacement between adjacent frames. shape (batch * 2, sweep_num, 2, h, w)
    pixel_cat_map_gt: Shape (batch, 2, h, w, cat_num)
    non_empty_map: Shape (batch, 2, h, w)
    trans_matrices: Shape (batch, 2, sweep_num, 4, 4)
    """
    #print('trans_matrices.shape')
    #print(trans_matrices.shape) # torch.Size([2, 1, 25, 4, 4])

    criterion = nn.SmoothL1Loss(reduction='sum')

    non_empty_map_numpy = non_empty_map.numpy()
    pixel_cat_maps = pixel_cat_map_gt.numpy()
    max_prob = np.amax(pixel_cat_maps, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_maps = np.argmax(pixel_cat_maps, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_maps = (pixel_cat_maps * non_empty_map_numpy * filter_mask)  # (batch, 2, h, w)

    trans_matrices = trans_matrices.numpy()
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    seq_1_absolute_pred_list = list()
    seq_2_absolute_pred_list = list()

    seq_1_absolute_pred_list.append(seq_1_pred[:, 1])
    for i in range(2, pred_shape[1]):
        seq_1_absolute_pred_list.append(seq_1_pred[:, i] + seq_1_absolute_pred_list[i - 2])

    seq_2_absolute_pred_list.append(seq_2_pred[:, 0])
    for i in range(1, pred_shape[1] - 1):
        seq_2_absolute_pred_list.append(seq_2_pred[:, i] + seq_2_absolute_pred_list[i - 1])

    # ----------------- Compute the consistency loss -----------------
    # Compute the transformation matrices
    # First, transform the coordinate
    transformed_disp_pred_list = list()

    trans_matrix_global = trans_matrices[:, 1]  # (batch, sweep_num, 4, 4)
    trans_matrix_global = trans_matrix_global[:, trans_matrix_idx, 0:3]  # (batch, 3, 4)  # <---
    trans_matrix_global = trans_matrix_global[:, :, (0, 1, 3)]  # (batch, 3, 3)
    trans_matrix_global[:, 2] = np.array([0.0, 0.0, 1.0])

    # --- Move pixel coord to global and rescale; then rotate; then move back to local pixel coord
    translate_to_global = np.array([[1.0, 0.0, -120.0], [0.0, 1.0, -120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale_global = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    trans_global = scale_global @ translate_to_global
    inv_trans_global = np.linalg.inv(trans_global)

    trans_global = np.expand_dims(trans_global, axis=0)
    inv_trans_global = np.expand_dims(inv_trans_global, axis=0)
    trans_matrix_total = inv_trans_global @ trans_matrix_global @ trans_global

    # --- Generate grid transformation matrix, so as to use Pytorch affine_grid and grid_sample function
    w, h = pred_shape[-2], pred_shape[-1]
    resize_m = np.array([
        [2 / w, 0.0, -1],
        [0.0, 2 / h, -1],
        [0.0, 0.0, 1]
    ], dtype=np.float32)
    inverse_m = np.linalg.inv(resize_m)
    resize_m = np.expand_dims(resize_m, axis=0)
    inverse_m = np.expand_dims(inverse_m, axis=0)

    grid_trans_matrix = resize_m @ trans_matrix_total @ inverse_m  # (batch, 3, 3)
    grid_trans_matrix = grid_trans_matrix[:, :2].astype(np.float32)
    grid_trans_matrix = torch.from_numpy(grid_trans_matrix)

    # --- For displacement field
    trans_matrix_translation_global = np.eye(trans_matrix_total.shape[1])
    trans_matrix_translation_global = np.expand_dims(trans_matrix_translation_global, axis=0)
    trans_matrix_translation_global = np.repeat(trans_matrix_translation_global, grid_trans_matrix.shape[0], axis=0)
    trans_matrix_translation_global[:, :, 2] = trans_matrix_global[:, :, 2]  # only translation
    trans_matrix_translation_total = inv_trans_global @ trans_matrix_translation_global @ trans_global

    grid_trans_matrix_disp = resize_m @ trans_matrix_translation_total @ inverse_m
    grid_trans_matrix_disp = grid_trans_matrix_disp[:, :2].astype(np.float32)
    grid_trans_matrix_disp = torch.from_numpy(grid_trans_matrix_disp).to(device)

    disp_rotate_matrix = trans_matrix_global[:, 0:2, 0:2].astype(np.float32)  # (batch, 2, 2)
    disp_rotate_matrix = torch.from_numpy(disp_rotate_matrix).to(device)

    for i in range(len(seq_1_absolute_pred_list)):

        # --- Start transformation for displacement field
        curr_pred = seq_1_absolute_pred_list[i]  # (batch, 2, h, w)

        # First, rotation
        curr_pred = curr_pred.permute(0, 2, 3, 1).contiguous()  # (batch, h, w, 2)
        curr_pred = curr_pred.view(-1, h * w, 2)
        curr_pred = torch.bmm(curr_pred, disp_rotate_matrix)
        curr_pred = curr_pred.view(-1, h, w, 2)
        curr_pred = curr_pred.permute(0, 3, 1, 2).contiguous()  # (batch, 2, h, w)

        # Next, translation
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()  # swap x and y axis
        curr_pred = torch.flip(curr_pred, dims=[2])

        grid = F.affine_grid(grid_trans_matrix_disp, curr_pred.size())
        if use_nn_sampling:
            curr_pred = F.grid_sample(curr_pred, grid, mode='nearest')
        else:
            curr_pred = F.grid_sample(curr_pred, grid)

        curr_pred = torch.flip(curr_pred, dims=[2])
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()

        transformed_disp_pred_list.append(curr_pred)

    # --- Start transformation for category map
    pixel_cat_map = pixel_cat_maps[:, 0]  # (batch, h, w)
    pixel_cat_map = torch.from_numpy(pixel_cat_map.astype(np.float32))
    pixel_cat_map = pixel_cat_map[:, None, :, :]  # (batch, 1, h, w)
    trans_pixel_cat_map = pixel_cat_map.permute(0, 1, 3, 2)  # (batch, 1, h, w), swap x and y axis
    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])

    grid = F.affine_grid(grid_trans_matrix, pixel_cat_map.size())
    trans_pixel_cat_map = F.grid_sample(trans_pixel_cat_map, grid, mode='nearest')

    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])
    trans_pixel_cat_map = trans_pixel_cat_map.permute(0, 1, 3, 2)

    # --- Compute the loss, using smooth l1 loss
    adj_pixel_cat_map = pixel_cat_maps[:, 1]
    adj_pixel_cat_map = torch.from_numpy(adj_pixel_cat_map.astype(np.float32))
    adj_pixel_cat_map = torch.unsqueeze(adj_pixel_cat_map, dim=1)

    mask_common = trans_pixel_cat_map == adj_pixel_cat_map
    mask_common = mask_common.float()
    non_empty_map_gpu = non_empty_map.to(device)
    non_empty_map_gpu = non_empty_map_gpu[:, 1:2, :, :]  # select the second sequence, keep dim
    mask_common = mask_common.to(device)
    mask_common = mask_common * non_empty_map_gpu

    loss_list = list()
    for i in range(len(seq_1_absolute_pred_list)):
        trans_seq_1_pred = transformed_disp_pred_list[i]  # (batch, 2, h, w)
        seq_2_pred = seq_2_absolute_pred_list[i]  # (batch, 2, h, w)

        trans_seq_1_pred = trans_seq_1_pred * mask_common
        seq_2_pred = seq_2_pred * mask_common

        num_non_empty_cells = torch.nonzero(mask_common).size(0)
        if num_non_empty_cells != 0:
            loss = criterion(trans_seq_1_pred, seq_2_pred) / num_non_empty_cells
            loss_list.append(loss)

    res_loss = torch.mean(torch.stack(loss_list, 0))

    return res_loss


# We name it instance spatial-temporal consistency loss because it involves each instance
def instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map):
    print('--instance_spatial_temporal_consistency_loss--')
    device = disp_pred.device
    pred_shape = disp_pred.size() # [bs*2, 20, 2, 256, 256]
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4]) # [bs, 2, 20, 2, 256, 256]

    seq_1_pred = disp_pred[:, 0] # [bs, 20, 2, 256, 256]
    seq_2_pred = disp_pred[:, 1] # [bs, 20, 2, 256, 256]

    pixel_instance_map = pixel_instance_map.numpy() # [bs, 2, 256, 256]
    batch = pixel_instance_map.shape[0]

    spatial_loss = 0.0
    temporal_loss = 0.0
    counter = 0
    criterion = nn.SmoothL1Loss()

    for i in range(batch):
        curr_batch_instance_maps = pixel_instance_map[i] # (2, 256, 256)

        seq_1_instance_map = curr_batch_instance_maps[0] # (256, 256)
        seq_2_instance_map = curr_batch_instance_maps[1]

        seq_1_instance_ids = np.unique(seq_1_instance_map)
        seq_2_instance_ids = np.unique(seq_2_instance_map)
        print(seq_1_instance_ids.shape)
        print(seq_1_instance_ids)

        common_instance_ids = np.intersect1d(seq_1_instance_ids, seq_2_instance_ids, assume_unique=True)

        seq_1_batch_pred = seq_1_pred[i]  # (sweep_num, 2, h, w)
        seq_2_batch_pred = seq_2_pred[i]

        for h in common_instance_ids: ## Only compute common_instance_ids for spatial consistency is inaccurate
            if h == 0:  # do not consider the background instance
                continue

            print('h:', h)
            seq_1_mask = np.where(seq_1_instance_map == h)
            seq_1_idx_x = torch.from_numpy(seq_1_mask[0]).to(device)
            seq_1_idx_y = torch.from_numpy(seq_1_mask[1]).to(device)
            seq_1_selected_cells = seq_1_batch_pred[:, :, seq_1_idx_x, seq_1_idx_y]

            seq_2_mask = np.where(seq_2_instance_map == h)
            seq_2_idx_x = torch.from_numpy(seq_2_mask[0]).to(device)
            seq_2_idx_y = torch.from_numpy(seq_2_mask[1]).to(device)
            seq_2_selected_cells = seq_2_batch_pred[:, :, seq_2_idx_x, seq_2_idx_y]

            seq_1_selected_cell_num = seq_1_selected_cells.size(2)
            seq_2_selected_cell_num = seq_2_selected_cells.size(2)

            # for spatial loss
            if use_spatial_consistency:
                tmp_seq_1 = 0
                if seq_1_selected_cell_num > 1:
                    tmp_seq_1 = criterion(seq_1_selected_cells[:, :, :-1], seq_1_selected_cells[:, :, 1:]) ## This is not the best way to calc spatial consistency error

                tmp_seq_2 = 0
                if seq_2_selected_cell_num > 1:
                    tmp_seq_2 = criterion(seq_2_selected_cells[:, :, :-1], seq_2_selected_cells[:, :, 1:])

                spatial_loss += tmp_seq_1 + tmp_seq_2

            if use_fg_temporal_consistency:
                seq_1_mean = torch.mean(seq_1_selected_cells, dim=2)
                seq_2_mean = torch.mean(seq_2_selected_cells, dim=2)
                temporal_loss += criterion(seq_1_mean, seq_2_mean)

            counter += 1

    if counter != 0:
        spatial_loss = spatial_loss / counter
        temporal_loss = temporal_loss / counter

    total_loss = reg_weight_sc * spatial_loss + reg_weight_fg_tc * temporal_loss

    spatial_loss_value = 0 if type(spatial_loss) == float else spatial_loss.item()
    temporal_loss_value = 0 if type(temporal_loss) == float else temporal_loss.item()

    return total_loss, spatial_loss_value, temporal_loss_value


if __name__ == "__main__":
    main()
