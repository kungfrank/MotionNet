# python3 no ros

# ins/ro frame:
#   x -> top
#   y -> right
# disp frame:
#   disp0, x -> right
#   disp1, y -> top
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import pandas as pd
import time
import os
import numpy as np
from numpy.linalg import inv
from datetime import datetime
import time
import math

from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import open3d as o3d
import copy
from oxford_model import RaMNet
from matplotlib.colors import hsv_to_rgb
import matplotlib

import rospy
from sensor_msgs.msg import Image
import sys
import os
pid = os.getpid()
print('pid:', pid)

save = 1
disp = 0

which_model = 'RaMNet'
trained_model_path = '/mnt/Disk1/trained_model/train_multi_seq/2021-04-30_13-16-22/epoch_15.pth' # 2021-03-23_23-22-37/epoch_10.pth'
#trained_model_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-04-13_23-33-39/epoch_50.pth' # 2021-03-23_23-22-37/epoch_10.pth'

img_save_dir = '/mnt/Disk1/viz_result/itri/thres0.15_seg_ransac3_viz_sparse'
#img_save_dir = '/mnt/Disk1/viz_result/oxford/front_thres0.1'


history_scan_num = 2
num_past_frames = 2
out_seq_len = 1
height_feat_size=1
use_temporal_info = True

plot_motion_seg = True
global radar_buffer, ax

if save:
  matplotlib.use('Agg')
if disp:
  #fig, ax = plt.subplots(1, 2, figsize=(22, 11))
  fig, ax = plt.subplots(1, 1, figsize=(22, 11))
  plt.tight_layout()

radar_buffer = []

##### Load Model #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained network weights
loaded_models = list()
if which_model == "RaMNet":
    model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
    model = nn.DataParallel(model)
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'], False)
    model = model.to(device)
    loaded_models = [model]
else:
    print('model error')
print("Loaded pretrained model {}".format(which_model))

model = loaded_models[0]
model.eval()

def callback(data):
  global radar_buffer, ax
  print(data.header.stamp)

  t1 = time.time()
  radar_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width)
  radar_img = radar_img.astype(np.float32)/255.

  #print(radar_img.shape)
  #radar_img = radar_img[:256,256:256*2]
  #print(radar_img.shape)

  radar_img[radar_img<0.15]=0

  radar_buffer.append(radar_img)
  if len(radar_buffer)<num_past_frames:
    return
  if len(radar_buffer)>num_past_frames:
    radar_buffer.pop(0)

  assert len(radar_buffer) == num_past_frames

  raw_radars = list()
  for i in range(num_past_frames):
    raw_radars.append(np.expand_dims(radar_buffer[num_past_frames-i-1], axis=2))
  raw_radars = np.stack(raw_radars, 0).astype(np.float32)
  raw_radars_list = list()
  raw_radars_list.append(raw_radars)
  raw_radars = np.stack(raw_radars_list, 0)
  raw_radars = torch.tensor(raw_radars)
  raw_radar = raw_radars[0,0,:,:,0]
  t2 = time.time()
  print('preprocessing time cost:',t2-t1)

  t1 = time.time()
  with torch.no_grad():
    if use_temporal_info:
      disp_pred, cat_pred, motion_pred = model(raw_radars)
    else:
      raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 256, 256, 1)).astype(np.float32))
      raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
      raw_radars_curr = raw_radars_curr.to(device)
      disp_pred, cat_pred, motion_pred = model(raw_radars_curr)
    #print('disp_pred:',disp_pred.shape)
  t2 = time.time()
  print('inference time cost:',t2-t1)
  # convert all output to numpy
  cat_pred_numpy = cat_pred.cpu().numpy()
  motion_pred_numpy = motion_pred.cpu().numpy()
  disp_pred_numpy = disp_pred.cpu().numpy()
  raw_radars = raw_radars.detach().numpy()

  t1 = time.time()
  if save or disp:
    # visualize network output #
    viz_cat_pred = cat_pred_numpy[0,0,:,:]
    viz_cat_pred[viz_cat_pred>0] = 1
    viz_cat_pred[viz_cat_pred<=0] = 0

    viz_motion_pred = motion_pred_numpy[0,0,:,:]
    viz_motion_pred[viz_motion_pred>0] = 1
    viz_motion_pred[viz_motion_pred<=0] = 0
    viz_motion_pred = viz_motion_pred * viz_cat_pred

    M, outlier_mask = calc_odom_by_disp_map(disp_pred_numpy[0,0], disp_pred_numpy[0,1], cat_pred_numpy[0,0,:,:], motion_pred_numpy[0,0,:,:])
    viz_motion_pred = outlier_mask

    if save:
      fig, ax = plt.subplots(1, 2, figsize=(22, 11)) # 20 7

    t1 = time.time()
    ax[0].imshow(viz_combined(raw_radar*2., viz_cat_pred, viz_motion_pred))
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    #ax.imshow(raw_radar, cmap='gray')
    ax[1].imshow(radar_img*2., cmap='gray', vmax=1., vmin=0.)
    ax[1].axis('off')
    ax[1].set_aspect('equal')
    #ax.title.set_text('Result')
    t2 = time.time()
    print('plotting time cost:',t2-t1)
    qk1, qk2 = plot_quiver(ax[1], -disp_pred_numpy[0,0], -disp_pred_numpy[0,1], viz_cat_pred, viz_motion_pred)

#    t1 = time.time()
#    raw_radar_viz_img = np.zeros((256,256,3))
#    raw_radar_viz_img = np.stack((raw_radar,raw_radar,raw_radar), axis=2)

#    disp_pred_img = flow_to_img(disp_pred_numpy[0,0,...], -disp_pred_numpy[0,1,...])
#    disp_pred_viz = (disp_pred_img+raw_radar_viz_img)/1
#    disp_pred_viz[disp_pred_viz>1.] = 1.
#    ax[1].imshow(disp_pred_viz)
#    ax[1].axis('off')
#    ax[1].set_aspect('equal')
#    ax[1].title.set_text('Flow')
#  t2 = time.time()
#  print('plotting time cost:',t2-t1)

  if disp:
    plt.pause(0.001)
  if save:
    t1 = time.time()
    plt.savefig(os.path.join(img_save_dir, str(data.header.stamp) + '.png'), bbox_inches='tight')
    #cv2.imwrite(img_save_dir+'/opticalflow/'+str(count)+'.png', (opticalflow_viz*255).astype(np.uint8))
    #cv2.imwrite(img_save_dir+'/predflow/'+str(count)+'.png', (disp_pred_viz*255).astype(np.uint8))
    t2 = time.time()
    print('saving time cost:',t2-t1)
    plt.close(fig)
  if save or disp:
    ax[1].clear
    #ax[1].clear
    qk1.remove()
    qk2.remove()




def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/Navtech/Cartesian", Image, callback, queue_size=100000)
    #rospy.Subscriber("/radar_cart", Image, callback, queue_size=100000)
    plt.show()
    rospy.spin()

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
def calc_odom_by_disp_map(disp0, disp1, radar_mask, moving_mask):
  disp_map = np.zeros((256,256,2))
  disp_map[:,:,0] = disp0
  disp_map[:,:,1] = disp1

  radar_mask[radar_mask>0]=1
  radar_mask[radar_mask<=0]=0

  pointcloud = list()
  pointcloud_ = list()
  N=256
  center=N/2-0.5
  for i in range(N): # row
    for j in range(N): # col
      if radar_mask[i,j]==1:
        point = np.array([center-i, j-center, 0]) # x, y in ego-motion frame
        delta = np.array([disp_map[i,j,1], disp_map[i,j,0], 0])
        point_ = point + delta
        pointcloud.append(point)
        pointcloud_.append(point_)
  pc = np.array(pointcloud)
  pc_ = np.array(pointcloud_)
  #print(pc.shape)
  #print(pc_.shape)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pc)
  pcd.paint_uniform_color([1, 0, 0])
  pcd_ = o3d.geometry.PointCloud()
  pcd_.points = o3d.utility.Vector3dVector(pc_)
  pcd_.paint_uniform_color([0, 1, 0])
  arr = np.expand_dims(np.arange(pc.shape[0]),axis=0)
  np_corres = np.concatenate((arr, arr), axis=0).T
  corres = o3d.utility.Vector2iVector(np_corres)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  M, mask = cv2.findHomography(pc, pc_, cv2.RANSAC, 3.0) # 1~10 => strict~loose 3.0

  # gen outlier mask
  outlier_mask = np.copy(radar_mask).astype(np.bool)
  pc_inlier = np.delete(pc, np.where(mask==0), axis=0)
  for point in pc_inlier:
    i = (center-point[0]).astype(np.int)
    j = (center+point[1]).astype(np.int)
    outlier_mask[i,j] = 0

  #print(M[1,2], M[0,2], -np.arctan(M[1,0]/M[0,0])) # w.r.t. disp frame
  np_corres_new = np_corres[mask.squeeze().astype(np.bool),:]
  corres_new = o3d.utility.Vector2iVector(np_corres_new)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres_new, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  return M, outlier_mask

def viz_combined(img, denoised_img, motion_seg):
  viz_img = np.zeros((256,256,3))
  viz_img = np.stack((img,img,img), axis=2)
  viz_denoised_img = np.zeros((256,256,3))
  if plot_motion_seg:
    viz_denoised_img[:,:,2] = denoised_img * np.logical_not(motion_seg)
    viz_seg = np.zeros((256,256,3))
    viz_seg[:,:,0] = motion_seg
    return (viz_img*1.+viz_denoised_img+viz_seg)/2.
  else:
    viz_denoised_img[:,:,2] = denoised_img
    return (viz_img*1.+viz_denoised_img)/2.


def plot_quiver(ax_, disp0, disp1, viz_cat, viz_motion):
  # Plot quiver.
  field_gt = np.zeros((256,256,2))
  field_gt[:,:,0] = disp0
  field_gt[:,:,1] = disp1
  idx_x = np.arange(field_gt.shape[0])
  idx_y = np.arange(field_gt.shape[1])
  idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
  # For cells with very small movements, we threshold them to be static
  field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
  #thd_mask = field_gt_norm <= 0.1
  #field_gt[thd_mask, :] = 0
  # Get the displacement field

  row_sparse_mask = np.zeros((256,256)).astype(np.bool)
  col_sparse_mask = np.zeros((256,256)).astype(np.bool)

  row_sparse_mask[np.arange(0,255,3)] = 1
  col_sparse_mask[:,np.arange(0,255,3)] = 1
  sparse_mask = row_sparse_mask * col_sparse_mask

  mask = viz_cat.astype(np.bool) * viz_motion.astype(np.bool) * sparse_mask
  print(mask.shape)
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  #qk1 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=30, headaxislength=30, color='r', alpha=0.9, minlength=10, minshaft=1) #'g'
  qk1 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0011, headwidth=10, headlength=10, headaxislength=10, color='r', alpha=0.9, minlength=3, minshaft=1)

  mask = viz_cat.astype(np.bool) * np.logical_not(viz_motion.astype(np.bool)) * sparse_mask
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  #qk2 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=20, headaxislength=20, color=[(0.1,0.15,1.0)], alpha=0.9, minlength=10, minshaft=1)
  # dodgerblue
  qk2 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0011, headwidth=10, headlength=10, headaxislength=10, color=[(0.1,0.15,1.0)], alpha=0.9, minlength=3, minshaft=1)
  return qk1, qk2


def flow_to_img(flow_x, flow_y):
  hsv = np.zeros((flow_x.shape[0],flow_x.shape[1],3)).astype(np.float32)
  hsv[...,1] = 1
  #mag, ang = cv2.cartToPolar(flow_x, flow_y)
  mag = np.sqrt(np.power(flow_x,2)+np.power(flow_y,2))
  ang = np.arctan2(flow_y, flow_x)
  ang[ang<0] = ang[ang<0]+np.pi*2
  mag[mag>=15]=15 # upper bound
  hsv[...,0] = ang/np.pi/2
  hsv[...,2] = mag/15.
  rgb = hsv_to_rgb(hsv)
  return rgb


if __name__ == '__main__':
    listener()





